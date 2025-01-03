//! GPU mesh preprocessing.
//!
//! This is an optional pass that uses a compute shader to reduce the amount of
//! data that has to be transferred from the CPU to the GPU. When enabled,
//! instead of transferring [`MeshUniform`]s to the GPU, we transfer the smaller
//! [`MeshInputUniform`]s instead and use the GPU to calculate the remaining
//! derived fields in [`MeshUniform`].

use core::num::NonZero;

use bevy_app::{App, Plugin};
use bevy_asset::{load_internal_asset, Handle};
use bevy_core_pipeline::core_3d::graph::Core3d;
use bevy_ecs::{
    component::Component,
    entity::Entity,
    query::{Has, QueryState, Without},
    schedule::{common_conditions::resource_exists, IntoSystemConfigs as _},
    system::{lifetimeless::Read, Commands, Res, ResMut, Resource},
    world::{FromWorld, World},
};
use bevy_render::{
    batching::gpu_preprocessing::{
        BatchedInstanceBuffers, GpuPreprocessingSupport, IndirectBatchSet,
        IndirectParametersBuffers, IndirectParametersIndexed, IndirectParametersMetadata,
        IndirectParametersNonIndexed, PreprocessWorkItem, PreprocessWorkItemBuffer,
    },
    render_graph::{Node, NodeRunError, RenderGraphApp, RenderGraphContext},
    render_resource::{
        binding_types::{storage_buffer, storage_buffer_read_only, uniform_buffer},
        BindGroup, BindGroupEntries, BindGroupLayout, BindingResource, Buffer, BufferBinding,
        CachedComputePipelineId, ComputePassDescriptor, ComputePipelineDescriptor,
        DynamicBindGroupLayoutEntries, PipelineCache, Shader, ShaderStages, ShaderType,
        SpecializedComputePipeline, SpecializedComputePipelines,
    },
    renderer::{RenderContext, RenderDevice, RenderQueue},
    settings::WgpuFeatures,
    view::{NoIndirectDrawing, ViewUniform, ViewUniformOffset, ViewUniforms},
    Render, RenderApp, RenderSet,
};
use bevy_utils::tracing::warn;
use bitflags::bitflags;
use smallvec::{smallvec, SmallVec};

use crate::{
    graph::NodePbr, MeshCullingData, MeshCullingDataBuffer, MeshInputUniform, MeshUniform,
};

/// The handle to the `mesh_preprocess.wgsl` compute shader.
pub const MESH_PREPROCESS_SHADER_HANDLE: Handle<Shader> =
    Handle::weak_from_u128(16991728318640779533);
/// The handle to the `mesh_preprocess_types.wgsl` compute shader.
pub const MESH_PREPROCESS_TYPES_SHADER_HANDLE: Handle<Shader> =
    Handle::weak_from_u128(2720440370122465935);
pub const BUILD_INDIRECT_PARAMS_SHADER_HANDLE: Handle<Shader> =
    Handle::weak_from_u128(3711077208359699672);

/// The GPU workgroup size.
const WORKGROUP_SIZE: usize = 64;

/// A plugin that builds mesh uniforms on GPU.
///
/// This will only be added if the platform supports compute shaders (e.g. not
/// on WebGL 2).
pub struct GpuMeshPreprocessPlugin {
    /// Whether we're building [`MeshUniform`]s on GPU.
    ///
    /// This requires compute shader support and so will be forcibly disabled if
    /// the platform doesn't support those.
    pub use_gpu_instance_buffer_builder: bool,
}

/// The render node for the mesh uniform building pass.
pub struct GpuPreprocessNode {
    view_query: QueryState<
        (
            Entity,
            Read<PreprocessBindGroups>,
            Read<ViewUniformOffset>,
            Has<NoIndirectDrawing>,
        ),
        Without<SkipGpuPreprocess>,
    >,
}

pub struct BuildIndirectParametersNode {
    view_query: QueryState<
        Read<PreprocessBindGroups>,
        (Without<SkipGpuPreprocess>, Without<NoIndirectDrawing>),
    >,
}

/// The compute shader pipelines for the mesh uniform building pass.
#[derive(Resource)]
pub struct PreprocessPipelines {
    /// The pipeline used for CPU culling. This pipeline doesn't populate
    /// indirect parameters.
    pub direct_preprocess: PreprocessPipeline,
    /// The pipeline used for GPU culling. This pipeline populates indirect
    /// parameters.
    pub gpu_culling_preprocess: PreprocessPipeline,
    pub build_indexed_indirect_params: BuildIndirectParametersPipeline,
    pub build_non_indexed_indirect_params: BuildIndirectParametersPipeline,
}

/// The pipeline for the GPU mesh preprocessing shader.
pub struct PreprocessPipeline {
    /// The bind group layout for the compute shader.
    pub bind_group_layout: BindGroupLayout,
    /// The pipeline ID for the compute shader.
    ///
    /// This gets filled in `prepare_preprocess_pipelines`.
    pub pipeline_id: Option<CachedComputePipelineId>,
}

pub struct BuildIndirectParametersPipeline {
    /// The bind group layout for the compute shader.
    pub bind_group_layout: BindGroupLayout,
    /// The pipeline ID for the compute shader.
    ///
    /// This gets filled in `prepare_preprocess_pipelines`.
    pub pipeline_id: Option<CachedComputePipelineId>,
}

bitflags! {
    /// Specifies variants of the mesh preprocessing shader.
    #[derive(Clone, Copy, PartialEq, Eq, Hash)]
    pub struct PreprocessPipelineKey: u8 {
        /// Whether GPU culling is in use.
        ///
        /// This `#define`'s `GPU_CULLING` in the shader.
        const GPU_CULLING = 1;
    }

    #[derive(Clone, Copy, PartialEq, Eq, Hash)]
    pub struct BuildIndirectParametersPipelineKey: u8 {
        const INDEXED = 1;
        const MULTI_DRAW_INDIRECT_COUNT_SUPPORTED = 2;
    }
}

/// The compute shader bind group for the mesh uniform building pass.
///
/// This goes on the view.
#[derive(Component, Clone)]
pub enum PreprocessBindGroups {
    Direct(BindGroup),
    Indirect {
        indexed: Option<BindGroup>,
        non_indexed: Option<BindGroup>,
    },
}

#[derive(Resource)]
pub struct BuildIndirectParametersBindGroups {
    indexed: Option<BindGroup>,
    non_indexed: Option<BindGroup>,
}

/// Stops the `GpuPreprocessNode` attempting to generate the buffer for this view
/// useful to avoid duplicating effort if the bind group is shared between views
#[derive(Component)]
pub struct SkipGpuPreprocess;

impl Plugin for GpuMeshPreprocessPlugin {
    fn build(&self, app: &mut App) {
        load_internal_asset!(
            app,
            MESH_PREPROCESS_SHADER_HANDLE,
            "mesh_preprocess.wgsl",
            Shader::from_wgsl
        );
        load_internal_asset!(
            app,
            MESH_PREPROCESS_TYPES_SHADER_HANDLE,
            "mesh_preprocess_types.wgsl",
            Shader::from_wgsl
        );
        load_internal_asset!(
            app,
            BUILD_INDIRECT_PARAMS_SHADER_HANDLE,
            "build_indirect_params.wgsl",
            Shader::from_wgsl
        );
    }

    fn finish(&self, app: &mut App) {
        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };

        // This plugin does nothing if GPU instance buffer building isn't in
        // use.
        let gpu_preprocessing_support = render_app.world().resource::<GpuPreprocessingSupport>();
        if !self.use_gpu_instance_buffer_builder || !gpu_preprocessing_support.is_available() {
            return;
        }

        render_app
            .init_resource::<PreprocessPipelines>()
            .init_resource::<SpecializedComputePipelines<PreprocessPipeline>>()
            .init_resource::<SpecializedComputePipelines<BuildIndirectParametersPipeline>>()
            .add_systems(
                Render,
                (
                    prepare_preprocess_pipelines.in_set(RenderSet::Prepare),
                    prepare_preprocess_bind_groups
                        .run_if(
                            resource_exists::<BatchedInstanceBuffers<MeshUniform, MeshInputUniform>>,
                        )
                        .in_set(RenderSet::PrepareBindGroups),
                    write_mesh_culling_data_buffer.in_set(RenderSet::PrepareResourcesFlush),
                )
            )
            .add_render_graph_node::<GpuPreprocessNode>(Core3d, NodePbr::GpuPreprocess)
            .add_render_graph_node::<BuildIndirectParametersNode>(Core3d, NodePbr::BuildIndirectParametersNode)
            .add_render_graph_edges(
                Core3d,
                (NodePbr::GpuPreprocess, NodePbr::BuildIndirectParametersNode, NodePbr::ShadowPass)
            );
    }
}

impl FromWorld for GpuPreprocessNode {
    fn from_world(world: &mut World) -> Self {
        Self {
            view_query: QueryState::new(world),
        }
    }
}

impl Node for GpuPreprocessNode {
    fn update(&mut self, world: &mut World) {
        self.view_query.update_archetypes(world);
    }

    fn run<'w>(
        &self,
        _: &mut RenderGraphContext,
        render_context: &mut RenderContext<'w>,
        world: &'w World,
    ) -> Result<(), NodeRunError> {
        // Grab the [`BatchedInstanceBuffers`].
        let BatchedInstanceBuffers {
            work_item_buffers: ref index_buffers,
            ..
        } = world.resource::<BatchedInstanceBuffers<MeshUniform, MeshInputUniform>>();

        let pipeline_cache = world.resource::<PipelineCache>();
        let preprocess_pipelines = world.resource::<PreprocessPipelines>();

        let mut compute_pass =
            render_context
                .command_encoder()
                .begin_compute_pass(&ComputePassDescriptor {
                    label: Some("mesh preprocessing"),
                    timestamp_writes: None,
                });

        // Run the compute passes.
        for (view, bind_groups, view_uniform_offset, no_indirect_drawing) in
            self.view_query.iter_manual(world)
        {
            // Grab the index buffer for this view.
            let Some(index_buffer) = index_buffers.get(&view) else {
                warn!("The preprocessing index buffer wasn't present");
                continue;
            };

            // Select the right pipeline, depending on whether GPU culling is in
            // use.
            let maybe_pipeline_id = if !no_indirect_drawing {
                preprocess_pipelines.gpu_culling_preprocess.pipeline_id
            } else {
                preprocess_pipelines.direct_preprocess.pipeline_id
            };

            // Fetch the pipeline.
            let Some(preprocess_pipeline_id) = maybe_pipeline_id else {
                warn!("The build mesh uniforms pipeline wasn't ready");
                return Ok(());
            };

            let Some(preprocess_pipeline) =
                pipeline_cache.get_compute_pipeline(preprocess_pipeline_id)
            else {
                // This will happen while the pipeline is being compiled and is fine.
                return Ok(());
            };

            compute_pass.set_pipeline(preprocess_pipeline);

            let mut dynamic_offsets: SmallVec<[u32; 1]> = smallvec![];
            if !no_indirect_drawing {
                dynamic_offsets.push(view_uniform_offset.offset);
            }

            match (bind_groups, index_buffer) {
                (
                    PreprocessBindGroups::Direct(bind_group),
                    PreprocessWorkItemBuffer::Direct(work_item_buffer),
                ) => {
                    compute_pass.set_bind_group(0, bind_group, &dynamic_offsets);
                    let workgroup_count = work_item_buffer.len().div_ceil(WORKGROUP_SIZE);
                    if workgroup_count > 0 {
                        compute_pass.dispatch_workgroups(workgroup_count as u32, 1, 1);
                    }
                }

                (
                    PreprocessBindGroups::Indirect {
                        indexed: maybe_indexed_bind_group,
                        non_indexed: maybe_non_indexed_bind_group,
                    },
                    PreprocessWorkItemBuffer::Indirect {
                        indexed: indexed_buffer,
                        non_indexed: non_indexed_buffer,
                    },
                ) => {
                    if let Some(indexed_bind_group) = maybe_indexed_bind_group {
                        compute_pass.set_bind_group(0, indexed_bind_group, &dynamic_offsets);
                        let workgroup_count = indexed_buffer.len().div_ceil(WORKGROUP_SIZE);
                        if workgroup_count > 0 {
                            compute_pass.dispatch_workgroups(workgroup_count as u32, 1, 1);
                        }
                    }

                    if let Some(non_indexed_bind_group) = maybe_non_indexed_bind_group {
                        compute_pass.set_bind_group(0, non_indexed_bind_group, &dynamic_offsets);
                        let workgroup_count = non_indexed_buffer.len().div_ceil(WORKGROUP_SIZE);
                        if workgroup_count > 0 {
                            compute_pass.dispatch_workgroups(workgroup_count as u32, 1, 1);
                        }
                    }
                }

                (_, _) => {}
            }
        }

        Ok(())
    }
}

impl FromWorld for BuildIndirectParametersNode {
    fn from_world(world: &mut World) -> Self {
        Self {
            view_query: QueryState::new(world),
        }
    }
}

impl Node for BuildIndirectParametersNode {
    fn update(&mut self, world: &mut World) {
        self.view_query.update_archetypes(world);
    }

    fn run<'w>(
        &self,
        _: &mut RenderGraphContext,
        render_context: &mut RenderContext<'w>,
        world: &'w World,
    ) -> Result<(), NodeRunError> {
        let Some(build_indirect_params_bind_groups) =
            world.get_resource::<BuildIndirectParametersBindGroups>()
        else {
            return Ok(());
        };

        let pipeline_cache = world.resource::<PipelineCache>();
        let preprocess_pipelines = world.resource::<PreprocessPipelines>();
        let indirect_parameters_buffers = world.resource::<IndirectParametersBuffers>();

        let mut compute_pass =
            render_context
                .command_encoder()
                .begin_compute_pass(&ComputePassDescriptor {
                    label: Some("build indirect parameters"),
                    timestamp_writes: None,
                });

        // Run the compute passes.
        let (maybe_indexed_pipeline_id, maybe_non_indexed_pipeline_id) = (
            preprocess_pipelines
                .build_indexed_indirect_params
                .pipeline_id,
            preprocess_pipelines
                .build_non_indexed_indirect_params
                .pipeline_id,
        );

        // Fetch the pipeline.
        let (
            Some(build_indexed_indirect_params_pipeline_id),
            Some(build_non_indexed_indirect_params_pipeline_id),
        ) = (maybe_indexed_pipeline_id, maybe_non_indexed_pipeline_id)
        else {
            warn!("The build indirect parameters pipelines weren't ready");
            return Ok(());
        };

        let (
            Some(build_indexed_indirect_params_pipeline),
            Some(build_non_indexed_indirect_params_pipeline),
        ) = (
            pipeline_cache.get_compute_pipeline(build_indexed_indirect_params_pipeline_id),
            pipeline_cache.get_compute_pipeline(build_non_indexed_indirect_params_pipeline_id),
        )
        else {
            // This will happen while the pipeline is being compiled and is fine.
            return Ok(());
        };

        // Build indexed indirect parameters.
        if let Some(ref build_indirect_indexed_params_bind_group) =
            build_indirect_params_bind_groups.indexed
        {
            compute_pass.set_pipeline(build_indexed_indirect_params_pipeline);
            compute_pass.set_bind_group(0, build_indirect_indexed_params_bind_group, &[]);
            let workgroup_count = indirect_parameters_buffers
                .indexed_len()
                .div_ceil(WORKGROUP_SIZE);
            if workgroup_count > 0 {
                compute_pass.dispatch_workgroups(workgroup_count as u32, 1, 1);
            }
        }

        // Build non-indexed indirect parameters.
        if let Some(ref build_indirect_non_indexed_params_bind_group) =
            build_indirect_params_bind_groups.non_indexed
        {
            compute_pass.set_pipeline(build_non_indexed_indirect_params_pipeline);
            compute_pass.set_bind_group(0, build_indirect_non_indexed_params_bind_group, &[]);
            let workgroup_count = indirect_parameters_buffers
                .non_indexed_len()
                .div_ceil(WORKGROUP_SIZE);
            if workgroup_count > 0 {
                compute_pass.dispatch_workgroups(workgroup_count as u32, 1, 1);
            }
        }

        Ok(())
    }
}

impl PreprocessPipelines {
    pub(crate) fn pipelines_are_loaded(&self, pipeline_cache: &PipelineCache) -> bool {
        self.direct_preprocess.is_loaded(pipeline_cache)
            && self.gpu_culling_preprocess.is_loaded(pipeline_cache)
    }
}

impl PreprocessPipeline {
    fn is_loaded(&self, pipeline_cache: &PipelineCache) -> bool {
        self.pipeline_id
            .is_some_and(|pipeline_id| pipeline_cache.get_compute_pipeline(pipeline_id).is_some())
    }
}

impl SpecializedComputePipeline for PreprocessPipeline {
    type Key = PreprocessPipelineKey;

    fn specialize(&self, key: Self::Key) -> ComputePipelineDescriptor {
        let mut shader_defs = vec![];
        if key.contains(PreprocessPipelineKey::GPU_CULLING) {
            shader_defs.push("INDIRECT".into());
            shader_defs.push("FRUSTUM_CULLING".into());
        }

        ComputePipelineDescriptor {
            label: Some(
                format!(
                    "mesh preprocessing ({})",
                    if key.contains(PreprocessPipelineKey::GPU_CULLING) {
                        "GPU culling"
                    } else {
                        "direct"
                    }
                )
                .into(),
            ),
            layout: vec![self.bind_group_layout.clone()],
            push_constant_ranges: vec![],
            shader: MESH_PREPROCESS_SHADER_HANDLE,
            shader_defs,
            entry_point: "main".into(),
            zero_initialize_workgroup_memory: false,
        }
    }
}

impl FromWorld for PreprocessPipelines {
    fn from_world(world: &mut World) -> Self {
        let render_device = world.resource::<RenderDevice>();

        // GPU culling bind group parameters are a superset of those in the CPU
        // culling (direct) shader.
        let direct_bind_group_layout_entries = preprocess_direct_bind_group_layout_entries();
        let gpu_culling_bind_group_layout_entries = preprocess_direct_bind_group_layout_entries()
            .extend_sequential((
                // `indirect_parameters_metadata`
                storage_buffer::<IndirectParametersMetadata>(/* has_dynamic_offset= */ false),
                // `mesh_culling_data`
                storage_buffer_read_only::<MeshCullingData>(/* has_dynamic_offset= */ false),
                // `view`
                uniform_buffer::<ViewUniform>(/* has_dynamic_offset= */ true),
            ));

        let build_indexed_indirect_params_bind_group_layout_entries =
            build_indirect_params_bind_group_layout_entries()
                .extend_sequential((storage_buffer::<IndirectParametersIndexed>(false),));
        let build_non_indexed_indirect_params_bind_group_layout_entries =
            build_indirect_params_bind_group_layout_entries()
                .extend_sequential((storage_buffer::<IndirectParametersNonIndexed>(false),));

        let direct_bind_group_layout = render_device.create_bind_group_layout(
            "build mesh uniforms direct bind group layout",
            &direct_bind_group_layout_entries,
        );
        let gpu_culling_bind_group_layout = render_device.create_bind_group_layout(
            "build mesh uniforms GPU culling bind group layout",
            &gpu_culling_bind_group_layout_entries,
        );
        let build_indexed_indirect_params_bind_group_layout = render_device
            .create_bind_group_layout(
                "build indexed indirect parameters bind group layout",
                &build_indexed_indirect_params_bind_group_layout_entries,
            );
        let build_non_indexed_indirect_params_bind_group_layout = render_device
            .create_bind_group_layout(
                "build non-indexed indirect parameters bind group layout",
                &build_non_indexed_indirect_params_bind_group_layout_entries,
            );

        PreprocessPipelines {
            direct_preprocess: PreprocessPipeline {
                bind_group_layout: direct_bind_group_layout,
                pipeline_id: None,
            },
            gpu_culling_preprocess: PreprocessPipeline {
                bind_group_layout: gpu_culling_bind_group_layout,
                pipeline_id: None,
            },
            build_indexed_indirect_params: BuildIndirectParametersPipeline {
                bind_group_layout: build_indexed_indirect_params_bind_group_layout,
                pipeline_id: None,
            },
            build_non_indexed_indirect_params: BuildIndirectParametersPipeline {
                bind_group_layout: build_non_indexed_indirect_params_bind_group_layout,
                pipeline_id: None,
            },
        }
    }
}

fn preprocess_direct_bind_group_layout_entries() -> DynamicBindGroupLayoutEntries {
    DynamicBindGroupLayoutEntries::sequential(
        ShaderStages::COMPUTE,
        (
            // `current_input`
            storage_buffer_read_only::<MeshInputUniform>(false),
            // `previous_input`
            storage_buffer_read_only::<MeshInputUniform>(false),
            // `indices`
            storage_buffer_read_only::<PreprocessWorkItem>(false),
            // `output`
            storage_buffer::<MeshUniform>(false),
        ),
    )
}

fn build_indirect_params_bind_group_layout_entries() -> DynamicBindGroupLayoutEntries {
    DynamicBindGroupLayoutEntries::sequential(
        ShaderStages::COMPUTE,
        (
            storage_buffer_read_only::<MeshInputUniform>(false),
            storage_buffer_read_only::<IndirectParametersMetadata>(false),
            storage_buffer::<IndirectBatchSet>(false),
        ),
    )
}

/// A system that specializes the `mesh_preprocess.wgsl` pipelines if necessary.
pub fn prepare_preprocess_pipelines(
    pipeline_cache: Res<PipelineCache>,
    render_device: Res<RenderDevice>,
    mut specialized_preprocess_pipelines: ResMut<SpecializedComputePipelines<PreprocessPipeline>>,
    mut specialized_build_indirect_parameters_pipelines: ResMut<
        SpecializedComputePipelines<BuildIndirectParametersPipeline>,
    >,
    mut preprocess_pipelines: ResMut<PreprocessPipelines>,
) {
    preprocess_pipelines.direct_preprocess.prepare(
        &pipeline_cache,
        &mut specialized_preprocess_pipelines,
        PreprocessPipelineKey::empty(),
    );
    preprocess_pipelines.gpu_culling_preprocess.prepare(
        &pipeline_cache,
        &mut specialized_preprocess_pipelines,
        PreprocessPipelineKey::GPU_CULLING,
    );

    let mut build_indirect_parameters_pipeline_key = BuildIndirectParametersPipelineKey::empty();
    if render_device
        .wgpu_device()
        .features()
        .contains(WgpuFeatures::MULTI_DRAW_INDIRECT_COUNT)
    {
        build_indirect_parameters_pipeline_key
            .insert(BuildIndirectParametersPipelineKey::MULTI_DRAW_INDIRECT_COUNT_SUPPORTED);
    }

    preprocess_pipelines.build_indexed_indirect_params.prepare(
        &pipeline_cache,
        &mut specialized_build_indirect_parameters_pipelines,
        build_indirect_parameters_pipeline_key | BuildIndirectParametersPipelineKey::INDEXED,
    );
    preprocess_pipelines
        .build_non_indexed_indirect_params
        .prepare(
            &pipeline_cache,
            &mut specialized_build_indirect_parameters_pipelines,
            build_indirect_parameters_pipeline_key,
        );
}

impl PreprocessPipeline {
    fn prepare(
        &mut self,
        pipeline_cache: &PipelineCache,
        pipelines: &mut SpecializedComputePipelines<PreprocessPipeline>,
        key: PreprocessPipelineKey,
    ) {
        if self.pipeline_id.is_some() {
            return;
        }

        let preprocess_pipeline_id = pipelines.specialize(pipeline_cache, self, key);
        self.pipeline_id = Some(preprocess_pipeline_id);
    }
}

impl SpecializedComputePipeline for BuildIndirectParametersPipeline {
    type Key = BuildIndirectParametersPipelineKey;

    fn specialize(&self, key: Self::Key) -> ComputePipelineDescriptor {
        let mut shader_defs = vec![];
        if key.contains(BuildIndirectParametersPipelineKey::INDEXED) {
            shader_defs.push("INDEXED".into());
        }
        if key.contains(BuildIndirectParametersPipelineKey::MULTI_DRAW_INDIRECT_COUNT_SUPPORTED) {
            shader_defs.push("MULTI_DRAW_INDIRECT_COUNT_SUPPORTED".into());
        }

        ComputePipelineDescriptor {
            label: if key.contains(BuildIndirectParametersPipelineKey::INDEXED) {
                Some("build indexed indirect parameters".into())
            } else {
                Some("build non-indexed indirect parameters".into())
            },
            layout: vec![self.bind_group_layout.clone()],
            push_constant_ranges: vec![],
            shader: BUILD_INDIRECT_PARAMS_SHADER_HANDLE,
            shader_defs,
            entry_point: "main".into(),
            zero_initialize_workgroup_memory: false,
        }
    }
}

impl BuildIndirectParametersPipeline {
    fn prepare(
        &mut self,
        pipeline_cache: &PipelineCache,
        pipelines: &mut SpecializedComputePipelines<BuildIndirectParametersPipeline>,
        key: BuildIndirectParametersPipelineKey,
    ) {
        if self.pipeline_id.is_some() {
            return;
        }

        let build_indirect_parameters_pipeline_id = pipelines.specialize(pipeline_cache, self, key);
        self.pipeline_id = Some(build_indirect_parameters_pipeline_id);
    }
}

/// A system that attaches the mesh uniform buffers to the bind groups for the
/// variants of the mesh preprocessing compute shader.
pub fn prepare_preprocess_bind_groups(
    mut commands: Commands,
    render_device: Res<RenderDevice>,
    batched_instance_buffers: Res<BatchedInstanceBuffers<MeshUniform, MeshInputUniform>>,
    indirect_parameters_buffers: Res<IndirectParametersBuffers>,
    mesh_culling_data_buffer: Res<MeshCullingDataBuffer>,
    view_uniforms: Res<ViewUniforms>,
    pipelines: Res<PreprocessPipelines>,
) {
    // Grab the `BatchedInstanceBuffers`.
    let BatchedInstanceBuffers {
        data_buffer: ref data_buffer_vec,
        ref work_item_buffers,
        current_input_buffer: ref current_input_buffer_vec,
        previous_input_buffer: ref previous_input_buffer_vec,
    } = batched_instance_buffers.into_inner();

    let (Some(current_input_buffer), Some(previous_input_buffer), Some(data_buffer)) = (
        current_input_buffer_vec.buffer().buffer(),
        previous_input_buffer_vec.buffer().buffer(),
        data_buffer_vec.buffer(),
    ) else {
        return;
    };

    let mut any_indirect = false;

    for (view, work_item_buffer_vec) in work_item_buffers {
        let bind_groups = match *work_item_buffer_vec {
            PreprocessWorkItemBuffer::Direct(ref work_item_buffer_vec) => {
                let Some(work_item_buffer) = work_item_buffer_vec.buffer() else {
                    continue;
                };

                // Don't use `as_entire_binding()` here; the shader reads the array
                // length and the underlying buffer may be longer than the actual size
                // of the vector.
                let work_item_buffer_size = NonZero::<u64>::try_from(
                    work_item_buffer_vec.len() as u64 * u64::from(PreprocessWorkItem::min_size()),
                )
                .ok();

                PreprocessBindGroups::Direct(render_device.create_bind_group(
                    "preprocess_direct_bind_group",
                    &pipelines.direct_preprocess.bind_group_layout,
                    &BindGroupEntries::sequential((
                        current_input_buffer.as_entire_binding(),
                        previous_input_buffer.as_entire_binding(),
                        BindingResource::Buffer(BufferBinding {
                            buffer: work_item_buffer,
                            offset: 0,
                            size: work_item_buffer_size,
                        }),
                        data_buffer.as_entire_binding(),
                    )),
                ))
            }

            PreprocessWorkItemBuffer::Indirect {
                indexed: ref indexed_buffer,
                non_indexed: ref non_indexed_buffer,
            } => {
                let (Some(mesh_culling_data_buffer), Some(view_uniforms_binding)) = (
                    mesh_culling_data_buffer.buffer(),
                    view_uniforms.uniforms.binding(),
                ) else {
                    continue;
                };

                let indexed_bind_group = match (
                    indexed_buffer.buffer(),
                    indirect_parameters_buffers.indexed_metadata_buffer(),
                ) {
                    (
                        Some(indexed_work_item_buffer),
                        Some(indexed_indirect_parameters_metadata_buffer),
                    ) => {
                        // Don't use `as_entire_binding()` here; the shader reads the array
                        // length and the underlying buffer may be longer than the actual size
                        // of the vector.
                        let indexed_work_item_buffer_size = NonZero::<u64>::try_from(
                            indexed_buffer.len() as u64 * u64::from(PreprocessWorkItem::min_size()),
                        )
                        .ok();
                        Some(render_device.create_bind_group(
                            "preprocess_indexed_indirect_gpu_culling_bind_group",
                            &pipelines.gpu_culling_preprocess.bind_group_layout,
                            &BindGroupEntries::sequential((
                                current_input_buffer.as_entire_binding(),
                                previous_input_buffer.as_entire_binding(),
                                BindingResource::Buffer(BufferBinding {
                                    buffer: indexed_work_item_buffer,
                                    offset: 0,
                                    size: indexed_work_item_buffer_size,
                                }),
                                data_buffer.as_entire_binding(),
                                indexed_indirect_parameters_metadata_buffer.as_entire_binding(),
                                mesh_culling_data_buffer.as_entire_binding(),
                                view_uniforms_binding.clone(),
                            )),
                        ))
                    }
                    _ => None,
                };

                let non_indexed_bind_group = match (
                    non_indexed_buffer.buffer(),
                    indirect_parameters_buffers.non_indexed_metadata_buffer(),
                ) {
                    (
                        Some(non_indexed_work_item_buffer),
                        Some(non_indexed_indirect_parameters_metadata_buffer),
                    ) => {
                        // Don't use `as_entire_binding()` here; the shader reads the array
                        // length and the underlying buffer may be longer than the actual size
                        // of the vector.
                        let non_indexed_work_item_buffer_size = NonZero::<u64>::try_from(
                            non_indexed_buffer.len() as u64
                                * u64::from(PreprocessWorkItem::min_size()),
                        )
                        .ok();
                        Some(render_device.create_bind_group(
                            "preprocess_non_indexed_indirect_gpu_culling_bind_group",
                            &pipelines.gpu_culling_preprocess.bind_group_layout,
                            &BindGroupEntries::sequential((
                                current_input_buffer.as_entire_binding(),
                                previous_input_buffer.as_entire_binding(),
                                BindingResource::Buffer(BufferBinding {
                                    buffer: non_indexed_work_item_buffer,
                                    offset: 0,
                                    size: non_indexed_work_item_buffer_size,
                                }),
                                data_buffer.as_entire_binding(),
                                non_indexed_indirect_parameters_metadata_buffer.as_entire_binding(),
                                mesh_culling_data_buffer.as_entire_binding(),
                                view_uniforms_binding,
                            )),
                        ))
                    }
                    _ => None,
                };

                any_indirect = true;

                PreprocessBindGroups::Indirect {
                    indexed: indexed_bind_group,
                    non_indexed: non_indexed_bind_group,
                }
            }
        };

        commands.entity(*view).insert(bind_groups);
    }

    if any_indirect {
        create_build_indirect_parameters_bind_groups(
            &mut commands,
            &render_device,
            &pipelines,
            current_input_buffer,
            &indirect_parameters_buffers,
        );
    }
}

fn create_build_indirect_parameters_bind_groups(
    commands: &mut Commands,
    render_device: &RenderDevice,
    pipelines: &PreprocessPipelines,
    current_input_buffer: &Buffer,
    indirect_parameters_buffer: &IndirectParametersBuffers,
) {
    commands.insert_resource(BuildIndirectParametersBindGroups {
        indexed: match (
            indirect_parameters_buffer.indexed_metadata_buffer(),
            indirect_parameters_buffer.indexed_data_buffer(),
            indirect_parameters_buffer.indexed_batch_sets_buffer(),
        ) {
            (
                Some(indexed_indirect_parameters_metadata_buffer),
                Some(indexed_indirect_parameters_data_buffer),
                Some(indexed_batch_sets_buffer),
            ) => Some(render_device.create_bind_group(
                "build_indexed_indirect_parameters_bind_group",
                &pipelines.build_indexed_indirect_params.bind_group_layout,
                &BindGroupEntries::sequential((
                    current_input_buffer.as_entire_binding(),
                    indexed_indirect_parameters_metadata_buffer.as_entire_binding(),
                    indexed_batch_sets_buffer.as_entire_binding(),
                    indexed_indirect_parameters_data_buffer.as_entire_binding(),
                )),
            )),
            _ => None,
        },
        non_indexed: match (
            indirect_parameters_buffer.non_indexed_metadata_buffer(),
            indirect_parameters_buffer.non_indexed_data_buffer(),
            indirect_parameters_buffer.non_indexed_batch_sets_buffer(),
        ) {
            (
                Some(non_indexed_indirect_parameters_metadata_buffer),
                Some(non_indexed_indirect_parameters_data_buffer),
                Some(non_indexed_batch_sets_buffer),
            ) => Some(
                render_device.create_bind_group(
                    "build_non_indexed_indirect_parameters_bind_group",
                    &pipelines
                        .build_non_indexed_indirect_params
                        .bind_group_layout,
                    &BindGroupEntries::sequential((
                        current_input_buffer.as_entire_binding(),
                        non_indexed_indirect_parameters_metadata_buffer.as_entire_binding(),
                        non_indexed_batch_sets_buffer.as_entire_binding(),
                        non_indexed_indirect_parameters_data_buffer.as_entire_binding(),
                    )),
                ),
            ),
            _ => None,
        },
    });
}

/// Writes the information needed to do GPU mesh culling to the GPU.
pub fn write_mesh_culling_data_buffer(
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
    mut mesh_culling_data_buffer: ResMut<MeshCullingDataBuffer>,
) {
    mesh_culling_data_buffer.write_buffer(&render_device, &render_queue);
}
