//! GPU mesh preprocessing.
//!
//! This is an optional pass that uses a compute shader to reduce the amount of
//! data that has to be transferred from the CPU to the GPU. When enabled,
//! instead of transferring [`MeshUniform`]s to the GPU, we transfer the smaller
//! [`MeshInputUniform`]s instead and use the GPU to calculate the remaining
//! derived fields in [`MeshUniform`].

use std::num::NonZeroU64;

use bevy_app::{App, Plugin};
use bevy_asset::{load_internal_asset, Handle};
use bevy_core_pipeline::core_3d::graph::{Core3d, Node3d};
use bevy_ecs::{
    component::Component,
    entity::Entity,
    query::{Has, QueryState},
    schedule::{common_conditions::resource_exists, IntoSystemConfigs as _},
    system::{lifetimeless::Read, Commands, Res, ResMut, Resource},
    world::{FromWorld, World},
};
use bevy_render::{
    batching::gpu_preprocessing::{
        BatchedInstanceBuffers, CullingBuffers, GpuPreprocessingSupport, IndirectParameters,
        IndirectParametersBuffers, PreprocessWorkItem,
    },
    camera::OcclusionCulling,
    render_graph::{Node, NodeRunError, RenderGraphApp, RenderGraphContext},
    render_resource::{
        binding_types::{storage_buffer, storage_buffer_read_only, texture_2d, uniform_buffer},
        BindGroup, BindGroupEntries, BindGroupLayout, BindingResource, BufferBinding,
        CachedComputePipelineId, ComputePassDescriptor, ComputePipelineDescriptor,
        DynamicBindGroupLayoutEntries, IntoBinding, PipelineCache, Shader, ShaderStages,
        ShaderType, SpecializedComputePipeline, SpecializedComputePipelines, TextureSampleType,
    },
    renderer::{RenderContext, RenderDevice, RenderQueue},
    view::{GpuCulling, ViewUniform, ViewUniformOffset, ViewUniforms},
    Render, RenderApp, RenderSet,
};
use bevy_utils::tracing::warn;
use bitflags::bitflags;
use smallvec::{smallvec, SmallVec};

use crate::{
    graph::NodePbr, MeshCullingData, MeshCullingDataBuffer, MeshInputUniform, MeshUniform,
    RenderHiZViewResources,
};

/// The handle to the `mesh_preprocess.wgsl` compute shader.
pub const MESH_PREPROCESS_SHADER_HANDLE: Handle<Shader> =
    Handle::weak_from_u128(16991728318640779533);

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

type GpuPreprocessViewQuery = QueryState<(
    Entity,
    Read<PreprocessBindGroups>,
    Read<ViewUniformOffset>,
    Has<GpuCulling>,
    Has<OcclusionCulling>,
)>;

/// The render node for the mesh uniform building pass.
pub struct EarlyGpuPreprocessNode {
    view_query: GpuPreprocessViewQuery,
}

pub struct LateGpuPreprocessNode {
    view_query: GpuPreprocessViewQuery,
}

/// The compute shader pipelines for the mesh uniform building pass.
#[derive(Resource)]
pub struct PreprocessPipelines {
    /// The pipeline used for CPU culling. This pipeline doesn't populate
    /// indirect parameters.
    pub direct: PreprocessPipeline,
    /// The pipeline used for GPU frustum culling. This pipeline populates
    /// indirect parameters.
    pub gpu_frustum_culling: PreprocessPipeline,
    /// The pipeline used for GPU occlusion culling. This pipeline populates two
    /// sets of indirect parameters.
    pub gpu_occlusion_culling: PreprocessPipeline,
}

/// The pipeline for the GPU mesh preprocessing shader.
pub struct PreprocessPipeline {
    /// The bind group layout for the compute shader.
    pub bind_group_layout: BindGroupLayout,
    /// The pipeline ID for the compute shader.
    ///
    /// This gets filled in in `prepare_preprocess_pipelines`.
    pub pipeline_id: Option<CachedComputePipelineId>,
}

bitflags! {
    /// Specifies variants of the mesh preprocessing shader.
    #[derive(Clone, Copy, PartialEq, Eq, Hash)]
    pub struct PreprocessPipelineKey: u8 {
        /// Whether GPU frustum culling is in use.
        ///
        /// This `#define`'s `FRUSTUM_CULLING` in the shader.
        const GPU_FRUSTUM_CULLING = 1;

        const GPU_OCCLUSION_CULLING = 2;
    }
}

/// The compute shader bind group for the mesh uniform building pass.
///
/// This goes on the view.
#[derive(Component)]
pub struct PreprocessBindGroups {
    early: BindGroup,
    late: Option<BindGroup>,
}

impl Plugin for GpuMeshPreprocessPlugin {
    fn build(&self, app: &mut App) {
        load_internal_asset!(
            app,
            MESH_PREPROCESS_SHADER_HANDLE,
            "mesh_preprocess.wgsl",
            Shader::from_wgsl
        );

        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };

        render_app
            .add_render_graph_node::<EarlyGpuPreprocessNode>(Core3d, NodePbr::EarlyGpuPreprocess)
            .add_render_graph_node::<LateGpuPreprocessNode>(Core3d, NodePbr::LateGpuPreprocess);
    }

    fn finish(&self, app: &mut App) {
        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };

        // This plugin does nothing if GPU instance buffer building isn't in
        // use.
        let gpu_preprocessing_support = render_app.world().resource::<GpuPreprocessingSupport>();
        if !self.use_gpu_instance_buffer_builder
            || *gpu_preprocessing_support == GpuPreprocessingSupport::None
        {
            return;
        }

        // Stitch the node in.
        render_app
            .add_render_graph_edges(Core3d, (NodePbr::EarlyGpuPreprocess, Node3d::EarlyPrepass))
            .add_render_graph_edges(Core3d, (Node3d::EarlyPrepass, NodePbr::LateGpuPreprocess, Node3d::LatePrepass))
            .add_render_graph_edges(Core3d, (NodePbr::EarlyGpuPreprocess, NodePbr::ShadowPass))
            .init_resource::<PreprocessPipelines>()
            .init_resource::<SpecializedComputePipelines<PreprocessPipeline>>()
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
            );
    }
}

impl FromWorld for EarlyGpuPreprocessNode {
    fn from_world(world: &mut World) -> Self {
        Self {
            view_query: QueryState::new(world),
        }
    }
}

impl Node for EarlyGpuPreprocessNode {
    fn update(&mut self, world: &mut World) {
        self.view_query.update_archetypes(world);
    }

    fn run<'w>(
        &self,
        _: &mut RenderGraphContext,
        render_context: &mut RenderContext<'w>,
        world: &'w World,
    ) -> Result<(), NodeRunError> {
        run_gpu_preprocess_node(false, &self.view_query, render_context, world)
    }
}

impl FromWorld for LateGpuPreprocessNode {
    fn from_world(world: &mut World) -> Self {
        Self {
            view_query: QueryState::new(world),
        }
    }
}

impl Node for LateGpuPreprocessNode {
    fn update(&mut self, world: &mut World) {
        self.view_query.update_archetypes(world);
    }

    fn run<'w>(
        &self,
        _: &mut RenderGraphContext,
        render_context: &mut RenderContext<'w>,
        world: &'w World,
    ) -> Result<(), NodeRunError> {
        run_gpu_preprocess_node(true, &self.view_query, render_context, world)
    }
}

fn run_gpu_preprocess_node<'w>(
    late: bool,
    view_query: &GpuPreprocessViewQuery,
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
                label: if late {
                    Some("late mesh preprocessing")
                } else {
                    Some("early mesh preprocessing")
                },
                timestamp_writes: None,
            });

    // Run the compute passes.
    for (view, bind_groups, view_uniform_offset, gpu_culling, occlusion_culling) in
        view_query.iter_manual(world)
    {
        // Grab the index buffer for this view.
        let Some(index_buffer) = index_buffers.get(&view) else {
            warn!("The preprocessing index buffer wasn't present");
            return Ok(());
        };

        // Select the appropriate pipeline, depending on the GPU culling mode.
        let maybe_pipeline_id = match (gpu_culling, occlusion_culling) {
            (true, true) => preprocess_pipelines.gpu_occlusion_culling.pipeline_id,
            (true, false) => preprocess_pipelines.gpu_frustum_culling.pipeline_id,
            (false, _) => preprocess_pipelines.direct.pipeline_id,
        };

        // Fetch the pipeline.
        let Some(preprocess_pipeline_id) = maybe_pipeline_id else {
            warn!("The build mesh uniforms pipeline wasn't ready");
            return Ok(());
        };

        let Some(preprocess_pipeline) = pipeline_cache.get_compute_pipeline(preprocess_pipeline_id)
        else {
            // This will happen while the pipeline is being compiled and is fine.
            return Ok(());
        };

        compute_pass.set_pipeline(preprocess_pipeline);

        let mut dynamic_offsets: SmallVec<[u32; 1]> = smallvec![];
        if gpu_culling {
            dynamic_offsets.push(view_uniform_offset.offset);
        }

        match bind_groups.late {
            Some(ref late_bind_group) if late => {
                compute_pass.set_bind_group(0, late_bind_group, &dynamic_offsets);
            }
            _ => compute_pass.set_bind_group(0, &bind_groups.early, &dynamic_offsets),
        }

        let workgroup_count = index_buffer.work_item_buffer.len().div_ceil(WORKGROUP_SIZE);
        compute_pass.dispatch_workgroups(workgroup_count as u32, 1, 1);
    }

    Ok(())
}

impl PreprocessPipelines {
    pub(crate) fn pipelines_are_loaded(&self, pipeline_cache: &PipelineCache) -> bool {
        self.direct.is_loaded(pipeline_cache) && self.gpu_frustum_culling.is_loaded(pipeline_cache)
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

        if key.contains(PreprocessPipelineKey::GPU_FRUSTUM_CULLING) {
            shader_defs.push("INDIRECT".into());
            shader_defs.push("FRUSTUM_CULLING".into());

            if key.contains(PreprocessPipelineKey::GPU_OCCLUSION_CULLING) {
                shader_defs.push("OCCLUSION_CULLING".into());
            }
        }

        ComputePipelineDescriptor {
            label: Some(
                format!(
                    "mesh preprocessing ({})",
                    if !key.contains(PreprocessPipelineKey::GPU_FRUSTUM_CULLING) {
                        "direct"
                    } else if !key.contains(PreprocessPipelineKey::GPU_OCCLUSION_CULLING) {
                        "GPU occlusion culling"
                    } else {
                        "GPU frustum culling"
                    }
                )
                .into(),
            ),
            layout: vec![self.bind_group_layout.clone()],
            push_constant_ranges: vec![],
            shader: MESH_PREPROCESS_SHADER_HANDLE,
            shader_defs,
            entry_point: "main".into(),
        }
    }
}

impl FromWorld for PreprocessPipelines {
    fn from_world(world: &mut World) -> Self {
        let render_device = world.resource::<RenderDevice>();

        // GPU frustum culling bind group parameters are a superset of those in
        // the CPU culling (direct) shader.
        let direct_bind_group_layout_entries = preprocess_direct_bind_group_layout_entries();
        let gpu_frustum_culling_bind_group_layout_entries =
            preprocess_gpu_frustum_culling_bind_group_layout_entries();
        let gpu_occlusion_culling_bind_group_layout_entries =
            preprocess_gpu_occlusion_culling_bind_group_layout_entries();

        let direct_bind_group_layout = render_device.create_bind_group_layout(
            "mesh preprocessing direct bind group layout",
            &direct_bind_group_layout_entries,
        );
        let gpu_frustum_culling_bind_group_layout = render_device.create_bind_group_layout(
            "mesh preprocessing GPU frustum culling bind group layout",
            &gpu_frustum_culling_bind_group_layout_entries,
        );
        let gpu_occlusion_culling_bind_group_layout = render_device.create_bind_group_layout(
            "mesh preprocessing GPU occlusion culling bind group layout",
            &gpu_occlusion_culling_bind_group_layout_entries,
        );

        PreprocessPipelines {
            direct: PreprocessPipeline {
                bind_group_layout: direct_bind_group_layout,
                pipeline_id: None,
            },
            gpu_frustum_culling: PreprocessPipeline {
                bind_group_layout: gpu_frustum_culling_bind_group_layout,
                pipeline_id: None,
            },
            gpu_occlusion_culling: PreprocessPipeline {
                bind_group_layout: gpu_occlusion_culling_bind_group_layout,
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
            // `main_pass_output`
            storage_buffer::<MeshUniform>(false),
        ),
    )
}

/// Creates the bind group layout entries for GPU frustum culling.
fn preprocess_gpu_frustum_culling_bind_group_layout_entries() -> DynamicBindGroupLayoutEntries {
    preprocess_direct_bind_group_layout_entries().extend_sequential((
        // FIXME: What about indirect mode with no frustum culling?
        // There should probably be a new bind group layout entry, right?
        // `main_pass_indirect_parameters`
        storage_buffer::<IndirectParameters>(/*has_dynamic_offset=*/ false),
        // `mesh_culling_data`
        storage_buffer_read_only::<MeshCullingData>(/*has_dynamic_offset=*/ false),
        // `view`
        uniform_buffer::<ViewUniform>(/*has_dynamic_offset=*/ true),
    ))
}

fn preprocess_gpu_occlusion_culling_bind_group_layout_entries() -> DynamicBindGroupLayoutEntries {
    preprocess_gpu_frustum_culling_bind_group_layout_entries().extend_sequential((
        // `prepass_output`
        storage_buffer::<MeshUniform>(/*has_dynamic_offset=*/ false),
        // `preprocess_indirect_parameters`
        storage_buffer::<IndirectParameters>(/*has_dynamic_offset=*/ false),
        // `visibility`
        storage_buffer::<u32>(/*has_dynamic_offset=*/ false),
        // `depth_pyramid`
        texture_2d(TextureSampleType::Float { filterable: false }),
    ))
}

/// A system that specializes the `mesh_preprocess.wgsl` pipelines if necessary.
pub fn prepare_preprocess_pipelines(
    pipeline_cache: Res<PipelineCache>,
    mut pipelines: ResMut<SpecializedComputePipelines<PreprocessPipeline>>,
    mut preprocess_pipelines: ResMut<PreprocessPipelines>,
) {
    preprocess_pipelines.direct.prepare(
        &pipeline_cache,
        &mut pipelines,
        PreprocessPipelineKey::empty(),
    );
    preprocess_pipelines.gpu_frustum_culling.prepare(
        &pipeline_cache,
        &mut pipelines,
        PreprocessPipelineKey::GPU_FRUSTUM_CULLING,
    );
    preprocess_pipelines.gpu_occlusion_culling.prepare(
        &pipeline_cache,
        &mut pipelines,
        PreprocessPipelineKey::GPU_FRUSTUM_CULLING | PreprocessPipelineKey::GPU_OCCLUSION_CULLING,
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

/// A system that attaches the mesh uniform buffers to the bind groups for the
/// variants of the mesh preprocessing compute shader.
#[allow(clippy::too_many_arguments)]
pub fn prepare_preprocess_bind_groups(
    mut commands: Commands,
    render_device: Res<RenderDevice>,
    batched_instance_buffers: Res<BatchedInstanceBuffers<MeshUniform, MeshInputUniform>>,
    indirect_parameters_buffers: Res<IndirectParametersBuffers>,
    mesh_culling_data_buffer: Res<MeshCullingDataBuffer>,
    view_uniforms: Res<ViewUniforms>,
    render_hi_z_view_resources: Res<RenderHiZViewResources>,
    pipelines: Res<PreprocessPipelines>,
) {
    // Grab the `BatchedInstanceBuffers`.
    let BatchedInstanceBuffers {
        early_prepass_output_data_buffer: ref early_prepass_output_data_buffer_vec,
        late_prepass_output_data_buffer: ref late_prepass_output_data_buffer_vec,
        main_pass_output_data_buffer: ref main_pass_output_data_buffer_vec,
        work_item_buffers: ref index_buffers,
        current_input_buffer: ref current_input_buffer_vec,
        previous_input_buffer: ref previous_input_buffer_vec,
    } = batched_instance_buffers.into_inner();

    let (
        Some(current_input_buffer),
        Some(previous_input_buffer),
        Some(main_pass_output_data_buffer),
    ) = (
        current_input_buffer_vec.buffer(),
        previous_input_buffer_vec.buffer(),
        main_pass_output_data_buffer_vec.buffer(),
    )
    else {
        return;
    };

    for (view, index_buffer_vec) in index_buffers {
        let Some(index_buffer) = index_buffer_vec.work_item_buffer.buffer() else {
            continue;
        };

        // Don't use `as_entire_binding()` here; the shader reads the array
        // length and the underlying buffer may be longer than the actual size
        // of the vector.
        let index_buffer_size = NonZeroU64::try_from(
            index_buffer_vec.work_item_buffer.len() as u64
                * u64::from(PreprocessWorkItem::min_size()),
        )
        .ok();

        let bind_group = if matches!(
            index_buffer_vec.culling_buffers,
            CullingBuffers::GpuFrustum | CullingBuffers::GpuOcclusion { .. }
        ) {
            let (
                Some(main_pass_indirect_parameters_buffer),
                Some(mesh_culling_data_buffer),
                Some(view_uniforms_binding),
            ) = (
                indirect_parameters_buffers.main_pass.buffer(),
                mesh_culling_data_buffer.buffer(),
                view_uniforms.uniforms.binding(),
            )
            else {
                continue;
            };

            let index_buffer_binding = BindingResource::Buffer(BufferBinding {
                buffer: index_buffer,
                offset: 0,
                size: index_buffer_size,
            });

            match index_buffer_vec.culling_buffers {
                CullingBuffers::GpuOcclusion {
                    visibility_buffer: ref visibility_buffer_vec,
                } => {
                    let (
                        Some(early_prepass_indirect_parameters_buffer),
                        Some(late_prepass_indirect_parameters_buffer),
                        Some(early_prepass_output_data_buffer),
                        Some(late_prepass_output_data_buffer),
                        Some(visibility_buffer),
                        Some(hi_z_view_resources),
                    ) = (
                        indirect_parameters_buffers.early_prepass.buffer(),
                        indirect_parameters_buffers.late_prepass.buffer(),
                        early_prepass_output_data_buffer_vec.buffer(),
                        late_prepass_output_data_buffer_vec.buffer(),
                        visibility_buffer_vec.buffer(),
                        render_hi_z_view_resources.get(view),
                    )
                    else {
                        continue;
                    };

                    PreprocessBindGroups {
                        early: render_device.create_bind_group(
                            "early preprocessing with occlusion culling bind group",
                            &pipelines.gpu_occlusion_culling.bind_group_layout,
                            &BindGroupEntries::sequential((
                                // `current_input`
                                current_input_buffer.as_entire_binding(),
                                // `previous_input`
                                previous_input_buffer.as_entire_binding(),
                                // `work_items`
                                index_buffer_binding.clone(),
                                // `main_pass_output`
                                main_pass_output_data_buffer.as_entire_binding(),
                                // `main_pass_indirect_parameters`
                                main_pass_indirect_parameters_buffer.as_entire_binding(),
                                // `mesh_culling_data`
                                mesh_culling_data_buffer.as_entire_binding(),
                                // `view`
                                view_uniforms_binding.clone(),
                                // `prepass_output`
                                early_prepass_output_data_buffer.as_entire_binding(),
                                // `prepass_indirect_parameters`
                                early_prepass_indirect_parameters_buffer.as_entire_binding(),
                                // `visibility`
                                visibility_buffer.as_entire_binding(),
                                // `depth_pyramid`
                                hi_z_view_resources.previous_depth_pyramid.into_binding(),
                            )),
                        ),

                        late: Some(render_device.create_bind_group(
                            "late preprocessing with occlusion culling bind group",
                            &pipelines.gpu_occlusion_culling.bind_group_layout,
                            &BindGroupEntries::sequential((
                                // `current_input`
                                current_input_buffer.as_entire_binding(),
                                // `previous_input`
                                previous_input_buffer.as_entire_binding(),
                                // `work_items`
                                index_buffer_binding,
                                // `main_pass_output`
                                main_pass_output_data_buffer.as_entire_binding(),
                                // `main_pass_indirect_parameters`
                                main_pass_indirect_parameters_buffer.as_entire_binding(),
                                // `mesh_culling_data`
                                mesh_culling_data_buffer.as_entire_binding(),
                                // `view`
                                view_uniforms_binding,
                                // `prepass_output`
                                late_prepass_output_data_buffer.as_entire_binding(),
                                // `prepass_indirect_parameters`
                                late_prepass_indirect_parameters_buffer.as_entire_binding(),
                                // `visibility`
                                visibility_buffer.as_entire_binding(),
                                // `depth_pyramid`
                                hi_z_view_resources.previous_depth_pyramid.into_binding(),
                            )),
                        )),
                    }
                }
                CullingBuffers::Cpu | CullingBuffers::GpuFrustum => {
                    PreprocessBindGroups {
                        early: render_device.create_bind_group(
                            "early preprocessing with frustum culling bind group",
                            &pipelines.gpu_frustum_culling.bind_group_layout,
                            &BindGroupEntries::sequential((
                                // `current_input`
                                current_input_buffer.as_entire_binding(),
                                // `previous_input`
                                previous_input_buffer.as_entire_binding(),
                                // `work_items`
                                index_buffer_binding,
                                // `main_pass_output`
                                main_pass_output_data_buffer.as_entire_binding(),
                                // `main_pass_indirect_parameters`
                                main_pass_indirect_parameters_buffer.as_entire_binding(),
                                // `mesh_culling_data`
                                mesh_culling_data_buffer.as_entire_binding(),
                                // `view`
                                view_uniforms_binding,
                            )),
                        ),
                        late: None,
                    }
                }
            }
        } else {
            PreprocessBindGroups {
                early: render_device.create_bind_group(
                    "direct preprocessing bind group",
                    &pipelines.direct.bind_group_layout,
                    &BindGroupEntries::sequential((
                        current_input_buffer.as_entire_binding(),
                        previous_input_buffer.as_entire_binding(),
                        BindingResource::Buffer(BufferBinding {
                            buffer: index_buffer,
                            offset: 0,
                            size: index_buffer_size,
                        }),
                        main_pass_output_data_buffer.as_entire_binding(),
                    )),
                ),
                late: None,
            }
        };

        commands.entity(*view).insert(bind_group);
    }
}

/// Writes the information needed to do GPU mesh culling to the GPU.
pub fn write_mesh_culling_data_buffer(
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
    mut mesh_culling_data_buffer: ResMut<MeshCullingDataBuffer>,
) {
    mesh_culling_data_buffer.write_buffer(&render_device, &render_queue);
    mesh_culling_data_buffer.clear();
}
