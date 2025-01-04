//! GPU mesh preprocessing.
//!
//! This is an optional pass that uses a compute shader to reduce the amount of
//! data that has to be transferred from the CPU to the GPU. When enabled,
//! instead of transferring [`MeshUniform`]s to the GPU, we transfer the smaller
//! [`MeshInputUniform`]s instead and use the GPU to calculate the remaining
//! derived fields in [`MeshUniform`].

use core::{mem, num::NonZero};

use bevy_app::{App, Plugin};
use bevy_asset::{load_internal_asset, Handle};
use bevy_core_pipeline::{
    core_3d::graph::{Core3d, Node3d},
    mip_generation::ViewDepthPyramid,
};
use bevy_derive::{Deref, DerefMut};
use bevy_ecs::{
    component::Component,
    entity::{Entity, EntityHashMap},
    query::{Has, QueryState, With, Without},
    schedule::{common_conditions::resource_exists, IntoSystemConfigs as _},
    system::{lifetimeless::Read, Commands, Query, Res, ResMut, Resource},
    world::{FromWorld, World},
};
use bevy_render::{
    batching::gpu_preprocessing::{
        BatchedInstanceBuffers, GpuPreprocessingSupport, IndirectBatchSet,
        IndirectParametersBuffers, IndirectParametersIndexed, IndirectParametersMetadata,
        IndirectParametersNonIndexed, PreprocessWorkItem, PreprocessWorkItemBuffers,
    },
    occlusion_culling::OcclusionCulling,
    render_graph::{Node, NodeRunError, RenderGraphApp, RenderGraphContext},
    render_resource::{
        binding_types::{storage_buffer, storage_buffer_read_only, texture_2d, uniform_buffer},
        BindGroup, BindGroupEntries, BindGroupLayout, BindingResource, Buffer, BufferBinding,
        BufferUsages, CachedComputePipelineId, ComputePassDescriptor, ComputePipelineDescriptor,
        DynamicBindGroupLayoutEntries, PipelineCache, RawBufferVec, Shader, ShaderStages,
        ShaderType, SpecializedComputePipeline, SpecializedComputePipelines, TextureSampleType,
    },
    renderer::{RenderContext, RenderDevice, RenderQueue},
    settings::WgpuFeatures,
    view::{NoIndirectDrawing, ViewUniform, ViewUniformOffset, ViewUniforms},
    Render, RenderApp, RenderSet,
};
use bevy_utils::{tracing::warn, Entry, TypeIdMap};
use bitflags::bitflags;
use bytemuck::{Pod, Zeroable};
use smallvec::{smallvec, SmallVec};

use crate::{
    graph::NodePbr, MeshCullingData, MeshCullingDataBuffer, MeshInputUniform, MeshUniform,
};

/// The handle to the `mesh_preprocess.wgsl` compute shader.
pub const MESH_PREPROCESS_SHADER_HANDLE: Handle<Shader> =
    Handle::weak_from_u128(16991728318640779533);
pub const RESET_INDIRECT_BATCH_SETS_SHADER_HANDLE: Handle<Shader> =
    Handle::weak_from_u128(2602194133710559644);
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
pub struct EarlyGpuPreprocessNode {
    view_query: QueryState<
        (
            Entity,
            Read<PreprocessBindGroups>,
            Read<ViewUniformOffset>,
            Has<NoIndirectDrawing>,
            Has<OcclusionCulling>,
        ),
        Without<SkipGpuPreprocess>,
    >,
}

pub struct LateGpuPreprocessNode {
    view_query: QueryState<
        (Entity, Read<PreprocessBindGroups>, Read<ViewUniformOffset>),
        (
            Without<SkipGpuPreprocess>,
            Without<NoIndirectDrawing>,
            With<OcclusionCulling>,
        ),
    >,
}

pub struct EarlyPrepassBuildIndirectParametersNode {
    view_query: QueryState<
        Read<PreprocessBindGroups>,
        (Without<SkipGpuPreprocess>, Without<NoIndirectDrawing>),
    >,
}

pub struct LatePrepassBuildIndirectParametersNode {
    view_query: QueryState<
        Read<PreprocessBindGroups>,
        (Without<SkipGpuPreprocess>, Without<NoIndirectDrawing>),
    >,
}

pub struct MainBuildIndirectParametersNode {
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
    pub gpu_frustum_culling_preprocess: PreprocessPipeline,
    /// The pipeline used for GPU culling. This pipeline populates indirect
    /// parameters.
    pub early_gpu_occlusion_culling_preprocess: PreprocessPipeline,
    pub late_gpu_occlusion_culling_preprocess: PreprocessPipeline,
    pub gpu_frustum_culling_build_indexed_indirect_params: BuildIndirectParametersPipeline,
    pub gpu_frustum_culling_build_non_indexed_indirect_params: BuildIndirectParametersPipeline,
    pub early_phase: PreprocessPhasePipelines,
    pub late_phase: PreprocessPhasePipelines,
    pub main_phase: PreprocessPhasePipelines,
}

#[derive(Clone)]
pub struct PreprocessPhasePipelines {
    pub reset_indirect_batch_sets: ResetIndirectBatchSetsPipeline,
    pub gpu_occlusion_culling_build_indexed_indirect_params: BuildIndirectParametersPipeline,
    pub gpu_occlusion_culling_build_non_indexed_indirect_params: BuildIndirectParametersPipeline,
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

#[derive(Clone)]
pub struct ResetIndirectBatchSetsPipeline {
    pub bind_group_layout: BindGroupLayout,
    pub pipeline_id: Option<CachedComputePipelineId>,
}

#[derive(Clone)]
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
        /// Whether GPU frustum culling is in use.
        ///
        /// This `#define`'s `FRUSTUM_CULLING` in the shader.
        const FRUSTUM_CULLING = 1;
        const OCCLUSION_CULLING = 2;
        const EARLY = 4;
    }

    #[derive(Clone, Copy, PartialEq, Eq, Hash)]
    pub struct BuildIndirectParametersPipelineKey: u8 {
        const INDEXED = 1;
        const MULTI_DRAW_INDIRECT_COUNT_SUPPORTED = 2;
        const OCCLUSION_CULLING = 4;
        const EARLY_PHASE = 8;
        const LATE_PHASE = 16;
        const MAIN_PHASE = 32;
    }
}

/// The compute shader bind group for the mesh uniform building pass.
///
/// This goes on the view.
#[derive(Component, Clone, Deref, DerefMut)]
pub struct PreprocessBindGroups(pub TypeIdMap<PhasePreprocessBindGroups>);

#[derive(Clone)]
pub enum PhasePreprocessBindGroups {
    Direct(BindGroup),
    IndirectFrustumCulling {
        indexed: Option<BindGroup>,
        non_indexed: Option<BindGroup>,
    },
    IndirectOcclusionCulling {
        early_indexed: Option<BindGroup>,
        early_non_indexed: Option<BindGroup>,
        late_indexed: Option<BindGroup>,
        late_non_indexed: Option<BindGroup>,
    },
}

#[derive(Resource)]
pub struct BuildIndirectParametersBindGroups {
    reset_indexed_indirect_batch_sets: Option<BindGroup>,
    reset_non_indexed_indirect_batch_sets: Option<BindGroup>,
    build_indexed_indirect: Option<BindGroup>,
    build_non_indexed_indirect: Option<BindGroup>,
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
            RESET_INDIRECT_BATCH_SETS_SHADER_HANDLE,
            "reset_indirect_batch_sets.wgsl",
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
            .init_resource::<SpecializedComputePipelines<ResetIndirectBatchSetsPipeline>>()
            .init_resource::<SpecializedComputePipelines<BuildIndirectParametersPipeline>>()
            .init_resource::<OcclusionCullingVisibilityBuffers>()
            .add_systems(
                Render,
                (
                    prepare_preprocess_pipelines.in_set(RenderSet::Prepare),
                    (prepare_preprocess_bind_groups, prepare_occlusion_culling_visibility_buffers)
                        .chain()
                        .run_if(
                            resource_exists::<BatchedInstanceBuffers<MeshUniform, MeshInputUniform>>,
                        )
                        .in_set(RenderSet::PrepareBindGroups),
                    write_mesh_culling_data_buffer.in_set(RenderSet::PrepareResourcesFlush),
                )
            )
            .add_render_graph_node::<EarlyGpuPreprocessNode>(Core3d, NodePbr::EarlyGpuPreprocess)
            .add_render_graph_node::<LateGpuPreprocessNode>(Core3d, NodePbr::LateGpuPreprocess)
            .add_render_graph_node::<EarlyPrepassBuildIndirectParametersNode>(Core3d, NodePbr::EarlyPrepassBuildIndirectParameters)
            .add_render_graph_node::<LatePrepassBuildIndirectParametersNode>(Core3d, NodePbr::LatePrepassBuildIndirectParameters)
            .add_render_graph_node::<MainBuildIndirectParametersNode>(Core3d, NodePbr::MainBuildIndirectParameters)
            .add_render_graph_edges(
                Core3d,
                (
                    NodePbr::EarlyGpuPreprocess,
                    NodePbr::EarlyPrepassBuildIndirectParameters,
                    Node3d::EarlyPrepass,
                    Node3d::DownsampleDepth,
                    NodePbr::LateGpuPreprocess,
                    NodePbr::LatePrepassBuildIndirectParameters,
                    // TODO: Fix shadow pass
                    Node3d::LatePrepass,
                    NodePbr::MainBuildIndirectParameters,
                    Node3d::StartMainPass
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
                    label: Some("early mesh preprocessing"),
                    timestamp_writes: None,
                });

        // Run the compute passes.
        for (view, bind_groups, view_uniform_offset, no_indirect_drawing, occlusion_culling) in
            self.view_query.iter_manual(world)
        {
            // Grab the work item buffers for this view.
            let Some(phase_work_item_buffers) = index_buffers.get(&view) else {
                warn!("The preprocessing index buffer wasn't present");
                continue;
            };

            // Select the right pipeline, depending on whether GPU culling is in
            // use.
            let maybe_pipeline_id = if no_indirect_drawing {
                preprocess_pipelines.direct_preprocess.pipeline_id
            } else if occlusion_culling {
                preprocess_pipelines
                    .early_gpu_occlusion_culling_preprocess
                    .pipeline_id
            } else {
                preprocess_pipelines
                    .gpu_frustum_culling_preprocess
                    .pipeline_id
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

            for (phase_type_id, work_item_buffers) in phase_work_item_buffers {
                let Some(phase_bind_groups) = bind_groups.get(phase_type_id) else {
                    continue;
                };

                let mut dynamic_offsets: SmallVec<[u32; 1]> = smallvec![];
                if !no_indirect_drawing {
                    dynamic_offsets.push(view_uniform_offset.offset);
                }

                match *phase_bind_groups {
                    PhasePreprocessBindGroups::Direct(ref bind_group) => {
                        let PreprocessWorkItemBuffers::Direct(work_item_buffer) = work_item_buffers
                        else {
                            continue;
                        };
                        compute_pass.set_bind_group(0, bind_group, &dynamic_offsets);
                        let workgroup_count = work_item_buffer.len().div_ceil(WORKGROUP_SIZE);
                        if workgroup_count > 0 {
                            compute_pass.dispatch_workgroups(workgroup_count as u32, 1, 1);
                        }
                    }

                    PhasePreprocessBindGroups::IndirectFrustumCulling {
                        indexed: ref maybe_indexed_bind_group,
                        non_indexed: ref maybe_non_indexed_bind_group,
                    }
                    | PhasePreprocessBindGroups::IndirectOcclusionCulling {
                        early_indexed: ref maybe_indexed_bind_group,
                        early_non_indexed: ref maybe_non_indexed_bind_group,
                        ..
                    } => {
                        let PreprocessWorkItemBuffers::Indirect {
                            indexed: indexed_buffer,
                            non_indexed: non_indexed_buffer,
                            ..
                        } = work_item_buffers
                        else {
                            continue;
                        };

                        if let Some(indexed_bind_group) = maybe_indexed_bind_group {
                            compute_pass.set_bind_group(0, indexed_bind_group, &dynamic_offsets);
                            let workgroup_count = indexed_buffer.len().div_ceil(WORKGROUP_SIZE);
                            if workgroup_count > 0 {
                                compute_pass.dispatch_workgroups(workgroup_count as u32, 1, 1);
                            }
                        }

                        if let Some(non_indexed_bind_group) = maybe_non_indexed_bind_group {
                            compute_pass.set_bind_group(
                                0,
                                non_indexed_bind_group,
                                &dynamic_offsets,
                            );
                            let workgroup_count = non_indexed_buffer.len().div_ceil(WORKGROUP_SIZE);
                            if workgroup_count > 0 {
                                compute_pass.dispatch_workgroups(workgroup_count as u32, 1, 1);
                            }
                        }
                    }
                }
            }
        }

        Ok(())
    }
}

impl FromWorld for EarlyPrepassBuildIndirectParametersNode {
    fn from_world(world: &mut World) -> Self {
        Self {
            view_query: QueryState::new(world),
        }
    }
}

impl FromWorld for LatePrepassBuildIndirectParametersNode {
    fn from_world(world: &mut World) -> Self {
        Self {
            view_query: QueryState::new(world),
        }
    }
}

impl FromWorld for MainBuildIndirectParametersNode {
    fn from_world(world: &mut World) -> Self {
        Self {
            view_query: QueryState::new(world),
        }
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
        // Grab the [`BatchedInstanceBuffers`].
        let BatchedInstanceBuffers {
            ref work_item_buffers,
            ..
        } = world.resource::<BatchedInstanceBuffers<MeshUniform, MeshInputUniform>>();

        let pipeline_cache = world.resource::<PipelineCache>();
        let preprocess_pipelines = world.resource::<PreprocessPipelines>();

        let mut compute_pass =
            render_context
                .command_encoder()
                .begin_compute_pass(&ComputePassDescriptor {
                    label: Some("late mesh preprocessing"),
                    timestamp_writes: None,
                });

        // Run the compute passes.
        for (view, bind_groups, view_uniform_offset) in self.view_query.iter_manual(world) {
            // Grab the work item buffers for this view.
            let Some(phase_work_item_buffers) = work_item_buffers.get(&view) else {
                warn!("The preprocessing index buffer wasn't present");
                continue;
            };

            let maybe_pipeline_id = preprocess_pipelines
                .late_gpu_occlusion_culling_preprocess
                .pipeline_id;

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

            for (phase_type_id, work_item_buffers) in phase_work_item_buffers {
                let (
                    &PreprocessWorkItemBuffers::Indirect {
                        indexed: ref indexed_work_item_buffer,
                        non_indexed: ref non_indexed_work_item_buffer,
                        ..
                    },
                    Some(&PhasePreprocessBindGroups::IndirectOcclusionCulling {
                        late_indexed: ref late_indexed_bind_group,
                        late_non_indexed: ref late_non_indexed_bind_group,
                        ..
                    }),
                ) = (work_item_buffers, bind_groups.get(phase_type_id))
                else {
                    continue;
                };

                let mut dynamic_offsets: SmallVec<[u32; 1]> = smallvec![];
                dynamic_offsets.push(view_uniform_offset.offset);

                compute_pass.set_bind_group(
                    0,
                    late_indexed_bind_group.as_deref(),
                    &dynamic_offsets,
                );
                let workgroup_count = indexed_work_item_buffer.len().div_ceil(WORKGROUP_SIZE);
                if workgroup_count > 0 {
                    compute_pass.dispatch_workgroups(workgroup_count as u32, 1, 1);
                }

                compute_pass.set_bind_group(
                    0,
                    late_non_indexed_bind_group.as_deref(),
                    &dynamic_offsets,
                );
                let workgroup_count = non_indexed_work_item_buffer.len().div_ceil(WORKGROUP_SIZE);
                if workgroup_count > 0 {
                    compute_pass.dispatch_workgroups(workgroup_count as u32, 1, 1);
                }
            }
        }

        Ok(())
    }
}

impl Node for EarlyPrepassBuildIndirectParametersNode {
    fn update(&mut self, world: &mut World) {
        self.view_query.update_archetypes(world);
    }

    fn run<'w>(
        &self,
        _: &mut RenderGraphContext,
        render_context: &mut RenderContext<'w>,
        world: &'w World,
    ) -> Result<(), NodeRunError> {
        let preprocess_pipelines = world.resource::<PreprocessPipelines>();

        run_build_indirect_parameters_node(
            render_context,
            world,
            &preprocess_pipelines.early_phase,
            "early indirect parameters building",
        )
    }
}

impl Node for LatePrepassBuildIndirectParametersNode {
    fn update(&mut self, world: &mut World) {
        self.view_query.update_archetypes(world);
    }

    fn run<'w>(
        &self,
        _: &mut RenderGraphContext,
        render_context: &mut RenderContext<'w>,
        world: &'w World,
    ) -> Result<(), NodeRunError> {
        let preprocess_pipelines = world.resource::<PreprocessPipelines>();

        run_build_indirect_parameters_node(
            render_context,
            world,
            &preprocess_pipelines.late_phase,
            "late prepass indirect parameters building",
        )
    }
}

impl Node for MainBuildIndirectParametersNode {
    fn update(&mut self, world: &mut World) {
        self.view_query.update_archetypes(world);
    }

    fn run<'w>(
        &self,
        _: &mut RenderGraphContext,
        render_context: &mut RenderContext<'w>,
        world: &'w World,
    ) -> Result<(), NodeRunError> {
        let preprocess_pipelines = world.resource::<PreprocessPipelines>();

        run_build_indirect_parameters_node(
            render_context,
            world,
            &preprocess_pipelines.main_phase,
            "main indirect parameters building",
        )
    }
}

fn run_build_indirect_parameters_node(
    render_context: &mut RenderContext,
    world: &World,
    preprocess_phase_pipelines: &PreprocessPhasePipelines,
    label: &'static str,
) -> Result<(), NodeRunError> {
    let Some(build_indirect_params_bind_groups) =
        world.get_resource::<BuildIndirectParametersBindGroups>()
    else {
        return Ok(());
    };

    let pipeline_cache = world.resource::<PipelineCache>();
    let indirect_parameters_buffers = world.resource::<IndirectParametersBuffers>();

    let mut compute_pass =
        render_context
            .command_encoder()
            .begin_compute_pass(&ComputePassDescriptor {
                label: Some(label),
                timestamp_writes: None,
            });

    // Fetch the pipeline.
    let (
        Some(reset_indirect_batch_sets_pipeline_id),
        Some(build_indexed_indirect_params_pipeline_id),
        Some(build_non_indexed_indirect_params_pipeline_id),
    ) = (
        preprocess_phase_pipelines
            .reset_indirect_batch_sets
            .pipeline_id,
        preprocess_phase_pipelines
            .gpu_occlusion_culling_build_indexed_indirect_params
            .pipeline_id,
        preprocess_phase_pipelines
            .gpu_occlusion_culling_build_non_indexed_indirect_params
            .pipeline_id,
    )
    else {
        warn!("The build indirect parameters pipelines weren't ready");
        return Ok(());
    };

    let (
        Some(reset_indirect_batch_sets_pipeline),
        Some(build_indexed_indirect_params_pipeline),
        Some(build_non_indexed_indirect_params_pipeline),
    ) = (
        pipeline_cache.get_compute_pipeline(reset_indirect_batch_sets_pipeline_id),
        pipeline_cache.get_compute_pipeline(build_indexed_indirect_params_pipeline_id),
        pipeline_cache.get_compute_pipeline(build_non_indexed_indirect_params_pipeline_id),
    )
    else {
        // This will happen while the pipeline is being compiled and is fine.
        return Ok(());
    };

    // Build indexed indirect parameters.
    if let (
        Some(reset_indexed_indirect_batch_sets_bind_group),
        Some(build_indirect_indexed_params_bind_group),
    ) = (
        &build_indirect_params_bind_groups.reset_indexed_indirect_batch_sets,
        &build_indirect_params_bind_groups.build_indexed_indirect,
    ) {
        compute_pass.set_pipeline(reset_indirect_batch_sets_pipeline);
        compute_pass.set_bind_group(0, reset_indexed_indirect_batch_sets_bind_group, &[]);
        let workgroup_count = indirect_parameters_buffers
            .batch_set_count(true)
            .div_ceil(WORKGROUP_SIZE);
        if workgroup_count > 0 {
            compute_pass.dispatch_workgroups(workgroup_count as u32, 1, 1);
        }

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
    if let (
        Some(reset_non_indexed_indirect_batch_sets_bind_group),
        Some(build_indirect_non_indexed_params_bind_group),
    ) = (
        &build_indirect_params_bind_groups.reset_non_indexed_indirect_batch_sets,
        &build_indirect_params_bind_groups.build_non_indexed_indirect,
    ) {
        compute_pass.set_pipeline(reset_indirect_batch_sets_pipeline);
        compute_pass.set_bind_group(0, reset_non_indexed_indirect_batch_sets_bind_group, &[]);
        let workgroup_count = indirect_parameters_buffers
            .batch_set_count(false)
            .div_ceil(WORKGROUP_SIZE);
        if workgroup_count > 0 {
            compute_pass.dispatch_workgroups(workgroup_count as u32, 1, 1);
        }

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

impl PreprocessPipelines {
    pub(crate) fn pipelines_are_loaded(&self, pipeline_cache: &PipelineCache) -> bool {
        self.direct_preprocess.is_loaded(pipeline_cache)
            && self
                .gpu_frustum_culling_preprocess
                .is_loaded(pipeline_cache)
            && self
                .early_gpu_occlusion_culling_preprocess
                .is_loaded(pipeline_cache)
            && self
                .late_gpu_occlusion_culling_preprocess
                .is_loaded(pipeline_cache)
            && self
                .gpu_frustum_culling_build_indexed_indirect_params
                .is_loaded(pipeline_cache)
            && self
                .gpu_frustum_culling_build_non_indexed_indirect_params
                .is_loaded(pipeline_cache)
            && self.early_phase.is_loaded(pipeline_cache)
            && self.late_phase.is_loaded(pipeline_cache)
            && self.main_phase.is_loaded(pipeline_cache)
    }
}

impl PreprocessPhasePipelines {
    fn is_loaded(&self, pipeline_cache: &PipelineCache) -> bool {
        self.reset_indirect_batch_sets.is_loaded(pipeline_cache)
            && self
                .gpu_occlusion_culling_build_indexed_indirect_params
                .is_loaded(pipeline_cache)
            && self
                .gpu_occlusion_culling_build_non_indexed_indirect_params
                .is_loaded(pipeline_cache)
    }
}

impl PreprocessPipeline {
    fn is_loaded(&self, pipeline_cache: &PipelineCache) -> bool {
        self.pipeline_id
            .is_some_and(|pipeline_id| pipeline_cache.get_compute_pipeline(pipeline_id).is_some())
    }
}

impl ResetIndirectBatchSetsPipeline {
    fn is_loaded(&self, pipeline_cache: &PipelineCache) -> bool {
        self.pipeline_id
            .is_some_and(|pipeline_id| pipeline_cache.get_compute_pipeline(pipeline_id).is_some())
    }
}

impl BuildIndirectParametersPipeline {
    fn is_loaded(&self, pipeline_cache: &PipelineCache) -> bool {
        self.pipeline_id
            .is_some_and(|pipeline_id| pipeline_cache.get_compute_pipeline(pipeline_id).is_some())
    }
}

impl SpecializedComputePipeline for PreprocessPipeline {
    type Key = PreprocessPipelineKey;

    fn specialize(&self, key: Self::Key) -> ComputePipelineDescriptor {
        let mut shader_defs = vec![];
        if key.contains(PreprocessPipelineKey::FRUSTUM_CULLING) {
            shader_defs.push("INDIRECT".into());
            shader_defs.push("FRUSTUM_CULLING".into());
        }
        if key.contains(PreprocessPipelineKey::OCCLUSION_CULLING) {
            shader_defs.push("OCCLUSION_CULLING".into());
            if key.contains(PreprocessPipelineKey::EARLY) {
                shader_defs.push("EARLY".into());
            } else {
                shader_defs.push("LATE".into());
            }
        }

        ComputePipelineDescriptor {
            label: Some(
                format!(
                    "mesh preprocessing ({})",
                    if key.contains(
                        PreprocessPipelineKey::OCCLUSION_CULLING | PreprocessPipelineKey::EARLY
                    ) {
                        "early GPU occlusion culling"
                    } else if key.contains(PreprocessPipelineKey::OCCLUSION_CULLING) {
                        "late GPU occlusion culling"
                    } else if key.contains(PreprocessPipelineKey::FRUSTUM_CULLING) {
                        "GPU frustum culling"
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
        let gpu_frustum_culling_bind_group_layout_entries = gpu_culling_bind_group_layout_entries();
        let gpu_early_occlusion_culling_bind_group_layout_entries =
            gpu_occlusion_culling_bind_group_layout_entries().extend_sequential((
                storage_buffer_read_only::<OcclusionCullingMeshVisibility>(false),
            ));
        let gpu_late_occlusion_culling_bind_group_layout_entries =
            gpu_occlusion_culling_bind_group_layout_entries()
                .extend_sequential((texture_2d(TextureSampleType::Float { filterable: true }),));

        let reset_indirect_batch_sets_bind_group_layout_entries =
            DynamicBindGroupLayoutEntries::sequential(
                ShaderStages::COMPUTE,
                (storage_buffer::<IndirectBatchSet>(false),),
            );

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
        let gpu_frustum_culling_bind_group_layout = render_device.create_bind_group_layout(
            "build mesh uniforms GPU frustum culling bind group layout",
            &gpu_frustum_culling_bind_group_layout_entries,
        );
        let gpu_early_occlusion_culling_bind_group_layout = render_device.create_bind_group_layout(
            "build mesh uniforms GPU early occlusion culling bind group layout",
            &gpu_early_occlusion_culling_bind_group_layout_entries,
        );
        let gpu_late_occlusion_culling_bind_group_layout = render_device.create_bind_group_layout(
            "build mesh uniforms GPU late occlusion culling bind group layout",
            &gpu_late_occlusion_culling_bind_group_layout_entries,
        );
        let reset_indirect_batch_sets_bind_group_layout = render_device.create_bind_group_layout(
            "reset indirect batch sets bind group layout",
            &reset_indirect_batch_sets_bind_group_layout_entries,
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

        let preprocess_phase_pipelines = PreprocessPhasePipelines {
            reset_indirect_batch_sets: ResetIndirectBatchSetsPipeline {
                bind_group_layout: reset_indirect_batch_sets_bind_group_layout.clone(),
                pipeline_id: None,
            },
            gpu_occlusion_culling_build_indexed_indirect_params: BuildIndirectParametersPipeline {
                bind_group_layout: build_indexed_indirect_params_bind_group_layout.clone(),
                pipeline_id: None,
            },
            gpu_occlusion_culling_build_non_indexed_indirect_params:
                BuildIndirectParametersPipeline {
                    bind_group_layout: build_non_indexed_indirect_params_bind_group_layout.clone(),
                    pipeline_id: None,
                },
        };

        PreprocessPipelines {
            direct_preprocess: PreprocessPipeline {
                bind_group_layout: direct_bind_group_layout,
                pipeline_id: None,
            },
            gpu_frustum_culling_preprocess: PreprocessPipeline {
                bind_group_layout: gpu_frustum_culling_bind_group_layout,
                pipeline_id: None,
            },
            early_gpu_occlusion_culling_preprocess: PreprocessPipeline {
                bind_group_layout: gpu_early_occlusion_culling_bind_group_layout,
                pipeline_id: None,
            },
            late_gpu_occlusion_culling_preprocess: PreprocessPipeline {
                bind_group_layout: gpu_late_occlusion_culling_bind_group_layout,
                pipeline_id: None,
            },
            gpu_frustum_culling_build_indexed_indirect_params: BuildIndirectParametersPipeline {
                bind_group_layout: build_indexed_indirect_params_bind_group_layout.clone(),
                pipeline_id: None,
            },
            gpu_frustum_culling_build_non_indexed_indirect_params:
                BuildIndirectParametersPipeline {
                    bind_group_layout: build_non_indexed_indirect_params_bind_group_layout.clone(),
                    pipeline_id: None,
                },
            early_phase: preprocess_phase_pipelines.clone(),
            late_phase: preprocess_phase_pipelines.clone(),
            main_phase: preprocess_phase_pipelines.clone(),
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

fn gpu_culling_bind_group_layout_entries() -> DynamicBindGroupLayoutEntries {
    // GPU culling bind group parameters are a superset of those in the CPU
    // culling (direct) shader.
    preprocess_direct_bind_group_layout_entries().extend_sequential((
        // `indirect_parameters`
        storage_buffer::<IndirectParametersMetadata>(/* has_dynamic_offset= */ false),
        // `mesh_culling_data`
        storage_buffer_read_only::<MeshCullingData>(/* has_dynamic_offset= */ false),
        // `view`
        uniform_buffer::<ViewUniform>(/* has_dynamic_offset= */ true),
    ))
}

fn gpu_occlusion_culling_bind_group_layout_entries() -> DynamicBindGroupLayoutEntries {
    gpu_culling_bind_group_layout_entries().extend_sequential((
        // `view_visibility`
        storage_buffer::<OcclusionCullingMeshVisibility>(/* has_dynamic_offset= */ false),
    ))
}

/// A system that specializes the `mesh_preprocess.wgsl` pipelines if necessary.
pub fn prepare_preprocess_pipelines(
    pipeline_cache: Res<PipelineCache>,
    render_device: Res<RenderDevice>,
    mut specialized_preprocess_pipelines: ResMut<SpecializedComputePipelines<PreprocessPipeline>>,
    mut specialized_reset_indirect_batch_sets_pipelines: ResMut<
        SpecializedComputePipelines<ResetIndirectBatchSetsPipeline>,
    >,
    mut specialized_build_indirect_parameters_pipelines: ResMut<
        SpecializedComputePipelines<BuildIndirectParametersPipeline>,
    >,
    preprocess_pipelines: ResMut<PreprocessPipelines>,
) {
    let preprocess_pipelines = preprocess_pipelines.into_inner();

    preprocess_pipelines.direct_preprocess.prepare(
        &pipeline_cache,
        &mut specialized_preprocess_pipelines,
        PreprocessPipelineKey::empty(),
    );
    preprocess_pipelines.gpu_frustum_culling_preprocess.prepare(
        &pipeline_cache,
        &mut specialized_preprocess_pipelines,
        PreprocessPipelineKey::FRUSTUM_CULLING,
    );
    preprocess_pipelines
        .early_gpu_occlusion_culling_preprocess
        .prepare(
            &pipeline_cache,
            &mut specialized_preprocess_pipelines,
            PreprocessPipelineKey::FRUSTUM_CULLING
                | PreprocessPipelineKey::OCCLUSION_CULLING
                | PreprocessPipelineKey::EARLY,
        );
    preprocess_pipelines
        .late_gpu_occlusion_culling_preprocess
        .prepare(
            &pipeline_cache,
            &mut specialized_preprocess_pipelines,
            PreprocessPipelineKey::FRUSTUM_CULLING | PreprocessPipelineKey::OCCLUSION_CULLING,
        );

    let mut build_indirect_parameters_pipeline_key = BuildIndirectParametersPipelineKey::empty();
    /*if render_device
        .wgpu_device()
        .features()
        .contains(WgpuFeatures::MULTI_DRAW_INDIRECT_COUNT)
    {
        build_indirect_parameters_pipeline_key
            .insert(BuildIndirectParametersPipelineKey::MULTI_DRAW_INDIRECT_COUNT_SUPPORTED);
    }*/

    preprocess_pipelines
        .gpu_frustum_culling_build_indexed_indirect_params
        .prepare(
            &pipeline_cache,
            &mut specialized_build_indirect_parameters_pipelines,
            build_indirect_parameters_pipeline_key | BuildIndirectParametersPipelineKey::INDEXED,
        );
    preprocess_pipelines
        .gpu_frustum_culling_build_non_indexed_indirect_params
        .prepare(
            &pipeline_cache,
            &mut specialized_build_indirect_parameters_pipelines,
            build_indirect_parameters_pipeline_key,
        );

    for (preprocess_phase_pipelines, build_indirect_parameters_phase_pipeline_key) in [
        (
            &mut preprocess_pipelines.early_phase,
            BuildIndirectParametersPipelineKey::EARLY_PHASE,
        ),
        (
            &mut preprocess_pipelines.late_phase,
            BuildIndirectParametersPipelineKey::LATE_PHASE,
        ),
        (
            &mut preprocess_pipelines.main_phase,
            BuildIndirectParametersPipelineKey::MAIN_PHASE,
        ),
    ] {
        preprocess_phase_pipelines
            .reset_indirect_batch_sets
            .prepare(
                &pipeline_cache,
                &mut specialized_reset_indirect_batch_sets_pipelines,
            );
        preprocess_phase_pipelines
            .gpu_occlusion_culling_build_indexed_indirect_params
            .prepare(
                &pipeline_cache,
                &mut specialized_build_indirect_parameters_pipelines,
                build_indirect_parameters_pipeline_key
                    | build_indirect_parameters_phase_pipeline_key
                    | BuildIndirectParametersPipelineKey::INDEXED
                    | BuildIndirectParametersPipelineKey::OCCLUSION_CULLING,
            );
        preprocess_phase_pipelines
            .gpu_occlusion_culling_build_non_indexed_indirect_params
            .prepare(
                &pipeline_cache,
                &mut specialized_build_indirect_parameters_pipelines,
                build_indirect_parameters_pipeline_key
                    | build_indirect_parameters_phase_pipeline_key
                    | BuildIndirectParametersPipelineKey::OCCLUSION_CULLING,
            );
    }
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

impl SpecializedComputePipeline for ResetIndirectBatchSetsPipeline {
    type Key = ();

    fn specialize(&self, _: Self::Key) -> ComputePipelineDescriptor {
        ComputePipelineDescriptor {
            label: Some("reset indirect batch sets".into()),
            layout: vec![self.bind_group_layout.clone()],
            push_constant_ranges: vec![],
            shader: RESET_INDIRECT_BATCH_SETS_SHADER_HANDLE,
            shader_defs: vec![],
            entry_point: "main".into(),
            zero_initialize_workgroup_memory: false,
        }
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
        if key.contains(BuildIndirectParametersPipelineKey::OCCLUSION_CULLING) {
            shader_defs.push("OCCLUSION_CULLING".into());
        }
        if key.contains(BuildIndirectParametersPipelineKey::EARLY_PHASE) {
            shader_defs.push("EARLY_PHASE".into());
        }
        if key.contains(BuildIndirectParametersPipelineKey::LATE_PHASE) {
            shader_defs.push("LATE_PHASE".into());
        }
        if key.contains(BuildIndirectParametersPipelineKey::MAIN_PHASE) {
            shader_defs.push("MAIN_PHASE".into());
        }

        let label = format!(
            "{} build {}indexed indirect parameters",
            if !key.contains(BuildIndirectParametersPipelineKey::OCCLUSION_CULLING) {
                "frustum culling"
            } else if key.contains(BuildIndirectParametersPipelineKey::EARLY_PHASE) {
                "early occlusion culling"
            } else if key.contains(BuildIndirectParametersPipelineKey::LATE_PHASE) {
                "late occlusion culling"
            } else {
                "main occlusion culling"
            },
            if key.contains(BuildIndirectParametersPipelineKey::INDEXED) {
                ""
            } else {
                "non-"
            }
        );

        ComputePipelineDescriptor {
            label: Some(label.into()),
            layout: vec![self.bind_group_layout.clone()],
            push_constant_ranges: vec![],
            shader: BUILD_INDIRECT_PARAMS_SHADER_HANDLE,
            shader_defs,
            entry_point: "main".into(),
            zero_initialize_workgroup_memory: false,
        }
    }
}

impl ResetIndirectBatchSetsPipeline {
    fn prepare(
        &mut self,
        pipeline_cache: &PipelineCache,
        pipelines: &mut SpecializedComputePipelines<ResetIndirectBatchSetsPipeline>,
    ) {
        if self.pipeline_id.is_some() {
            return;
        }

        let reset_indirect_batch_sets_pipeline_id = pipelines.specialize(pipeline_cache, self, ());
        self.pipeline_id = Some(reset_indirect_batch_sets_pipeline_id);
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

/// Extra buffers used for occlusion culling.
///
/// This is shared among all views.
#[derive(Resource, Default)]
pub struct OcclusionCullingVisibilityBuffers {
    buffers: EntityHashMap<TypeIdMap<PhaseOcclusionCullingVisibilityBuffers>>,
}

struct PhaseOcclusionCullingVisibilityBuffers {
    current_frame: RawBufferVec<OcclusionCullingMeshVisibility>,
    previous_frame: RawBufferVec<OcclusionCullingMeshVisibility>,
}

#[derive(Clone, Copy, Default, ShaderType, Pod, Zeroable)]
#[repr(C)]
struct OcclusionCullingMeshVisibility {
    visibility: u32,
    pad_a: f32,
    pad_b: f32,
    pad_c: f32,
}

pub fn prepare_occlusion_culling_visibility_buffers(
    mut views: Query<Entity, With<OcclusionCulling>>,
    batched_instance_buffers: Res<BatchedInstanceBuffers<MeshUniform, MeshInputUniform>>,
    mut occlusion_culling_visibility_buffers: ResMut<OcclusionCullingVisibilityBuffers>,
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
) {
    for view_entity in &mut views {
        let Some(work_item_buffers) = batched_instance_buffers.work_item_buffers.get(&view_entity)
        else {
            continue;
        };

        let view_occlusion_culling_visibility_buffers = occlusion_culling_visibility_buffers
            .buffers
            .entry(view_entity)
            .or_insert_with(TypeIdMap::default);

        for &phase_id in work_item_buffers.keys() {
            let mut new_visibility_buffer = RawBufferVec::new(BufferUsages::STORAGE);
            // TODO: Make this more efficient?
            for _ in 0..batched_instance_buffers.current_input_buffer.len() {
                new_visibility_buffer.push(OcclusionCullingMeshVisibility::default());
            }
            new_visibility_buffer.write_buffer(&render_device, &render_queue);

            match view_occlusion_culling_visibility_buffers.entry(phase_id) {
                Entry::Occupied(mut occupied_entry) => {
                    let existing_visibility_buffers = occupied_entry.get_mut();
                    existing_visibility_buffers.previous_frame = mem::replace(
                        &mut existing_visibility_buffers.current_frame,
                        new_visibility_buffer,
                    );
                }

                Entry::Vacant(vacant_entry) => {
                    let mut previous_frame = RawBufferVec::new(BufferUsages::STORAGE);
                    previous_frame.push(OcclusionCullingMeshVisibility::default());

                    vacant_entry.insert(PhaseOcclusionCullingVisibilityBuffers {
                        current_frame: new_visibility_buffer,
                        previous_frame,
                    });
                }
            }
        }
    }

    occlusion_culling_visibility_buffers
        .buffers
        .retain(|&view_entity, _| views.contains(view_entity));
}

/// A system that attaches the mesh uniform buffers to the bind groups for the
/// variants of the mesh preprocessing compute shader.
#[allow(clippy::too_many_arguments)]
pub fn prepare_preprocess_bind_groups(
    mut commands: Commands,
    view_depth_pyramids: Query<&ViewDepthPyramid>,
    render_device: Res<RenderDevice>,
    batched_instance_buffers: Res<BatchedInstanceBuffers<MeshUniform, MeshInputUniform>>,
    indirect_parameters_buffers: Res<IndirectParametersBuffers>,
    mesh_culling_data_buffer: Res<MeshCullingDataBuffer>,
    occlusion_culling_visibility_buffers: Res<OcclusionCullingVisibilityBuffers>,
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
        previous_input_buffer_vec.buffer(),
        data_buffer_vec.buffer(),
    ) else {
        return;
    };

    let mut any_indirect = false;

    for (view, phase_work_item_buffers) in work_item_buffers {
        let mut bind_groups = TypeIdMap::default();

        for (&phase_id, work_item_buffers) in phase_work_item_buffers {
            let bind_group = match *work_item_buffers {
                PreprocessWorkItemBuffers::Direct(ref work_item_buffer) => {
                    // Don't use `as_entire_binding()` here; the shader reads the array
                    // length and the underlying buffer may be longer than the actual size
                    // of the vector.
                    let work_item_buffer_size = NonZero::<u64>::try_from(
                        work_item_buffer.len() as u64 * u64::from(PreprocessWorkItem::min_size()),
                    )
                    .ok();

                    let Some(work_item_buffer) = work_item_buffer.buffer() else {
                        continue;
                    };

                    PhasePreprocessBindGroups::Direct(render_device.create_bind_group(
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

                PreprocessWorkItemBuffers::Indirect {
                    indexed: ref indexed_work_item_buffer,
                    non_indexed: ref non_indexed_work_item_buffer,
                    gpu_occlusion_culling: true,
                } => {
                    let (
                        Some(mesh_culling_data_buffer),
                        Some(view_uniforms_binding),
                        Some(view_occlusion_culling_visibility_buffers),
                        Ok(view_depth_pyramid),
                    ) = (
                        mesh_culling_data_buffer.buffer(),
                        view_uniforms.uniforms.binding(),
                        occlusion_culling_visibility_buffers.buffers.get(view),
                        view_depth_pyramids.get(*view),
                    )
                    else {
                        continue;
                    };

                    let Some(phase_occlusion_culling_visibility_buffers) =
                        view_occlusion_culling_visibility_buffers.get(&phase_id)
                    else {
                        continue;
                    };

                    let (
                        Some(occlusion_culling_visibility_buffer),
                        Some(previous_frame_occlusion_culling_visibility_buffer),
                    ) = (
                        phase_occlusion_culling_visibility_buffers
                            .current_frame
                            .buffer(),
                        phase_occlusion_culling_visibility_buffers
                            .previous_frame
                            .buffer(),
                    )
                    else {
                        continue;
                    };

                    any_indirect = true;

                    PhasePreprocessBindGroups::IndirectOcclusionCulling {
                        early_indexed: match (
                            indirect_parameters_buffers.indexed_metadata_buffer(),
                            indexed_work_item_buffer.buffer(),
                        ) {
                            (Some(indexed_metadata_buffer), Some(indexed_work_item_gpu_buffer)) => {
                                // Don't use `as_entire_binding()` here; the shader reads the array
                                // length and the underlying buffer may be longer than the actual size
                                // of the vector.
                                let indexed_work_item_buffer_size = NonZero::<u64>::try_from(
                                    indexed_work_item_buffer.len() as u64
                                        * u64::from(PreprocessWorkItem::min_size()),
                                )
                                .ok();

                                Some(
                                    render_device.create_bind_group(
                                        "preprocess_early_indexed_gpu_occlusion_culling_bind_group",
                                        &pipelines
                                            .early_gpu_occlusion_culling_preprocess
                                            .bind_group_layout,
                                        &BindGroupEntries::sequential((
                                            current_input_buffer.as_entire_binding(),
                                            previous_input_buffer.as_entire_binding(),
                                            BindingResource::Buffer(BufferBinding {
                                                buffer: indexed_work_item_gpu_buffer,
                                                offset: 0,
                                                size: indexed_work_item_buffer_size,
                                            }),
                                            data_buffer.as_entire_binding(),
                                            indexed_metadata_buffer.as_entire_binding(),
                                            mesh_culling_data_buffer.as_entire_binding(),
                                            view_uniforms_binding.clone(),
                                            occlusion_culling_visibility_buffer.as_entire_binding(),
                                            previous_frame_occlusion_culling_visibility_buffer
                                                .as_entire_binding(),
                                        )),
                                    ),
                                )
                            }
                            _ => None,
                        },

                        early_non_indexed: match (
                            indirect_parameters_buffers.non_indexed_metadata_buffer(),
                            non_indexed_work_item_buffer.buffer(),
                        ) {
                            (
                                Some(non_indexed_metadata_buffer),
                                Some(non_indexed_work_item_gpu_buffer),
                            ) => {
                                // Don't use `as_entire_binding()` here; the shader reads the array
                                // length and the underlying buffer may be longer than the actual size
                                // of the vector.
                                let non_indexed_work_item_buffer_size = NonZero::<u64>::try_from(
                                    non_indexed_work_item_buffer.len() as u64
                                        * u64::from(PreprocessWorkItem::min_size()),
                                )
                                .ok();

                                Some(
                                    render_device.create_bind_group(
                                        "preprocess_early_non_indexed_gpu_occlusion_culling_bind_group",
                                        &pipelines
                                            .early_gpu_occlusion_culling_preprocess
                                            .bind_group_layout,
                                        &BindGroupEntries::sequential((
                                            current_input_buffer.as_entire_binding(),
                                            previous_input_buffer.as_entire_binding(),
                                            BindingResource::Buffer(BufferBinding {
                                                buffer: non_indexed_work_item_gpu_buffer,
                                                offset: 0,
                                                size: non_indexed_work_item_buffer_size,
                                            }),
                                            data_buffer.as_entire_binding(),
                                            non_indexed_metadata_buffer.as_entire_binding(),
                                            mesh_culling_data_buffer.as_entire_binding(),
                                            view_uniforms_binding.clone(),
                                            occlusion_culling_visibility_buffer.as_entire_binding(),
                                            previous_frame_occlusion_culling_visibility_buffer
                                                .as_entire_binding(),
                                        )),
                                    ),
                                )
                            }
                            _ => None,
                        },

                        late_indexed: match (
                            indirect_parameters_buffers.indexed_metadata_buffer(),
                            indexed_work_item_buffer.buffer(),
                        ) {
                            (Some(indexed_metadata_buffer), Some(indexed_work_item_gpu_buffer)) => {
                                // Don't use `as_entire_binding()` here; the shader reads the array
                                // length and the underlying buffer may be longer than the actual size
                                // of the vector.
                                let indexed_work_item_buffer_size = NonZero::<u64>::try_from(
                                    indexed_work_item_buffer.len() as u64
                                        * u64::from(PreprocessWorkItem::min_size()),
                                )
                                .ok();

                                Some(
                                    render_device.create_bind_group(
                                        "preprocess_late_indexed_gpu_occlusion_culling_bind_group",
                                        &pipelines
                                            .late_gpu_occlusion_culling_preprocess
                                            .bind_group_layout,
                                        &BindGroupEntries::sequential((
                                            current_input_buffer.as_entire_binding(),
                                            previous_input_buffer.as_entire_binding(),
                                            BindingResource::Buffer(BufferBinding {
                                                buffer: indexed_work_item_gpu_buffer,
                                                offset: 0,
                                                size: indexed_work_item_buffer_size,
                                            }),
                                            data_buffer.as_entire_binding(),
                                            indexed_metadata_buffer.as_entire_binding(),
                                            mesh_culling_data_buffer.as_entire_binding(),
                                            view_uniforms_binding.clone(),
                                            occlusion_culling_visibility_buffer.as_entire_binding(),
                                            &view_depth_pyramid.all_mips,
                                        )),
                                    ),
                                )
                            }
                            _ => None,
                        },

                        late_non_indexed: match (
                            indirect_parameters_buffers.non_indexed_metadata_buffer(),
                            non_indexed_work_item_buffer.buffer(),
                        ) {
                            (
                                Some(non_indexed_metadata_buffer),
                                Some(non_indexed_work_item_gpu_buffer),
                            ) => {
                                // Don't use `as_entire_binding()` here; the shader reads the array
                                // length and the underlying buffer may be longer than the actual size
                                // of the vector.
                                let non_indexed_work_item_buffer_size = NonZero::<u64>::try_from(
                                    non_indexed_work_item_buffer.len() as u64
                                        * u64::from(PreprocessWorkItem::min_size()),
                                )
                                .ok();

                                Some(
                                    render_device.create_bind_group(
                                        "preprocess_late_non_indexed_gpu_occlusion_culling_bind_group",
                                        &pipelines
                                            .late_gpu_occlusion_culling_preprocess
                                            .bind_group_layout,
                                        &BindGroupEntries::sequential((
                                            current_input_buffer.as_entire_binding(),
                                            previous_input_buffer.as_entire_binding(),
                                            BindingResource::Buffer(BufferBinding {
                                                buffer: non_indexed_work_item_gpu_buffer,
                                                offset: 0,
                                                size: non_indexed_work_item_buffer_size,
                                            }),
                                            data_buffer.as_entire_binding(),
                                            non_indexed_metadata_buffer.as_entire_binding(),
                                            mesh_culling_data_buffer.as_entire_binding(),
                                            view_uniforms_binding.clone(),
                                            occlusion_culling_visibility_buffer.as_entire_binding(),
                                            &view_depth_pyramid.all_mips,
                                        )),
                                    ),
                                )
                            }
                            _ => None,
                        },
                    }
                }

                PreprocessWorkItemBuffers::Indirect {
                    indexed: ref indexed_work_item_buffer,
                    non_indexed: ref non_indexed_work_item_buffer,
                    gpu_occlusion_culling: false,
                } => {
                    let (Some(mesh_culling_data_buffer), Some(view_uniforms_binding)) = (
                        mesh_culling_data_buffer.buffer(),
                        view_uniforms.uniforms.binding(),
                    ) else {
                        continue;
                    };

                    PhasePreprocessBindGroups::IndirectFrustumCulling {
                        indexed: match (
                            indirect_parameters_buffers.indexed_metadata_buffer(),
                            indexed_work_item_buffer.buffer(),
                        ) {
                            (Some(indexed_metadata_buffer), Some(indexed_work_item_gpu_buffer)) => {
                                // Don't use `as_entire_binding()` here; the shader reads the array
                                // length and the underlying buffer may be longer than the actual size
                                // of the vector.
                                let indexed_work_item_buffer_size = NonZero::<u64>::try_from(
                                    indexed_work_item_buffer.len() as u64
                                        * u64::from(PreprocessWorkItem::min_size()),
                                )
                                .ok();

                                Some(render_device.create_bind_group(
                                    "preprocess_gpu_indexed_frustum_culling_bind_group",
                                    &pipelines.gpu_frustum_culling_preprocess.bind_group_layout,
                                    &BindGroupEntries::sequential((
                                        current_input_buffer.as_entire_binding(),
                                        previous_input_buffer.as_entire_binding(),
                                        BindingResource::Buffer(BufferBinding {
                                            buffer: indexed_work_item_gpu_buffer,
                                            offset: 0,
                                            size: indexed_work_item_buffer_size,
                                        }),
                                        data_buffer.as_entire_binding(),
                                        indexed_metadata_buffer.as_entire_binding(),
                                        mesh_culling_data_buffer.as_entire_binding(),
                                        view_uniforms_binding.clone(),
                                    )),
                                ))
                            }
                            _ => None,
                        },

                        non_indexed: match (
                            indirect_parameters_buffers.non_indexed_metadata_buffer(),
                            non_indexed_work_item_buffer.buffer(),
                        ) {
                            (
                                Some(non_indexed_metadata_buffer),
                                Some(non_indexed_work_item_gpu_buffer),
                            ) => {
                                // Don't use `as_entire_binding()` here; the shader reads the array
                                // length and the underlying buffer may be longer than the actual size
                                // of the vector.
                                let non_indexed_work_item_buffer_size = NonZero::<u64>::try_from(
                                    non_indexed_work_item_buffer.len() as u64
                                        * u64::from(PreprocessWorkItem::min_size()),
                                )
                                .ok();

                                Some(render_device.create_bind_group(
                                    "preprocess_gpu_non_indexed_frustum_culling_bind_group",
                                    &pipelines.gpu_frustum_culling_preprocess.bind_group_layout,
                                    &BindGroupEntries::sequential((
                                        current_input_buffer.as_entire_binding(),
                                        previous_input_buffer.as_entire_binding(),
                                        BindingResource::Buffer(BufferBinding {
                                            buffer: non_indexed_work_item_gpu_buffer,
                                            offset: 0,
                                            size: non_indexed_work_item_buffer_size,
                                        }),
                                        data_buffer.as_entire_binding(),
                                        non_indexed_metadata_buffer.as_entire_binding(),
                                        mesh_culling_data_buffer.as_entire_binding(),
                                        view_uniforms_binding.clone(),
                                    )),
                                ))
                            }
                            _ => None,
                        },
                    }
                }
            };

            bind_groups.insert(phase_id, bind_group);
        }

        commands
            .entity(*view)
            .insert(PreprocessBindGroups(bind_groups));
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
        reset_indexed_indirect_batch_sets: match (
            indirect_parameters_buffer.indexed_batch_sets_buffer(),
        ) {
            (Some(indexed_batch_sets_buffer),) => Some(
                render_device.create_bind_group(
                    "reset_indexed_indirect_batch_sets_bind_group",
                    // The early bind group is good for the main phase and late
                    // phase too. They bind the same buffers.
                    &pipelines
                        .early_phase
                        .reset_indirect_batch_sets
                        .bind_group_layout,
                    &BindGroupEntries::sequential((indexed_batch_sets_buffer.as_entire_binding(),)),
                ),
            ),
            _ => None,
        },

        reset_non_indexed_indirect_batch_sets: match (
            indirect_parameters_buffer.non_indexed_batch_sets_buffer(),
        ) {
            (Some(non_indexed_batch_sets_buffer),) => Some(
                render_device.create_bind_group(
                    "reset_non_indexed_indirect_batch_sets_bind_group",
                    // The early bind group is good for the main phase and late
                    // phase too. They bind the same buffers.
                    &pipelines
                        .early_phase
                        .reset_indirect_batch_sets
                        .bind_group_layout,
                    &BindGroupEntries::sequential((
                        non_indexed_batch_sets_buffer.as_entire_binding(),
                    )),
                ),
            ),
            _ => None,
        },

        build_indexed_indirect: match (
            indirect_parameters_buffer.indexed_metadata_buffer(),
            indirect_parameters_buffer.indexed_data_buffer(),
            indirect_parameters_buffer.indexed_batch_sets_buffer(),
        ) {
            (
                Some(indexed_indirect_parameters_metadata_buffer),
                Some(indexed_indirect_parameters_data_buffer),
                Some(indexed_batch_sets_buffer),
            ) => Some(
                render_device.create_bind_group(
                    "build_indexed_indirect_parameters_bind_group",
                    // The frustum culling bind group is good for occlusion culling
                    // too. They bind the same buffers.
                    &pipelines
                        .gpu_frustum_culling_build_indexed_indirect_params
                        .bind_group_layout,
                    &BindGroupEntries::sequential((
                        current_input_buffer.as_entire_binding(),
                        indexed_indirect_parameters_metadata_buffer.as_entire_binding(),
                        indexed_batch_sets_buffer.as_entire_binding(),
                        indexed_indirect_parameters_data_buffer.as_entire_binding(),
                    )),
                ),
            ),
            _ => None,
        },

        build_non_indexed_indirect: match (
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
                    // The frustum culling bind group is good for occlusion culling
                    // too. They bind the same buffers.
                    &pipelines
                        .gpu_frustum_culling_build_non_indexed_indirect_params
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
