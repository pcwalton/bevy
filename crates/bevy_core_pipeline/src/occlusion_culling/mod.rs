//! GPU occlusion culling.

use bevy_app::{App, Plugin};
use bevy_asset::{load_internal_asset, Handle};
use bevy_derive::{Deref, DerefMut};
use bevy_ecs::{
    component::Component,
    entity::Entity,
    prelude::ReflectComponent,
    query::{QueryState, With},
    schedule::IntoSystemConfigs as _,
    system::{Commands, Res, Resource},
    world::{FromWorld, World},
};
use bevy_reflect::{prelude::ReflectDefault, Reflect};
use bevy_render::{
    batching::gpu_preprocessing::{write_indirect_parameters_buffer, IndirectParametersBuffer},
    render_graph::{Node, NodeRunError, RenderGraphApp, RenderGraphContext},
    render_resource::{
        binding_types::storage_buffer_sized, BindGroup, BindGroupEntries, BindGroupLayout,
        BindGroupLayoutEntries, CachedComputePipelineId, ComputePassDescriptor,
        ComputePipelineDescriptor, PipelineCache, Shader, ShaderStages,
    },
    renderer::{RenderContext, RenderDevice},
    sync_component::SyncComponentPlugin,
    Render, RenderApp, RenderSet,
};

use crate::core_3d::graph::{Core3d, Node3d};

/// The handle to the `mesh_preprocess_types.wgsl` compute shader.
pub const MESH_PREPROCESS_TYPES_SHADER_HANDLE: Handle<Shader> =
    Handle::weak_from_u128(2720440370122465935);

pub const FINISH_CULLING_PHASE_SHADER_HANDLE: Handle<Shader> =
    Handle::weak_from_u128(1987880610895111036);

const FINISH_CULLING_WORKGROUP_SIZE: u32 = 64;

pub struct OcclusionCullingPlugin;

impl Plugin for OcclusionCullingPlugin {
    fn build(&self, app: &mut App) {
        load_internal_asset!(
            app,
            FINISH_CULLING_PHASE_SHADER_HANDLE,
            "finish_culling_phase.wgsl",
            Shader::from_wgsl
        );
        load_internal_asset!(
            app,
            MESH_PREPROCESS_TYPES_SHADER_HANDLE,
            "mesh_preprocess_types.wgsl",
            Shader::from_wgsl
        );

        app.register_type::<OcclusionCulling>()
            .add_plugins(SyncComponentPlugin::<OcclusionCulling>::default());

        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };

        render_app
            .add_systems(
                Render,
                prepare_finish_culling_phase_bind_group
                    .in_set(RenderSet::PrepareResourcesFlush)
                    .after(write_indirect_parameters_buffer),
            )
            .add_render_graph_node::<FinishEarlyCullingPhaseNode>(
                Core3d,
                Node3d::FinishEarlyCullingPhase,
            )
            .add_render_graph_node::<FinishMainCullingPhaseNode>(
                Core3d,
                Node3d::FinishMainCullingPhase,
            )
            .add_render_graph_edges(
                Core3d,
                (
                    Node3d::EarlyPrepass,
                    Node3d::FinishEarlyCullingPhase,
                    Node3d::DownsampleDepth,
                    // TODO: Second mesh preprocessing phase, this one indirect
                    Node3d::FinishMainCullingPhase,
                    Node3d::DeferredPrepass,
                ),
            );
    }

    fn finish(&self, app: &mut App) {
        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };

        render_app
            .init_resource::<FinishCullingPhaseBindGroupLayout>()
            .init_resource::<FinishCullingPhasePipelines>();
    }
}

#[derive(Component, Default, Reflect)]
#[reflect(Component, Default)]
pub struct OcclusionCulling;

#[derive(Resource, Deref, DerefMut)]
pub struct FinishCullingPhaseBindGroupLayout(BindGroupLayout);

impl FromWorld for FinishCullingPhaseBindGroupLayout {
    fn from_world(world: &mut World) -> Self {
        let render_device = world.resource::<RenderDevice>();

        Self(render_device.create_bind_group_layout(
            "finish culling phase bind group layout",
            &BindGroupLayoutEntries::single(
                ShaderStages::COMPUTE,
                storage_buffer_sized(false, None),
            ),
        ))
    }
}

#[derive(Resource, Deref, DerefMut)]
pub struct FinishCullingPhaseBindGroup(BindGroup);

pub fn prepare_finish_culling_phase_bind_group(
    mut commands: Commands,
    render_device: Res<RenderDevice>,
    indirect_parameters_buffer: Res<IndirectParametersBuffer>,
    finish_culling_phase_bind_group_layout: Res<FinishCullingPhaseBindGroupLayout>,
) {
    let Some(indirect_parameters_gpu_buffer) = indirect_parameters_buffer.buffer() else {
        return;
    };

    commands.insert_resource(FinishCullingPhaseBindGroup(
        render_device.create_bind_group(
            "finish culling phase bind group",
            &finish_culling_phase_bind_group_layout,
            &BindGroupEntries::single(indirect_parameters_gpu_buffer.as_entire_binding()),
        ),
    ));
}

#[derive(Resource)]
pub struct FinishCullingPhasePipelines {
    early: CachedComputePipelineId,
    main: CachedComputePipelineId,
}

impl FromWorld for FinishCullingPhasePipelines {
    fn from_world(world: &mut World) -> Self {
        let finish_culling_phase_bind_group_layout =
            (*world.resource::<FinishCullingPhaseBindGroupLayout>()).clone();
        let pipeline_cache = world.resource_mut::<PipelineCache>();

        Self {
            early: pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
                label: Some("finish early culling phase pipeline".into()),
                layout: vec![finish_culling_phase_bind_group_layout.clone()],
                push_constant_ranges: vec![],
                shader: FINISH_CULLING_PHASE_SHADER_HANDLE,
                shader_defs: vec![],
                entry_point: "finish_early_culling_phase".into(),
                zero_initialize_workgroup_memory: false,
            }),
            main: pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
                label: Some("finish main culling phase pipeline".into()),
                layout: vec![finish_culling_phase_bind_group_layout.clone()],
                push_constant_ranges: vec![],
                shader: FINISH_CULLING_PHASE_SHADER_HANDLE,
                shader_defs: vec![],
                entry_point: "finish_main_culling_phase".into(),
                zero_initialize_workgroup_memory: false,
            }),
        }
    }
}

fn run_finish_culling_phase(
    render_context: &mut RenderContext,
    world: &World,
    pipeline_id: CachedComputePipelineId,
) -> Result<(), NodeRunError> {
    let finish_culling_phase_bind_group = world.resource::<FinishCullingPhaseBindGroup>();
    let pipeline_cache = world.resource::<PipelineCache>();
    let indirect_parameters_buffer = world.resource::<IndirectParametersBuffer>();

    let mut compute_pass =
        render_context
            .command_encoder()
            .begin_compute_pass(&ComputePassDescriptor {
                label: Some("finish early culling phase"),
                timestamp_writes: None,
            });

    let Some(pipeline) = pipeline_cache.get_compute_pipeline(pipeline_id) else {
        // This will happen while the pipeline is being compiled and is fine.
        return Ok(());
    };

    compute_pass.set_pipeline(pipeline);
    compute_pass.set_bind_group(0, &*finish_culling_phase_bind_group.0, &[]);

    let workgroup_count =
        (indirect_parameters_buffer.len() as u32).div_ceil(FINISH_CULLING_WORKGROUP_SIZE);
    compute_pass.dispatch_workgroups(workgroup_count, 1, 1);

    Ok(())
}

#[derive(Default)]
pub struct FinishEarlyCullingPhaseNode;

impl Node for FinishEarlyCullingPhaseNode {
    fn run<'w>(
        &self,
        _: &mut RenderGraphContext,
        render_context: &mut RenderContext<'w>,
        world: &'w World,
    ) -> Result<(), NodeRunError> {
        let finish_culling_phase_pipelines = world.resource::<FinishCullingPhasePipelines>();
        run_finish_culling_phase(render_context, world, finish_culling_phase_pipelines.early)
    }
}

#[derive(Default)]
pub struct FinishMainCullingPhaseNode;

impl Node for FinishMainCullingPhaseNode {
    fn run<'w>(
        &self,
        _: &mut RenderGraphContext,
        render_context: &mut RenderContext<'w>,
        world: &'w World,
    ) -> Result<(), NodeRunError> {
        let finish_culling_phase_pipelines = world.resource::<FinishCullingPhasePipelines>();
        run_finish_culling_phase(render_context, world, finish_culling_phase_pipelines.main)
    }
}
