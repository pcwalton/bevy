//! GPU occlusion culling.

use bevy_app::{App, Plugin};
use bevy_asset::{load_internal_asset, Handle};
use bevy_derive::{Deref, DerefMut};
use bevy_ecs::{
    component::Component,
    prelude::ReflectComponent,
    schedule::IntoSystemConfigs as _,
    system::{Commands, Res, ResMut, Resource},
    world::{FromWorld, World},
};
use bevy_math::UVec3;
use bevy_reflect::{prelude::ReflectDefault, Reflect};
use wgpu::BufferUsages;

use crate::{
    batching::gpu_preprocessing::{write_indirect_parameters_buffer, IndirectParametersBuffer},
    extract_component::ExtractComponent,
    render_graph::{Node, NodeRunError, RenderGraphContext},
    render_resource::{
        binding_types::storage_buffer_sized, BindGroup, BindGroupEntries, BindGroupLayout,
        BindGroupLayoutEntries, CachedComputePipelineId, ComputePassDescriptor,
        ComputePipelineDescriptor, PipelineCache, RawBufferVec, Shader, ShaderStages, ShaderType,
    },
    renderer::{RenderContext, RenderDevice, RenderQueue},
    Render, RenderApp, RenderSet,
};

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

        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };

        render_app
            .init_resource::<OriginalIndirectParameterFirstInstancesBuffer>()
            .add_systems(
                Render,
                update_and_write_original_indirect_parameter_first_instances_buffer
                    .in_set(RenderSet::PrepareResourcesFlush)
                    // This must be before `write_indirect_parameters_buffer`
                    // because it reads its length and
                    // `write_indirect_parameters_buffer` clears the buffer
                    // after uploading it to the GPU.
                    .before(write_indirect_parameters_buffer),
            )
            .add_systems(
                Render,
                prepare_finish_culling_phase_bind_group
                    .in_set(RenderSet::PrepareResourcesFlush)
                    .after(write_indirect_parameters_buffer),
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

#[derive(Component, ExtractComponent, Clone, Copy, Default, Reflect)]
#[reflect(Component, Default)]
pub struct OcclusionCulling;

#[derive(Resource, Deref, DerefMut)]
pub struct FinishCullingPhaseBindGroupLayout(BindGroupLayout);

impl FromWorld for FinishCullingPhaseBindGroupLayout {
    fn from_world(world: &mut World) -> Self {
        let render_device = world.resource::<RenderDevice>();

        Self(render_device.create_bind_group_layout(
            "finish culling phase bind group layout",
            &BindGroupLayoutEntries::sequential(
                ShaderStages::COMPUTE,
                (
                    storage_buffer_sized(false, None),
                    storage_buffer_sized(false, None),
                ),
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
    original_indirect_parameter_first_instance_buffer: Res<
        OriginalIndirectParameterFirstInstancesBuffer,
    >,
    finish_culling_phase_bind_group_layout: Res<FinishCullingPhaseBindGroupLayout>,
) {
    let (
        Some(indirect_parameters_gpu_buffer),
        Some(original_indirect_parameter_first_instance_buffer),
    ) = (
        indirect_parameters_buffer.buffer(),
        original_indirect_parameter_first_instance_buffer
            .buffer
            .buffer(),
    )
    else {
        return;
    };

    commands.insert_resource(FinishCullingPhaseBindGroup(
        render_device.create_bind_group(
            "finish culling phase bind group",
            &finish_culling_phase_bind_group_layout,
            &BindGroupEntries::sequential((
                indirect_parameters_gpu_buffer.as_entire_binding(),
                original_indirect_parameter_first_instance_buffer.as_entire_binding(),
            )),
        ),
    ));
}

#[derive(Resource)]
pub struct FinishCullingPhasePipelines {
    early: CachedComputePipelineId,
    late: CachedComputePipelineId,
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
            late: pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
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
    label: &'static str,
) -> Result<(), NodeRunError> {
    let finish_culling_phase_bind_group = world.resource::<FinishCullingPhaseBindGroup>();
    let pipeline_cache = world.resource::<PipelineCache>();
    let indirect_parameters_buffer = world.resource::<IndirectParametersBuffer>();

    let mut compute_pass =
        render_context
            .command_encoder()
            .begin_compute_pass(&ComputePassDescriptor {
                label: Some(label),
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
        run_finish_culling_phase(
            render_context,
            world,
            finish_culling_phase_pipelines.early,
            "finish early culling phase",
        )
    }
}

#[derive(Default)]
pub struct FinishLateCullingPhaseNode;

impl Node for FinishLateCullingPhaseNode {
    fn run<'w>(
        &self,
        _: &mut RenderGraphContext,
        render_context: &mut RenderContext<'w>,
        world: &'w World,
    ) -> Result<(), NodeRunError> {
        let finish_culling_phase_pipelines = world.resource::<FinishCullingPhasePipelines>();
        run_finish_culling_phase(
            render_context,
            world,
            finish_culling_phase_pipelines.late,
            "finish late culling phase",
        )
    }
}

#[derive(Clone, Copy, Default, ShaderType)]
pub struct OcclusionCullingIndirectCounts {
    pub workgroup_counts: UVec3,
    pub invocation_count: u32,
}

#[derive(Resource)]
pub struct OriginalIndirectParameterFirstInstancesBuffer {
    buffer: RawBufferVec<u32>,
}

impl Default for OriginalIndirectParameterFirstInstancesBuffer {
    fn default() -> Self {
        OriginalIndirectParameterFirstInstancesBuffer {
            buffer: RawBufferVec::new(BufferUsages::STORAGE),
        }
    }
}

fn update_and_write_original_indirect_parameter_first_instances_buffer(
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
    mut original_indirect_parameter_first_instances_buffer: ResMut<
        OriginalIndirectParameterFirstInstancesBuffer,
    >,
    indirect_parameters_buffer: Res<IndirectParametersBuffer>,
) {
    for _ in 0..indirect_parameters_buffer.len() {
        original_indirect_parameter_first_instances_buffer
            .buffer
            .push(0);
    }
    original_indirect_parameter_first_instances_buffer
        .buffer
        .write_buffer(&render_device, &render_queue);
    original_indirect_parameter_first_instances_buffer
        .buffer
        .clear();
}
