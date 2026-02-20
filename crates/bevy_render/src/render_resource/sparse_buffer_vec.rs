//! A buffer that's sparsely updated from the CPU.

use std::sync::atomic::AtomicU64;

use bevy_asset::{load_embedded_asset, Handle};
use bevy_derive::{Deref, DerefMut};
use bevy_ecs::{
    resource::Resource,
    system::{Res, ResMut},
    world::{FromWorld, World},
};
use bevy_material::{
    bind_group_layout_entries::{
        binding_types::{storage_buffer, storage_buffer_read_only, uniform_buffer},
        BindGroupLayoutEntries,
    },
    descriptor::{BindGroupLayoutDescriptor, CachedComputePipelineId, ComputePipelineDescriptor},
};
use bevy_platform::collections::HashMap;
use bevy_shader::Shader;
use bytemuck::{NoUninit, Pod, Zeroable};
use encase::ShaderType;
use wgpu::{BindGroup, ComputePassDescriptor, ShaderStages};

use crate::{
    render_resource::{PipelineCache, RawBufferVec, SpecializedComputePipeline, UniformBuffer},
    renderer::RenderContext,
};

pub struct SparseBufferPlugin;

impl Plugin for SparseBufferPlugin {
    fn build(&self, app: &mut App) {
        embedded_asset!(app, "sparse_buffer_update.wgsl");
    }
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug, Deref, DerefMut)]
pub struct SparseBufferId(pub u64);

static NEXT_SPARSE_BUFFER_ID: AtomicU64 = AtomicU64::new(0);

const SPARSE_BUFFER_UPDATE_WORKGROUP_SIZE: u32 = 256;

#[derive(Resource)]
pub struct SparseBufferUpdatePipelines {
    bind_group_layout: BindGroupLayoutDescriptor,
    shader: Handle<Shader>,
}

#[derive(Resource)]
pub struct SparseBufferUpdateBindGroups {
    bind_groups: HashMap<SparseBufferId, SparseBufferUpdateBindGroup>,
    pipeline_id: CachedComputePipelineId,
}

pub struct SparseBufferUpdateBindGroup {
    bind_group: BindGroup,
}

#[derive(Resource, Deref, DerefMut)]
pub struct SparseBufferUpdateJobs(pub Vec<SparseBufferUpdateJob>);

pub struct SparseBufferUpdateJob {
    sparse_buffer_id: SparseBufferId,
    count: u32,
}

#[derive(Clone, Copy, ShaderType, Pod, Zeroable)]
#[repr(C)]
struct GpuSparseBufferUpdateMetadata {
    element_size: u32,
    element_stride: u32,
    element_update_count: u32,
}

fn update_sparse_buffers(
    sparse_buffer_update_jobs: Res<SparseBufferUpdateJobs>,
    sparse_buffer_update_pipelines: Res<SparseBufferUpdatePipelines>,
    sparse_buffer_update_bind_groups: Res<SparseBufferUpdateBindGroups>,
    pipeline_cache: Res<PipelineCache>,
    mut render_context: RenderContext,
) {
    if sparse_buffer_update_jobs.is_empty() {
        return;
    }

    let diagnostics = render_context.diagnostic_recorder();
    diagnostics = diagnostics.as_deref();
    let time_span = diagnostics.time_span(render_context.command_encoder(), "sparse buffer update");

    let command_encoder = render_context.command_encoder();
    command_encoder.push_debug_group("sparse buffer update");

    let Some(compute_pipeline) =
        pipeline_cache.get_compute_pipeline(sparse_buffer_update_bind_groups.pipeline_id)
    else {
        return;
    };

    for sparse_buffer_update_job in sparse_buffer_update_jobs.iter() {
        let Some(sparse_buffer_update_bind_group) = sparse_buffer_update_bind_groups
            .bind_groups
            .get(&sparse_buffer_update_job.sparse_buffer_id)
        else {
            continue;
        };

        let mut sparse_buffer_update_pass =
            command_encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("sparse buffer update"),
                timestamp_writes: None,
            });
        sparse_buffer_update_pass.set_pipeline(compute_pipeline);
        sparse_buffer_update_pass.set_bind_group(0, sparse_buffer_update_bind_group, &[]);
        sparse_buffer_update_pass.dispatch_workgroups(
            sparse_buffer_update_job
                .count
                .div_ceil(SPARSE_BUFFER_UPDATE_WORKGROUP_SIZE),
        );
    }

    command_encoder.pop_debug_group();
    time_span.end(render_context.command_encoder());
}

fn clear_sparse_buffer_jobs(mut sparse_buffer_vec_update_jobs: ResMut<SparseBufferUpdateJobs>) {
    sparse_buffer_vec_update_jobs.clear();
}

impl FromWorld for SparseBufferUpdatePipelines {
    fn from_world(world: &mut World) -> Self {
        let bind_group_layout = BindGroupLayoutDescriptor::new(
            "sparse buffer update bind group layout",
            &BindGroupLayoutEntries::sequential(
                ShaderStages::COMPUTE,
                (
                    // @group(0) @binding(0) var<storage, read_write> dest_buffer: array<u32>;
                    storage_buffer(false),
                    // @group(0) @binding(1) var<storage> src_buffer: array<u32>;
                    storage_buffer_read_only(false),
                    // @group(0) @binding(2) var<storage> indices: array<u32>;
                    storage_buffer_read_only(false),
                    // @group(0) @binding(3) var<uniform> metadata:
                    // SparseBufferUpdateMetadata;
                    uniform_buffer::<GpuSparseBufferUpdateMetadata>(false),
                ),
            ),
        );

        SparseBufferUpdatePipelines {
            bind_group_layout,
            shader: load_embedded_asset!(world, "sparse_buffer_update.wgsl"),
        }
    }
}

impl SpecializedComputePipeline for SparseBufferUpdatePipelines {
    type Key = ();

    fn specialize(&self, key: Self::Key) -> ComputePipelineDescriptor {
        ComputePipelineDescriptor {
            label: Some("sparse buffer vec update pipeline".into()),
            layout: vec![self.bind_group_layout],
            shader: self.shader.clone(),
            shader_defs: vec![],
            ..ComputePipelineDescriptor::default()
        }
    }
}

pub struct SparseBufferVec<T> where T: NoUninit {
    values: Vec<T>,
    data_buffer: Option<Buffer>,
    staging_buffer: Option<RawBufferVec<u32>>,
    metadata_uniform: UniformBuffer<GpuSparseBufferUpdateMetadata>,
    capacity: usize,
    item_size: usize,
    buffer_usages: BufferUsages,
    label: Option<String>,
    changed: bool,
}
