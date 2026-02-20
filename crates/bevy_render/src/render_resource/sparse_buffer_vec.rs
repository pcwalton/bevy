//! A buffer that's sparsely updated from the CPU.

use std::{
    mem,
    sync::atomic::{AtomicU64, Ordering},
};

use bevy_app::{App, Plugin};
use bevy_asset::{embedded_asset, load_embedded_asset, Handle};
use bevy_derive::{Deref, DerefMut};
use bevy_ecs::{
    resource::Resource,
    system::{Res, ResMut},
    world::{FromWorld, World},
};
use bevy_log::error;
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
use wgpu::{BindGroup, BufferDescriptor, BufferUsages, ComputePassDescriptor, ShaderStages};

use crate::{
    render_resource::{
        BindGroupEntries, Buffer, PipelineCache, RawBufferVec, SpecializedComputePipeline,
        UniformBuffer,
    },
    renderer::{RenderContext, RenderDevice, RenderQueue},
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

pub struct SparseBufferVec<T>
where
    T: NoUninit,
{
    id: SparseBufferId,
    values: Vec<T>,
    data_buffer: Option<Buffer>,
    staging_buffers: SparseBufferStagingBuffers<T>,
    metadata_uniform: UniformBuffer<GpuSparseBufferUpdateMetadata>,
    capacity: usize,
    buffer_usages: BufferUsages,
    label: String,
    state: SparseBufferVecState,
}

struct SparseBufferStagingBuffers<T>
where
    T: NoUninit,
{
    source_data: RawBufferVec<T>,
    indices: RawBufferVec<u32>,
}

enum SparseBufferVecState {
    Clean,
    DirtySparse,
    DirtyDense,
}

impl<T> SparseBufferVec<T>
where
    T: NoUninit,
{
    pub const fn new(buffer_usages: BufferUsages, label: String) -> Self {
        let id = SparseBufferId(NEXT_SPARSE_BUFFER_ID.fetch_add(1, Ordering::Relaxed));
        Self {
            id,
            values: vec![],
            data_buffer: None,
            staging_buffers: SparseBufferStagingBuffers::new(&label),
            metadata_uniform: UniformBuffer::from(GpuSparseBufferUpdateMetadata::default()),
            capacity: 0,
            buffer_usages,
            label,
            state: SparseBufferVecState::Clean,
        }
    }

    pub fn reserve(&mut self, new_capacity: usize, render_device: &RenderDevice) {
        if new_capacity == 0 || new_capacity <= self.capacity {
            return;
        }

        self.capacity = new_capacity;
        self.buffer = Some(render_device.create_buffer(&BufferDescriptor {
            label: &self.label,
            size: size_of::<T>() as u64 * new_capacity as u64,
            usage: self.buffer_usage,
            mapped_at_creation: false,
        }));
        // Since we resized the buffer, we need to reupload it.
        self.state = SparseBufferVecState::DirtyDense;
    }

    pub fn write_buffer(
        &mut self,
        render_device: &RenderDevice,
        render_queue: &RenderQueue,
        pipeline_cache: &PipelineCache,
        sparse_buffer_update_jobs: &mut SparseBufferUpdateJobs,
        sparse_buffer_update_bind_groups: &mut SparseBufferUpdateBindGroups,
        sparse_buffer_update_pipelines: &SparseBufferUpdatePipelines,
    ) {
        if self.values.is_empty() {
            return;
        }

        // FIXME: Try 1.5 instead of power of two
        self.reserve(self.values.len().next_power_of_two(), render_device);

        match mem::replace(&mut self.state, SparseBufferVecState::Clean) {
            SparseBufferVecState::Clean => {}

            SparseBufferVecState::DirtyDense => {
                
            }

            SparseBufferVecState::DirtySparse => {
                let Some(ref mut staging_buffers) = self.staging_buffers else {
                    error!("Dirty sparse buffer should have had a staging buffer");
                    return;
                };
                let Some(ref mut data_buffer) = self.data_buffer else {
                    error!("Dirty sparse buffer should have created a data buffer by now");
                    return;
                };

                sparse_buffer_update_jobs.push(SparseBufferUpdateJob {
                    sparse_buffer_id: self.id,
                    count: staging_buffers.len() as u32,
                });

                staging_buffers.write_buffer(render_device, render_queue);
                staging_buffers.clear();

                self.metadata_uniform
                    .write_buffer(render_device, render_queue);
                let Some(ref mut metadata_buffer) = self.metadata_uniform.buffer() else {
                    error!("Dirty metadata buffer should now exist");
                    return;
                };

                let bind_group = render_device.create_bind_group(
                    format!("{} bind group", self.label),
                    &pipeline_cache
                        .get_bind_group_layout(&sparse_buffer_update_pipelines.bind_group_layout),
                    &BindGroupEntries::sequential((
                        // @group(0) @binding(0) var<storage, read_write> dest_buffer: array<u32>;
                        data_buffer,
                        // @group(0) @binding(1) var<storage> src_buffer: array<u32>;
                        staging_buffers.source_data,
                        // @group(0) @binding(2) var<storage> indices: array<u32>;
                        staging_buffers.indices,
                        // @group(0) @binding(3) var<uniform> metadata:
                        // SparseBufferUpdateMetadata;
                        metadata_buffer,
                    )),
                );

                sparse_buffer_update_bind_groups.bind_groups.insert(self.id, SparseBufferUpdateBindGroup {
                    bind_group,
                });
            }
        }
    }
}

impl<T> SparseBufferStagingBuffers<T>
where
    T: NoUninit,
{
    fn new(label: &str) -> SparseBufferStagingBuffers<T> {
        let mut source_data_buffer =
            RawBufferVec::new(BufferUsages::COPY_DST | BufferUsages::STORAGE);
        source_data_buffer.set_label(Some(&*format!("{} staging buffer", label)));
        let mut indices_buffer = RawBufferVec::new(BufferUsages::COPY_DST | BufferUsages::STORAGE);
        indices_buffer.set_label(Some(&*format!("{} index buffer", label)));
        SparseBufferStagingBuffers {
            source_data: source_data_buffer,
            indices: indices_buffer,
        }
    }

    fn clear(&mut self) {
        self.source_data.clear();
        self.indices.clear();
    }

    fn len(&self) -> u32 {
        self.source_data.len() as u32
    }

    fn is_empty(&self) -> bool {
        self.source_data.is_empty()
    }

    fn write_buffers(&mut self, render_device: &RenderDevice, render_queue: &RenderQueue) {
        self.source_data.write_buffer(&render_device, &render_queue);
        self.indices.write_buffer(&render_device, &render_queue);
    }
}
