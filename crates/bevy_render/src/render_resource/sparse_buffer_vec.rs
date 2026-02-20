//! A buffer that's sparsely updated from the CPU.

use core::{
    mem,
    sync::atomic::{AtomicU64, Ordering},
};

use bevy_app::{App, Plugin};
use bevy_asset::{embedded_asset, load_embedded_asset, Handle};
use bevy_derive::{Deref, DerefMut};
use bevy_ecs::{
    resource::Resource,
    schedule::IntoScheduleConfigs as _,
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
use wgpu::{BufferDescriptor, BufferUsages, ComputePassDescriptor, ShaderStages};

use crate::{
    diagnostic::{DiagnosticsRecorder, RecordDiagnostics as _},
    render_resource::{
        BindGroup, BindGroupEntries, BindingResource, Buffer, PipelineCache, RawBufferVec,
        SpecializedComputePipeline, SpecializedComputePipelines, UniformBuffer,
    },
    renderer::{RenderDevice, RenderGraph, RenderGraphSystems, RenderQueue},
    ExtractSchedule, RenderApp,
};

pub struct SparseBufferPlugin;

impl Plugin for SparseBufferPlugin {
    fn build(&self, app: &mut App) {
        embedded_asset!(app, "sparse_buffer_update.wgsl");
    }

    fn finish(&self, app: &mut App) {
        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };

        render_app
            .init_resource::<SparseBufferUpdateJobs>()
            .init_resource::<SparseBufferUpdatePipelines>()
            .init_resource::<SpecializedComputePipelines<SparseBufferUpdatePipelines>>()
            .init_resource::<SparseBufferUpdateBindGroups>()
            .add_systems(ExtractSchedule, clear_sparse_buffer_jobs)
            .add_systems(
                RenderGraph,
                update_sparse_buffers.in_set(RenderGraphSystems::Begin),
            );
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
    // TODO: make this a weak map
    bind_groups: HashMap<SparseBufferId, SparseBufferUpdateBindGroup>,
    pipeline_id: CachedComputePipelineId,
}

pub struct SparseBufferUpdateBindGroup {
    bind_group: BindGroup,
}

#[derive(Resource, Default, Deref, DerefMut)]
pub struct SparseBufferUpdateJobs(pub Vec<SparseBufferUpdateJob>);

pub struct SparseBufferUpdateJob {
    sparse_buffer_id: SparseBufferId,
    word_len: u32,
}

#[derive(Clone, Copy, Default, ShaderType, Pod, Zeroable)]
#[repr(C)]
struct GpuSparseBufferUpdateMetadata {
    element_size: u32,
    element_stride: u32,
    element_update_word_len: u32,
}

fn update_sparse_buffers(
    sparse_buffer_update_jobs: Res<SparseBufferUpdateJobs>,
    sparse_buffer_update_bind_groups: Res<SparseBufferUpdateBindGroups>,
    pipeline_cache: Res<PipelineCache>,
    mut diagnostics: Option<ResMut<DiagnosticsRecorder>>,
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
) {
    if sparse_buffer_update_jobs.is_empty() {
        return;
    }

    let mut command_encoder =
        render_device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("sparse buffer update"),
        });

    let time_span = diagnostics
        .as_mut()
        .map(|diagnostics| diagnostics.time_span(&mut command_encoder, "sparse buffer update"));

    command_encoder.push_debug_group("sparse buffer update");

    let Some(compute_pipeline) =
        pipeline_cache.get_compute_pipeline(sparse_buffer_update_bind_groups.pipeline_id)
    else {
        println!("failed to get compute pipeline");
        return;
    };

    for sparse_buffer_update_job in sparse_buffer_update_jobs.iter() {
        let Some(sparse_buffer_update_bind_group) = sparse_buffer_update_bind_groups
            .bind_groups
            .get(&sparse_buffer_update_job.sparse_buffer_id)
        else {
            println!("failed to get sparse buffer update bind group");
            continue;
        };

        let mut sparse_buffer_update_pass =
            command_encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("sparse buffer update"),
                timestamp_writes: None,
            });
        sparse_buffer_update_pass.set_pipeline(compute_pipeline);
        sparse_buffer_update_pass.set_bind_group(
            0,
            &sparse_buffer_update_bind_group.bind_group,
            &[],
        );
        sparse_buffer_update_pass.dispatch_workgroups(
            sparse_buffer_update_job
                .word_len
                .div_ceil(SPARSE_BUFFER_UPDATE_WORKGROUP_SIZE),
            1,
            1,
        );
    }

    command_encoder.pop_debug_group();
    if let Some(mut time_span) = time_span {
        time_span.end(&mut command_encoder);
    }

    render_queue.submit([command_encoder.finish()]);
}

fn clear_sparse_buffer_jobs(mut sparse_buffer_update_jobs: ResMut<SparseBufferUpdateJobs>) {
    sparse_buffer_update_jobs.clear();
}

impl FromWorld for SparseBufferUpdatePipelines {
    fn from_world(world: &mut World) -> Self {
        let bind_group_layout = BindGroupLayoutDescriptor::new(
            "sparse buffer update bind group layout",
            &BindGroupLayoutEntries::sequential(
                ShaderStages::COMPUTE,
                (
                    // @group(0) @binding(0) var<storage, read_write> dest_buffer: array<u32>;
                    storage_buffer::<u32>(false),
                    // @group(0) @binding(1) var<storage> src_buffer: array<u32>;
                    storage_buffer_read_only::<u32>(false),
                    // @group(0) @binding(2) var<storage> indices: array<u32>;
                    storage_buffer_read_only::<u32>(false),
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

    fn specialize(&self, _: Self::Key) -> ComputePipelineDescriptor {
        ComputePipelineDescriptor {
            label: Some("sparse buffer vec update pipeline".into()),
            layout: vec![self.bind_group_layout.clone()],
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

#[derive(Clone, Copy, PartialEq, Debug)]
enum SparseBufferVecState {
    Clean,
    DirtySparse,
    DirtyDense,
}

impl<T> SparseBufferVec<T>
where
    T: NoUninit,
{
    pub fn new(buffer_usages: BufferUsages, label: String) -> Self {
        let id = SparseBufferId(NEXT_SPARSE_BUFFER_ID.fetch_add(1, Ordering::Relaxed));
        Self {
            id,
            values: vec![],
            data_buffer: None,
            staging_buffers: SparseBufferStagingBuffers::new(&label),
            metadata_uniform: UniformBuffer::from(GpuSparseBufferUpdateMetadata {
                element_size: (size_of::<T>() / 4) as u32,
                element_stride: (size_of::<T>() / 4) as u32,
                element_update_word_len: 0,
            }),
            capacity: 0,
            buffer_usages: buffer_usages | BufferUsages::COPY_DST,
            label,
            state: SparseBufferVecState::Clean,
        }
    }

    pub fn reserve(&mut self, new_capacity: usize, render_device: &RenderDevice) {
        if new_capacity == 0 || new_capacity <= self.capacity {
            return;
        }

        self.capacity = new_capacity;
        self.data_buffer = Some(render_device.create_buffer(&BufferDescriptor {
            label: Some(&*self.label),
            size: size_of::<T>() as u64 * new_capacity as u64,
            usage: self.buffer_usages,
            mapped_at_creation: false,
        }));
        // Since we resized the buffer, we need to reupload it.
        self.state = SparseBufferVecState::DirtyDense;
    }

    pub fn buffer(&self) -> Option<&Buffer> {
        self.data_buffer.as_ref()
    }

    pub fn binding(&self) -> Option<BindingResource<'_>> {
        Some(BindingResource::Buffer(
            self.buffer()?.as_entire_buffer_binding(),
        ))
    }

    pub fn len(&self) -> u32 {
        self.values.len() as u32
    }

    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    pub fn clear(&mut self) {
        self.values.clear();
        self.state = SparseBufferVecState::Clean;
    }

    pub fn push(&mut self, value: T) -> u32 {
        let index = self.values.len() as u32;
        self.values.push(value);
        self.note_changed_index(index);
        index
    }

    pub fn get(&self, index: u32) -> &T {
        &self.values[index as usize]
    }

    pub fn set(&mut self, index: u32, value: T) {
        self.values[index as usize] = value;
        self.note_changed_index(index);
    }

    fn note_changed_index(&mut self, index: u32) {
        // TODO: switch to dense
        self.staging_buffers
            .source_data
            .push(self.values[index as usize]);
        self.staging_buffers.indices.push(index);
        self.state = SparseBufferVecState::DirtySparse;
    }

    pub fn write_buffers(&mut self, render_device: &RenderDevice, render_queue: &RenderQueue) {
        if self.values.is_empty() {
            return;
        }

        // FIXME: Try 1.5 instead of power of two
        self.reserve(self.values.len().next_power_of_two(), render_device);

        match self.state {
            SparseBufferVecState::Clean => {}

            SparseBufferVecState::DirtyDense => {
                let Some(ref mut data_buffer) = self.data_buffer else {
                    error!("Dirty sparse buffer should have created a data buffer by now");
                    return;
                };

                render_queue.write_buffer(data_buffer, 0, bytemuck::cast_slice(&self.values[..]));
            }

            SparseBufferVecState::DirtySparse => {
                self.metadata_uniform.get_mut().element_update_word_len =
                    self.staging_buffers.word_len();
                self.metadata_uniform
                    .write_buffer(render_device, render_queue);

                self.staging_buffers
                    .write_buffers(render_device, render_queue);
            }
        }
    }

    pub fn prepare_to_populate_buffers(
        &mut self,
        render_device: &RenderDevice,
        pipeline_cache: &PipelineCache,
        sparse_buffer_update_jobs: &mut SparseBufferUpdateJobs,
        sparse_buffer_update_bind_groups: &mut SparseBufferUpdateBindGroups,
        sparse_buffer_update_pipelines: &SparseBufferUpdatePipelines,
    ) {
        match mem::replace(&mut self.state, SparseBufferVecState::Clean) {
            SparseBufferVecState::Clean | SparseBufferVecState::DirtyDense => {}

            SparseBufferVecState::DirtySparse => {
                let (
                    Some(data_buffer),
                    Some(source_data_staging_buffer),
                    Some(indices_staging_buffer),
                    Some(metadata_buffer),
                ) = (
                    &self.data_buffer,
                    self.staging_buffers.source_data.buffer(),
                    self.staging_buffers.indices.buffer(),
                    self.metadata_uniform.buffer(),
                )
                else {
                    error!("Buffers should have been created by now");
                    return;
                };

                sparse_buffer_update_jobs.push(SparseBufferUpdateJob {
                    sparse_buffer_id: self.id,
                    word_len: self.staging_buffers.word_len(),
                });

                let bind_group = render_device.create_bind_group(
                    Some(&*format!("{} bind group", self.label)),
                    &pipeline_cache
                        .get_bind_group_layout(&sparse_buffer_update_pipelines.bind_group_layout),
                    &BindGroupEntries::sequential((
                        // @group(0) @binding(0) var<storage, read_write> dest_buffer: array<u32>;
                        data_buffer.as_entire_binding(),
                        // @group(0) @binding(1) var<storage> src_buffer: array<u32>;
                        source_data_staging_buffer.as_entire_binding(),
                        // @group(0) @binding(2) var<storage> indices: array<u32>;
                        indices_staging_buffer.as_entire_binding(),
                        // @group(0) @binding(3) var<uniform> metadata:
                        // SparseBufferUpdateMetadata;
                        metadata_buffer.as_entire_binding(),
                    )),
                );

                sparse_buffer_update_bind_groups
                    .bind_groups
                    .insert(self.id, SparseBufferUpdateBindGroup { bind_group });

                self.staging_buffers.clear();
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

    fn word_len(&self) -> u32 {
        (self.source_data.len() * size_of::<T>() / 4) as u32
    }

    #[allow(unused)]
    fn is_empty(&self) -> bool {
        self.source_data.is_empty()
    }

    fn write_buffers(&mut self, render_device: &RenderDevice, render_queue: &RenderQueue) {
        self.source_data.write_buffer(render_device, render_queue);
        self.indices.write_buffer(render_device, render_queue);
    }
}

impl FromWorld for SparseBufferUpdateBindGroups {
    fn from_world(world: &mut World) -> Self {
        world.resource_scope::<SpecializedComputePipelines<SparseBufferUpdatePipelines>, _>(
            |world, mut specialized_sparse_buffer_update_pipelines| {
                let pipeline_cache = world.resource::<PipelineCache>();
                let sparse_buffer_update_pipelines =
                    world.resource::<SparseBufferUpdatePipelines>();
                let pipeline_id = specialized_sparse_buffer_update_pipelines.specialize(
                    pipeline_cache,
                    sparse_buffer_update_pipelines,
                    (),
                );

                SparseBufferUpdateBindGroups {
                    bind_groups: HashMap::default(),
                    pipeline_id,
                }
            },
        )
    }
}
