use alloc::borrow::Cow;
use std::mem;

use crate::{
    render_asset::{AssetExtractionError, PrepareAssetError, RenderAsset, RenderAssetPlugin},
    render_resource::{Buffer, BufferUsages},
    renderer::{RenderDevice, RenderQueue},
    Render, RenderApp, RenderSystems,
};
use bevy_app::{App, Plugin};
use bevy_asset::{Asset, AssetApp, AssetId, RenderAssetUsages};
use bevy_derive::{Deref, DerefMut};
use bevy_ecs::{
    resource::Resource,
    schedule::IntoScheduleConfigs as _,
    system::{
        lifetimeless::{SRes, SResMut},
        ResMut, SystemParamItem,
    },
};
use bevy_platform::collections::HashSet;
use bevy_reflect::{prelude::ReflectDefault, Reflect};
use bevy_utils::default;
use encase::{internal::WriteInto, ShaderType};
use wgpu::util::BufferInitDescriptor;
use wgpu_types::BufferDescriptor;

/// Adds [`ShaderBuffer`] as an asset that is extracted and uploaded to the GPU.
#[derive(Default)]
pub struct StoragePlugin;

impl Plugin for StoragePlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(RenderAssetPlugin::<GpuShaderBuffer>::default())
            .init_asset::<ShaderBuffer>()
            .register_asset_reflect::<ShaderBuffer>();

        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };
        render_app
            .init_resource::<RenderChangedShaderBuffers>()
            .add_systems(
                Render,
                clear_changed_shader_buffers.in_set(RenderSystems::Cleanup),
            );
    }
}

/// A storage buffer that is prepared as a [`RenderAsset`] and uploaded to the GPU.
#[derive(Asset, Reflect, Debug, Clone)]
#[reflect(opaque)]
#[reflect(Default, Debug, Clone)]
pub struct ShaderBuffer {
    /// Optional data used to initialize the buffer.
    pub data: ShaderBufferData,
    pub label: Cow<'static, str>,
    pub buffer_usage: BufferUsages,
    /// The asset usage of the storage buffer.
    pub asset_usage: RenderAssetUsages,
    /// Whether this buffer should be copied on the GPU when resized.
    pub copy_on_resize: bool,
}

#[derive(Reflect, Debug, Clone)]
#[reflect(Default, Debug, Clone)]
pub enum ShaderBufferData {
    Uninitialized(usize),
    Initialized(Vec<u8>),
}

impl Default for ShaderBuffer {
    fn default() -> Self {
        Self {
            data: ShaderBufferData::Uninitialized(0),
            label: Cow::Borrowed("shader buffer"),
            buffer_usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
            asset_usage: RenderAssetUsages::default(),
            copy_on_resize: false,
        }
    }
}

impl Default for ShaderBufferData {
    fn default() -> Self {
        ShaderBufferData::Uninitialized(0)
    }
}

impl ShaderBuffer {
    /// Creates a new storage buffer with the given data and asset usage.
    pub fn new(data: Vec<u8>, asset_usage: RenderAssetUsages) -> Self {
        ShaderBuffer {
            data: ShaderBufferData::Initialized(data.to_vec()),
            asset_usage,
            ..default()
        }
    }

    /// Creates a new storage buffer with the given size and asset usage.
    pub fn with_size(size: usize, asset_usage: RenderAssetUsages) -> Self {
        ShaderBuffer {
            data: ShaderBufferData::Uninitialized(size),
            asset_usage,
            ..default()
        }
    }

    /// Sets the data of the storage buffer to the given [`ShaderType`].
    pub fn set_data<T>(&mut self, value: T)
    where
        T: ShaderType + WriteInto,
    {
        let size = value.size().get() as usize;
        let mut wrapper = encase::StorageBuffer::<Vec<u8>>::new(Vec::with_capacity(size));
        wrapper.write(&value).unwrap();
        self.data = ShaderBufferData::Initialized(wrapper.into_inner());
    }

    /// Resizes the buffer to the new size.
    ///
    /// If CPU data is present, it will be truncated or zero-extended.
    /// Does not preserve GPU data when the descriptor changes.
    pub fn resize(&mut self, new_size: u64) {
        match self.data {
            ShaderBufferData::Initialized(ref mut data) => {
                data.resize(new_size as usize, 0);
            }
            ShaderBufferData::Uninitialized(ref mut size) => {
                *size = new_size as usize;
                self.copy_on_resize = true;
            }
        }
    }

    /// Resizes the buffer to the new size, preserving existing data.
    ///
    /// If CPU data is present, it will be truncated or zero-extended.
    /// If no CPU data is present, sets `copy_on_resize` to preserve GPU data.
    pub fn resize_in_place(&mut self, new_size: u64) {
        match self.data {
            ShaderBufferData::Initialized(ref mut data) => {
                data.resize(new_size as usize, 0);
            }
            ShaderBufferData::Uninitialized(ref mut size) => {
                *size = new_size as usize;
                self.copy_on_resize = true;
            }
        }
    }

    pub fn len(&self) -> usize {
        match self.data {
            ShaderBufferData::Initialized(ref data) => data.len(),
            ShaderBufferData::Uninitialized(len) => len,
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl<T> From<T> for ShaderBuffer
where
    T: ShaderType + WriteInto,
{
    fn from(value: T) -> Self {
        let size = value.size().get() as usize;
        let mut wrapper = encase::StorageBuffer::<Vec<u8>>::new(Vec::with_capacity(size));
        wrapper.write(&value).unwrap();
        Self::new(wrapper.into_inner(), RenderAssetUsages::default())
    }
}

#[derive(Resource, Default, Deref, DerefMut)]
pub struct RenderChangedShaderBuffers(pub HashSet<AssetId<ShaderBuffer>>);

/// A storage buffer that is prepared as a [`RenderAsset`] and uploaded to the GPU.
pub struct GpuShaderBuffer {
    pub buffer: Buffer,
    pub label: Cow<'static, str>,
    pub buffer_usage: BufferUsages,
    pub had_data: bool,
}

impl RenderAsset for GpuShaderBuffer {
    type SourceAsset = ShaderBuffer;
    type Param = (
        SRes<RenderDevice>,
        SRes<RenderQueue>,
        SResMut<RenderChangedShaderBuffers>,
    );

    fn asset_usage(source_asset: &Self::SourceAsset) -> RenderAssetUsages {
        source_asset.asset_usage
    }

    fn take_gpu_data(
        source: &mut Self::SourceAsset,
        previous_gpu_asset: Option<&Self>,
    ) -> Result<Self::SourceAsset, AssetExtractionError> {
        let data = mem::take(&mut source.data);

        let valid_upload = matches!(data, ShaderBufferData::Initialized(_))
            || previous_gpu_asset.is_none_or(|prev| !prev.had_data);

        valid_upload
            .then(|| Self::SourceAsset {
                data,
                ..source.clone()
            })
            .ok_or(AssetExtractionError::AlreadyExtracted)
    }

    fn prepare_asset(
        source_asset: Self::SourceAsset,
        asset_id: AssetId<Self::SourceAsset>,
        &mut (ref render_device, ref render_queue, ref mut changed_shader_buffers): &mut SystemParamItem<
            Self::Param,
        >,
        previous_asset: Option<&Self>,
    ) -> Result<Self, PrepareAssetError<Self::SourceAsset>> {
        let had_data = matches!(source_asset.data, ShaderBufferData::Initialized(_));

        // FIXME: round up to amortize reallocation costs

        let buffer = if let Some(prev) = previous_asset
            && prev.buffer.size() == source_asset.len() as u64
            && prev.buffer.usage() == source_asset.buffer_usage
            && *prev.label == *source_asset.label
            && (!had_data || source_asset.buffer_usage.contains(BufferUsages::COPY_DST))
        {
            if let ShaderBufferData::Initialized(ref data) = source_asset.data {
                render_queue.write_buffer(&prev.buffer, 0, data);
            }
            prev.buffer.clone()
        } else if let ShaderBufferData::Initialized(ref data) = source_asset.data {
            println!("created new storage buffer, size={}", data.len());
            changed_shader_buffers.insert(asset_id);
            render_device.create_buffer_with_data(&BufferInitDescriptor {
                label: Some(&*source_asset.label),
                contents: data,
                usage: source_asset.buffer_usage,
            })
        } else {
            changed_shader_buffers.insert(asset_id);
            let new_buffer = render_device.create_buffer(&BufferDescriptor {
                label: Some(&*source_asset.label),
                size: source_asset.len() as u64,
                usage: source_asset.buffer_usage,
                mapped_at_creation: false,
            });
            if source_asset.copy_on_resize
                && let Some(previous) = previous_asset
                && previous.buffer.usage().contains(BufferUsages::COPY_SRC)
                && source_asset.buffer_usage.contains(BufferUsages::COPY_DST)
            {
                let copy_size = (source_asset.len() as u64).min(previous.buffer.size());
                let mut encoder =
                    render_device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some("copy_buffer_on_resize"),
                    });
                encoder.copy_buffer_to_buffer(&previous.buffer, 0, &new_buffer, 0, copy_size);
                render_queue.submit([encoder.finish()]);
            }
            new_buffer
        };

        println!("RENDER ASSET PREP asset id {:?} -> {}", asset_id, buffer.size());

        Ok(GpuShaderBuffer {
            buffer,
            label: source_asset.label,
            buffer_usage: source_asset.buffer_usage,
            had_data,
        })
    }
}

fn clear_changed_shader_buffers(mut changed_shader_buffers: ResMut<RenderChangedShaderBuffers>) {
    changed_shader_buffers.clear();
}
