//! Bindless resources.

use std::{
    borrow::Cow,
    num::{NonZeroU32, NonZeroU64},
};

use wgpu::{BindGroupLayoutEntry, SamplerBindingType, ShaderStages, TextureSampleType};

use crate::render_resource::binding_types::storage_buffer_read_only_sized;

use super::binding_types::{
    sampler, texture_1d, texture_2d, texture_2d_array, texture_3d, texture_cube, texture_cube_array,
};

#[cfg(any(target_os = "macos", target_os = "ios"))]
pub const AUTO_BINDLESS_SLOT_COUNT: u32 = 16;
#[cfg(not(any(target_os = "macos", target_os = "ios")))]
pub const AUTO_BINDLESS_SLOT_COUNT: u32 = 256;

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum BindlessSlotCount {
    Auto,
    Custom(u32),
}

pub struct BindlessDescriptor {
    pub resources: Cow<'static, [BindlessResourceType]>,
    // TODO: Require that this be sorted so we can binary search it?
    pub buffers: Cow<'static, [BindlessBufferDescriptor]>,
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
#[repr(u8)]
pub enum BindlessResourceType {
    None = 0,
    Buffer = 1,
    SamplerFiltering = 2,
    SamplerNonFiltering = 3,
    SamplerComparison = 4,
    Texture1d = 5,
    Texture2d = 6,
    Texture2dArray = 7,
    Texture3d = 8,
    TextureCube = 9,
    TextureCubeArray = 10,
}

#[derive(Clone, Copy)]
pub struct BindlessBufferDescriptor {
    pub index: u32,
    pub element_size: usize,
}

pub fn create_bindless_bind_group_layout_entries(
    bindless_resource_count: u32,
    bindless_slot_count: u32,
) -> Vec<BindGroupLayoutEntry> {
    let bindless_slot_count =
        NonZeroU32::new(bindless_slot_count).expect("Bindless slot count must be nonzero");

    vec![
        storage_buffer_read_only_sized(
            false,
            NonZeroU64::new(bindless_resource_count as u64 * size_of::<u32>() as u64),
        )
        .build(0, ShaderStages::FRAGMENT),
        sampler(SamplerBindingType::Filtering)
            .count(bindless_slot_count)
            .build(1, ShaderStages::FRAGMENT),
        sampler(SamplerBindingType::NonFiltering)
            .count(bindless_slot_count)
            .build(2, ShaderStages::FRAGMENT),
        sampler(SamplerBindingType::Comparison)
            .count(bindless_slot_count)
            .build(3, ShaderStages::FRAGMENT),
        texture_1d(TextureSampleType::Float { filterable: true })
            .count(bindless_slot_count)
            .build(4, ShaderStages::FRAGMENT),
        texture_2d(TextureSampleType::Float { filterable: true })
            .count(bindless_slot_count)
            .build(5, ShaderStages::FRAGMENT),
        texture_2d_array(TextureSampleType::Float { filterable: true })
            .count(bindless_slot_count)
            .build(6, ShaderStages::FRAGMENT),
        texture_3d(TextureSampleType::Float { filterable: true })
            .count(bindless_slot_count)
            .build(7, ShaderStages::FRAGMENT),
        texture_cube(TextureSampleType::Float { filterable: true })
            .count(bindless_slot_count)
            .build(8, ShaderStages::FRAGMENT),
        texture_cube_array(TextureSampleType::Float { filterable: true })
            .count(bindless_slot_count)
            .build(9, ShaderStages::FRAGMENT),
    ]
}

impl BindlessSlotCount {
    pub fn resolve(&self) -> u32 {
        match *self {
            BindlessSlotCount::Auto => AUTO_BINDLESS_SLOT_COUNT,
            BindlessSlotCount::Custom(limit) => limit,
        }
    }
}
