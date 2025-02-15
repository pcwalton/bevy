//! Bindless resources.

use std::{
    borrow::Cow,
    num::{NonZeroU32, NonZeroU64},
};

use wgpu::{
    BindGroupLayoutEntry, SamplerBindingType, ShaderStages, TextureSampleType, TextureViewDimension,
};

use crate::render_resource::binding_types::storage_buffer_read_only_sized;

use super::binding_types::{
    sampler, texture_1d, texture_2d, texture_2d_array, texture_3d, texture_cube, texture_cube_array,
};

#[cfg(any(target_os = "macos", target_os = "ios"))]
pub const AUTO_BINDLESS_SLOT_COUNT: u32 = 16;
#[cfg(not(any(target_os = "macos", target_os = "ios")))]
pub const AUTO_BINDLESS_SLOT_COUNT: u32 = 256;

pub static BINDING_NUMBERS: [(BindlessResourceType, u32); 9] = [
    (BindlessResourceType::SamplerFiltering, 1),
    (BindlessResourceType::SamplerNonFiltering, 2),
    (BindlessResourceType::SamplerComparison, 3),
    (BindlessResourceType::Texture1d, 4),
    (BindlessResourceType::Texture2d, 5),
    (BindlessResourceType::Texture2dArray, 6),
    (BindlessResourceType::Texture3d, 7),
    (BindlessResourceType::TextureCube, 8),
    (BindlessResourceType::TextureCubeArray, 9),
];

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

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub enum BindlessResourceType {
    None,
    Buffer,
    SamplerFiltering,
    SamplerNonFiltering,
    SamplerComparison,
    Texture1d,
    Texture2d,
    Texture2dArray,
    Texture3d,
    TextureCube,
    TextureCubeArray,
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

impl BindlessResourceType {
    pub fn binding_number(&self) -> Option<&'static u32> {
        match BINDING_NUMBERS.binary_search_by_key(self, |(key, _)| *key) {
            Ok(position) => Some(&BINDING_NUMBERS[position].1),
            Err(_) => None,
        }
    }
}

impl From<TextureViewDimension> for BindlessResourceType {
    fn from(texture_view_dimension: TextureViewDimension) -> Self {
        match texture_view_dimension {
            TextureViewDimension::D1 => BindlessResourceType::Texture1d,
            TextureViewDimension::D2 => BindlessResourceType::Texture2d,
            TextureViewDimension::D2Array => BindlessResourceType::Texture2dArray,
            TextureViewDimension::Cube => BindlessResourceType::TextureCube,
            TextureViewDimension::CubeArray => BindlessResourceType::TextureCubeArray,
            TextureViewDimension::D3 => BindlessResourceType::Texture3d,
        }
    }
}

impl From<SamplerBindingType> for BindlessResourceType {
    fn from(sampler_binding_type: SamplerBindingType) -> Self {
        match sampler_binding_type {
            SamplerBindingType::Filtering => BindlessResourceType::SamplerFiltering,
            SamplerBindingType::NonFiltering => BindlessResourceType::SamplerNonFiltering,
            SamplerBindingType::Comparison => BindlessResourceType::SamplerComparison,
        }
    }
}
