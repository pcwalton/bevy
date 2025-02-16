//! Bindless resources.

use std::{
    borrow::Cow,
    num::{NonZeroU32, NonZeroU64},
};

use bevy_derive::{Deref, DerefMut};
use wgpu::{
    BindGroupLayoutEntry, SamplerBindingType, ShaderStages, TextureSampleType, TextureViewDimension,
};

use crate::render_resource::binding_types::storage_buffer_read_only_sized;

use super::binding_types::{
    sampler, texture_1d, texture_2d, texture_2d_array, texture_3d, texture_cube, texture_cube_array,
};

#[cfg(any(target_os = "macos", target_os = "ios"))]
pub const AUTO_BINDLESS_SLOT_COUNT: u32 = 64;
#[cfg(not(any(target_os = "macos", target_os = "ios")))]
pub const AUTO_BINDLESS_SLOT_COUNT: u32 = 2048;

pub static BINDING_NUMBERS: [(BindlessResourceType, BindingNumber); 9] = [
    (
        BindlessResourceType::SamplerFiltering,
        BindingNumber(1),
    ),
    (
        BindlessResourceType::SamplerNonFiltering,
        BindingNumber(2),
    ),
    (
        BindlessResourceType::SamplerComparison,
        BindingNumber(3),
    ),
    (BindlessResourceType::Texture1d, BindingNumber(4)),
    (BindlessResourceType::Texture2d, BindingNumber(5)),
    (
        BindlessResourceType::Texture2dArray,
        BindingNumber(6),
    ),
    (BindlessResourceType::Texture3d, BindingNumber(7)),
    (BindlessResourceType::TextureCube, BindingNumber(8)),
    (
        BindlessResourceType::TextureCubeArray,
        BindingNumber(9),
    ),
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
    pub binding_number: BindingNumber,
    pub bindless_index: BindlessIndex,
    pub element_size: usize,
}

/// The index of the actual binding in the bind group.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Deref, DerefMut)]
pub struct BindingNumber(pub u32);

/// The index in the bindless table.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Deref, DerefMut)]
pub struct BindlessIndex(pub u32);

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
        .build(0, ShaderStages::all()),
        sampler(SamplerBindingType::Filtering)
            .count(bindless_slot_count)
            .build(1, ShaderStages::all()),
        sampler(SamplerBindingType::NonFiltering)
            .count(bindless_slot_count)
            .build(2, ShaderStages::all()),
        sampler(SamplerBindingType::Comparison)
            .count(bindless_slot_count)
            .build(3, ShaderStages::all()),
        texture_1d(TextureSampleType::Float { filterable: true })
            .count(bindless_slot_count)
            .build(4, ShaderStages::all()),
        texture_2d(TextureSampleType::Float { filterable: true })
            .count(bindless_slot_count)
            .build(5, ShaderStages::all()),
        texture_2d_array(TextureSampleType::Float { filterable: true })
            .count(bindless_slot_count)
            .build(6, ShaderStages::all()),
        texture_3d(TextureSampleType::Float { filterable: true })
            .count(bindless_slot_count)
            .build(7, ShaderStages::all()),
        texture_cube(TextureSampleType::Float { filterable: true })
            .count(bindless_slot_count)
            .build(8, ShaderStages::all()),
        texture_cube_array(TextureSampleType::Float { filterable: true })
            .count(bindless_slot_count)
            .build(9, ShaderStages::all()),
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
    pub fn binding_number(&self) -> Option<&'static BindingNumber> {
        match BINDING_NUMBERS.binary_search_by_key(self, |(key, _)| *key) {
            Ok(binding_number) => Some(&BINDING_NUMBERS[binding_number].1),
            Err(_) => None,
        }
    }
}

impl TryFrom<BindlessResourceType> for BindingNumber {
    type Error = ();
    fn try_from(bindless_resource_type: BindlessResourceType) -> Result<Self, Self::Error> {
        match bindless_resource_type.binding_number() {
            Some(binding_number_ref) => Ok(*binding_number_ref),
            None => Err(()),
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

impl From<u32> for BindlessIndex {
    fn from(value: u32) -> Self {
        Self(value)
    }
}

impl From<u32> for BindingNumber {
    fn from(value: u32) -> Self {
        Self(value)
    }
}
