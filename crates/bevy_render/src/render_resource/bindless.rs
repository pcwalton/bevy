//! Bindless resources.

use std::num::NonZeroU32;

use wgpu::{SamplerBindingType, ShaderStages, TextureSampleType};

use crate::renderer::RenderDevice;

use super::{
    binding_types::{
        sampler, texture_1d, texture_2d, texture_2d_array, texture_3d, texture_cube,
        texture_cube_array,
    },
    BindGroupLayout, BindGroupLayoutEntries,
};

pub fn create_bindless_bind_group_layout(
    render_device: &RenderDevice,
    bindless_slot_count: u32,
) -> BindGroupLayout {
    let bindless_slot_count =
        NonZeroU32::new(bindless_slot_count).expect("Bindless slot count must be nonzero");

    render_device.create_bind_group_layout(
        "bindless bind group layout",
        &BindGroupLayoutEntries::sequential(
            ShaderStages::FRAGMENT,
            (
                sampler(SamplerBindingType::Filtering).count(bindless_slot_count),
                sampler(SamplerBindingType::NonFiltering).count(bindless_slot_count),
                sampler(SamplerBindingType::Comparison).count(bindless_slot_count),
                texture_1d(TextureSampleType::Float { filterable: true })
                    .count(bindless_slot_count),
                texture_2d(TextureSampleType::Float { filterable: true })
                    .count(bindless_slot_count),
                texture_2d_array(TextureSampleType::Float { filterable: true })
                    .count(bindless_slot_count),
                texture_3d(TextureSampleType::Float { filterable: true })
                    .count(bindless_slot_count),
                texture_cube(TextureSampleType::Float { filterable: true })
                    .count(bindless_slot_count),
                texture_cube_array(TextureSampleType::Float { filterable: true })
                    .count(bindless_slot_count),
            ),
        ),
    )
}
