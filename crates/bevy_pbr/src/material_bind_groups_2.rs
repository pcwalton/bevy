use std::{marker::PhantomData, num::NonZeroU32, ops::Index};

use bevy_derive::{Deref, DerefMut};
use bevy_platform_support::collections::HashMap;
use bevy_reflect::Reflect;
use bevy_render::{
    render_resource::{
        binding_types::{
            sampler, storage_buffer, storage_buffer_read_only, texture_1d, texture_2d,
            texture_2d_array, texture_3d, texture_cube, texture_cube_array,
        },
        BindGroup, BindGroupEntry, BindGroupLayout, BindGroupLayoutEntries, BindingResource,
        BindingType, Buffer, BufferBinding, BufferId, BufferUsages, OwnedBindingResource,
        RawBufferVec, Sampler, SamplerBindingType, SamplerId, ShaderStages, TextureSampleType,
        TextureView, TextureViewDimension, TextureViewId, UnpreparedBindGroup, WgpuSampler,
        WgpuTextureView,
    },
    renderer::RenderDevice,
    texture::FallbackImage,
};

use crate::Material;

// FIXME: Make this something that can be specified in `AsBindGroup`.
const BINDINGS_SIZE: u32 = 32;

/// The minimum byte size of each fallback buffer.
const MIN_FALLBACK_BUFFER_SIZE: u64 = 16;

pub struct MaterialBindGroupAllocator2<M>
where
    M: Material,
{
    slabs: Vec<MaterialBindlessSlab<M>>,
    bind_group_layout: BindGroupLayout,
    slab_capacity: u32,
}

struct MaterialBindlessSlab<M>
where
    M: Material,
{
    pub bind_group: Option<BindGroup>,
    bindless_slot_count: u32,

    bindings: MaterialBindlessSlabBindings<M>,
    buffers: MaterialBindlessBindingArray<Buffer>,
    rw_buffers_rw: MaterialBindlessBindingArray<Buffer>,
    samplers_filtering: MaterialBindlessBindingArray<Sampler>,
    samplers_non_filtering: MaterialBindlessBindingArray<Sampler>,
    samplers_comparison: MaterialBindlessBindingArray<Sampler>,
    textures_1d: MaterialBindlessBindingArray<TextureView>,
    textures_2d: MaterialBindlessBindingArray<TextureView>,
    textures_2d_array: MaterialBindlessBindingArray<TextureView>,
    textures_3d: MaterialBindlessBindingArray<TextureView>,
    textures_cube: MaterialBindlessBindingArray<TextureView>,
    textures_cube_array: MaterialBindlessBindingArray<TextureView>,

    live_allocation_count: u32,
}

struct MaterialBindlessSlabBindings<M>
where
    M: Material,
{
    buffer: RawBufferVec<u32>,
    phantom: PhantomData<M>,
}

struct MaterialBindlessBindingArray<R> {
    bindings: Vec<Option<MaterialBindlessBinding<R>>>,
    resource_to_slot: HashMap<BindingResourceId, u32>,
    free_slots: Vec<u32>,
}

struct MaterialBindlessBinding<R> {
    resource: R,
    ref_count: u32,
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
enum BindingResourceId {
    Buffer(BufferId),
    TextureView(TextureViewDimension, TextureViewId),
    Sampler(SamplerId),
}

/// A temporary data structure that contains references to bindless resources.
///
/// We need this because the `wgpu` bindless API takes a slice of references.
/// Thus we need to create intermediate vectors of bindless resources in order
/// to satisfy the lifetime requirements.
enum BindingResourceArray<'a> {
    Buffers(Vec<BufferBinding<'a>>),
    TextureViews(TextureViewDimension, Vec<&'a WgpuTextureView>),
    Samplers(Vec<&'a WgpuSampler>),
}

struct MaterialFallbackBuffers(HashMap<u32, Buffer>);

#[derive(Clone, Copy, Debug, Default, Reflect)]
pub struct MaterialBindingId {
    /// The index of the bind group (slab) where the GPU data is located.
    pub group: MaterialBindGroupIndex,
    /// The slot within that bind group.
    pub slot: MaterialBindGroupSlot,
}

impl<'a> From<&'a OwnedBindingResource> for BindingResourceId {
    fn from(value: &'a OwnedBindingResource) -> Self {
        match *value {
            OwnedBindingResource::Buffer(ref buffer) => BindingResourceId::Buffer(buffer.id()),
            OwnedBindingResource::TextureView(ref texture_view_dimension, ref texture_view) => {
                BindingResourceId::TextureView(*texture_view_dimension, texture_view.id())
            }
            OwnedBindingResource::Sampler(ref sampler) => BindingResourceId::Sampler(sampler.id()),
        }
    }
}

impl<M> Index<u32> for MaterialBindlessSlabBindings<M>
where
    M: Material,
{
    type Output = [u32];

    fn index(&self, index: u32) -> &Self::Output {
        &self.buffer.values()[(index as usize * BINDINGS_SIZE as usize)
            ..(((index as usize) + 1) * BINDINGS_SIZE as usize)]
    }
}

impl<M> MaterialBindlessSlabBindings<M>
where
    M: Material,
{
    fn new() -> MaterialBindlessSlabBindings<M> {
        MaterialBindlessSlabBindings {
            buffer: RawBufferVec::new(BufferUsages::STORAGE),
            phantom: PhantomData,
        }
    }
}

impl<M> MaterialBindlessSlab<M>
where
    M: Material,
{
    fn try_allocate(
        &mut self,
        unprepared_bind_group: UnpreparedBindGroup<M::Data>,
    ) -> Result<MaterialBindingId, UnpreparedBindGroup<M::Data>> {
        let mut slots = Vec::with_capacity(unprepared_bind_group.bindings.len());
        // TODO
        for &(binding_resource_id, ref owned_binding_resource) in
            unprepared_bind_group.bindings.iter()
        {
            match *owned_binding_resource {
                OwnedBindingResource::Buffer(ref buffer) => {}
                OwnedBindingResource::TextureView(texture_view_dimension, ref texture_view) => {
                    todo!()
                }
                OwnedBindingResource::Sampler(ref sampler) => todo!(),
            }
        }

        Ok(())
    }

    fn ensure_bind_group(
        &mut self,
        render_device: &RenderDevice,
        bind_group_layout: &BindGroupLayout,
        fallback_buffers: &MaterialFallbackBuffers,
        fallback_sampler: &Sampler,
        fallback_image: &FallbackImage,
    ) {
        if self.bind_group.is_some() {
            return;
        }

        let binding_resource_arrays =
            self.create_binding_resource_arrays(fallback_buffers, fallback_sampler, fallback_image);

        let bind_group_entries: Vec<_> = binding_resource_arrays
            .iter()
            .map(|&(&binding, ref binding_resource_array)| BindGroupEntry {
                binding,
                resource: match *binding_resource_array {
                    BindingResourceArray::Buffers(ref buffer_bindings) => {
                        BindingResource::BufferArray(&buffer_bindings[..])
                    }
                    BindingResourceArray::TextureViews(_, ref texture_views) => {
                        BindingResource::TextureViewArray(&texture_views[..])
                    }
                    BindingResourceArray::Samplers(ref samplers) => {
                        BindingResource::SamplerArray(&samplers[..])
                    }
                },
            })
            .collect();

        self.bind_group = Some(render_device.create_bind_group(
            M::label(),
            bind_group_layout,
            &bind_group_entries,
        ));
    }

    fn create_binding_resource_arrays<'a>(
        &'a self,
        binding_numbers: &[u32],
        fallback_buffers: &'a MaterialFallbackBuffers,
        fallback_sampler: &'a Sampler,
        fallback_image: &'a FallbackImage,
    ) -> Vec<(&'a u32, BindingResourceArray<'a>)> {
        let mut binding_resource_arrays = vec![];
        let mut binding_number_iterator = binding_numbers.iter();

        // Build buffer bindings.
        for buffer_bindless_binding_array in [&self.buffers, &self.rw_buffers_rw] {
            let buffer_bindings = buffer_bindless_binding_array
                .bindings
                .iter()
                .map(|maybe_bindless_binding| {
                    let buffer = match *maybe_bindless_binding {
                        None => {
                            // TODO: Populate fallback buffers with a buffer of
                            // the appropriate size
                            todo!()
                        }
                        Some(ref bindless_binding) => &bindless_binding.resource,
                    };
                    BufferBinding {
                        buffer,
                        offset: 0,
                        size: None,
                    }
                })
                .collect();
            binding_resource_arrays.push((
                binding_numbers.next().expect("Not enough binding numbers"),
                BindingResourceArray::Buffers(buffer_bindings),
            ));
        }

        // Build sampler bindings.
        for sampler_bindless_binding_array in [
            &self.samplers_filtering,
            &self.samplers_non_filtering,
            &self.samplers_comparison,
        ] {
            let sampler_bindings = sampler_bindless_binding_array
                .bindings
                .iter()
                .map(|maybe_bindless_binding| match *maybe_bindless_binding {
                    Some(ref bindless_binding) => &bindless_binding.resource,
                    None => &**fallback_sampler,
                })
                .collect();
            binding_resource_arrays.push((
                binding_numbers.next().expect("Not enough binding numbers"),
                BindingResourceArray::Samplers(sampler_bindings),
            ));
        }

        // Build texture bindings.
        for (texture_bindless_binding_array, fallback_image, texture_view_dimension) in [
            (
                &self.textures_1d,
                &fallback_image.d1,
                TextureViewDimension::D1,
            ),
            (
                &self.textures_2d,
                &fallback_image.d2,
                TextureViewDimension::D2,
            ),
            (
                &self.textures_2d_array,
                &fallback_image.d2_array,
                TextureViewDimension::D2Array,
            ),
            (
                &self.textures_3d,
                &fallback_image.d3,
                TextureViewDimension::D3,
            ),
            (
                &self.textures_cube,
                &fallback_image.cube,
                TextureViewDimension::Cube,
            ),
            (
                &self.textures_cube_array,
                &fallback_image.cube_array,
                TextureViewDimension::CubeArray,
            ),
        ] {
            let texture_bindings = texture_bindless_binding_array
                .bindings
                .iter()
                .map(|maybe_bindless_binding| match *maybe_bindless_binding {
                    Some(ref bindless_binding) => &*bindless_binding.resource,
                    None => &*fallback_image.texture_view,
                })
                .collect();
            binding_resource_arrays.push((
                binding_numbers.next().expect("Not enough binding numbers"),
                BindingResourceArray::TextureViews(texture_view_dimension, &texture_bindings),
            ));
        }

        binding_resource_arrays
    }
}

impl<R> MaterialBindlessBindingArray<R> {
    fn is_empty(&self) -> bool {
        self.resource_to_slot.is_empty()
    }
}

fn create_bindless_material_bind_group_layout<M>(
    render_device: &RenderDevice,
    bindless_slot_count: u32,
) -> BindGroupLayout
where
    M: Material,
{
    let bindless_slot_count =
        NonZeroU32::new(bindless_slot_count).expect("Bindless slot count must be nonzero");

    render_device.create_bind_group_layout(
        "bindless material bind group layout",
        &BindGroupLayoutEntries::sequential(
            ShaderStages::FRAGMENT,
            (
                // TODO: Maybe use a more accurate size?
                storage_buffer_read_only::<u32>(false),
                storage_buffer_read_only::<u32>(false),
                storage_buffer::<u32>(false),
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
