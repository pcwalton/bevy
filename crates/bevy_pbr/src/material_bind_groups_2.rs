use std::{marker::PhantomData, num::NonZeroU32, ops::Index};

use bevy_derive::{Deref, DerefMut};
use bevy_platform_support::collections::HashMap;
use bevy_reflect::{prelude::ReflectDefault, Reflect};
use bevy_render::{
    render_resource::{
        BindGroup, BindGroupEntry, BindGroupLayout, BindGroupLayoutEntries, BindingResource,
        BindlessDescriptor, BindlessResourceType, Buffer, BufferBinding, BufferId, BufferUsages,
        OwnedBindingResource, RawBufferVec, Sampler, SamplerId, ShaderStages, TextureSampleType,
        TextureView, TextureViewDimension, TextureViewId, UnpreparedBindGroup, WgpuSampler,
        WgpuTextureView,
    },
    renderer::RenderDevice,
    texture::FallbackImage,
};
use tracing::{error, trace};

use crate::Material;

// FIXME: Make this something that can be specified in `AsBindGroup`.
const BINDINGS_SIZE: u32 = 32;

/// The minimum byte size of each fallback buffer.
const MIN_FALLBACK_BUFFER_SIZE: u64 = 16;

pub struct MaterialBindGroupAllocator<M>
where
    M: Material,
{
    slabs: Vec<MaterialBindlessSlab<M>>,
    bind_group_layout: BindGroupLayout,
    bindless_descriptor: BindlessDescriptor,
    slab_capacity: u32,
}

struct MaterialBindlessSlab<M>
where
    M: Material,
{
    pub bind_group: Option<BindGroup>,
    bindless_slot_count: u32,

    bindings: MaterialBindlessSlabBindings<M>,

    samplers: HashMap<BindlessResourceType, MaterialBindlessBindingArray<Sampler>>,
    textures: HashMap<BindlessResourceType, MaterialBindlessBindingArray<TextureView>>,
    buffers: HashMap<u32, MaterialBindlessBindingArray<Buffer>>,

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

/// The index of each material bind group.
///
/// In bindless mode, each bind group contains multiple materials. In
/// non-bindless mode, each bind group contains only one material.
#[derive(Clone, Copy, Debug, Default, Reflect, PartialEq, Deref, DerefMut)]
#[reflect(Default)]
pub struct MaterialBindGroupIndex(pub u32);

impl From<u32> for MaterialBindGroupIndex {
    fn from(value: u32) -> Self {
        MaterialBindGroupIndex(value)
    }
}

/// The index of the slot containing material data within each material bind
/// group.
///
/// In bindless mode, this slot is needed to locate the material data in each
/// bind group, since multiple materials are packed into a single slab. In
/// non-bindless mode, this slot is always 0.
#[derive(Clone, Copy, Debug, Default, Reflect, Deref, DerefMut)]
#[reflect(Default)]
pub struct MaterialBindGroupSlot(pub u32);

impl From<u32> for MaterialBindGroupSlot {
    fn from(value: u32) -> Self {
        MaterialBindGroupSlot(value)
    }
}

impl From<MaterialBindGroupSlot> for u32 {
    fn from(value: MaterialBindGroupSlot) -> Self {
        value.0
    }
}

impl<'a> From<&'a OwnedBindingResource> for BindingResourceId {
    fn from(value: &'a OwnedBindingResource) -> Self {
        match *value {
            OwnedBindingResource::Buffer(ref buffer) => BindingResourceId::Buffer(buffer.id()),
            OwnedBindingResource::TextureView(ref texture_view_dimension, ref texture_view) => {
                BindingResourceId::TextureView(*texture_view_dimension, texture_view.id())
            }
            OwnedBindingResource::Sampler(_, ref sampler) => {
                BindingResourceId::Sampler(sampler.id())
            }
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

impl<M> MaterialBindGroupAllocator<M>
where
    M: Material,
{
    pub fn new(render_device: &RenderDevice) -> MaterialBindGroupAllocator<M> {
        MaterialBindGroupAllocator {
            slabs: vec![],
            bind_group_layout: M::bind_group_layout(render_device),
            bindless_descriptor: M::bindless_descriptor().expect("TODO: non-bindless"),
            slab_capacity: M::bindless_slot_count()
                .expect("TODO: non-bindless")
                .resolve(),
        }
    }

    pub fn allocate(
        &mut self,
        mut unprepared_bind_group: UnpreparedBindGroup<M::Data>,
    ) -> MaterialBindingId {
        for (slab_index, slab) in slabs.iter_mut().enumerate() {
            trace!("Trying to allocate in slab {}", slab_index);
            match slab.try_allocate(unprepared_bind_group) {
                Ok(slot) => {
                    return MaterialBindingId {
                        group: MaterialBindGroupIndex(slab_index as u32),
                        slot,
                    };
                }
                Err(bind_group) => unprepared_bind_group = bind_group,
            }
        }

        slabs.push(MaterialBindlessSlab::new());
        slabs
            .last_mut()
            .expect("We just pushed a slab")
            .try_allocate(unprepared_bind_group)
            .expect("An allocation into an empty slab should always succeed")
    }
}

impl<M> MaterialBindlessSlab<M>
where
    M: Material,
{
    fn try_allocate(
        &mut self,
        unprepared_bind_group: UnpreparedBindGroup<M::Data>,
    ) -> Result<MaterialBindGroupSlot, UnpreparedBindGroup<M::Data>> {
        let mut pre_existing_resources = HashMap::default();
        let mut needed_free_buffer_slots = vec![];
        let mut needed_free_fixed_resource_slots = HashMap::default();

        // Locate pre-existing resources.

        for &(binding_index, ref owned_binding_resource) in unprepared_bind_group.bindings.iter() {
            match *owned_binding_resource {
                OwnedBindingResource::Buffer(ref buffer) => {
                    let Some(binding_array) = self.buffers.get(&binding_index) else {
                        error!(
                            "Binding array wasn't present for buffer at index {}",
                            binding_index
                        );
                        return Err(unprepared_bind_group);
                    };
                    match binding_array.search(buffer.id()) {
                        Some(slot) => {
                            pre_existing_resources.insert(binding_index, slot);
                        }
                        None => needed_free_buffer_slots.push(binding_index),
                    }
                }

                OwnedBindingResource::TextureView(texture_view_dimension, ref texture_view) => {
                    let bindless_resource_type = BindlessResourceType::from(texture_view_dimension);
                    match self
                        .textures
                        .get(&bindless_resource_type)
                        .expect("Missing binding array for texture")
                        .search(texture_view.id())
                    {
                        Some(slot) => {
                            pre_existing_resources.insert(binding_index, slot);
                        }
                        None => {
                            *needed_free_fixed_resource_slots
                                .entry(bindless_resource_type)
                                .or_default() += 1
                        }
                    }
                }

                OwnedBindingResource::Sampler(sampler_binding_type, ref sampler) => {
                    let bindless_resource_type = BindlessResourceType::from(sampler_binding_type);
                    match self
                        .samplers
                        .get(&bindless_resource_type)
                        .expect("Missing binding array for sampler")
                        .search(sampler.id())
                    {
                        Some(slot) => {
                            pre_existing_resources.insert(binding_index, slot);
                        }
                        None => {
                            *needed_free_fixed_resource_slots
                                .entry(bindless_resource_type)
                                .or_default() += 1
                        }
                    }
                }
            }
        }

        // Check to see if we have enough free space in buffer binding arrays.
        for (binding_index, needed_slot_count) in &needed_free_buffer_slots {
            if !self
                .buffers
                .get(binding_index)
                .expect("Buffer should be present")
                .has_free_slots(needed_slot_count)
            {
                trace!(
                    "Buffer at binding {} is full, can't allocate",
                    binding_index
                );
                return Err(unprepared_bind_group);
            }
        }

        // Check to see if we have enough free space in fixed resource binding
        // arrays.
        for (bindless_resource_type, needed_slot_count) in &needed_free_fixed_resource_slots {
            if let Some(sampler_binding_array) = self.samplers.get(bindless_resource_type) {
                if !sampler_binding_array.has_free_slots(needed_slot_count) {
                    trace!(
                        "Sampler binding array {:?} is full, can't allocate",
                        bindless_resource_type
                    );
                    return Err(unprepared_bind_group);
                }
                continue;
            }

            let texture_binding_array = self.textures.get(bindless_resource_type).expect(
                "This bindless resource type should describe either a live sampler binding array \
                or a live texture binding array",
            );
            if !texture_binding_array.has_free_slots(needed_slot_count) {
                trace!(
                    "Texture binding array {:?} is full, can't allocate",
                    bindless_resource_type
                );
                return Err(unprepared_bind_group);
            }
        }

        // OK, we can allocate in this slab.
        let mut slot = self.free_slots.pop().unwrap_or(self.live_allocation_count);
        self.live_allocation_count += 1;

        let mut allocated_resource_slots = HashMap::default();

        for (binding_index, owned_binding_resource) in unprepared_bind_group.bindings.drain(..) {
            // If this is an other reference to an object we've already
            // allocated, just bump its reference count.
            if let Some(pre_existing_resource_slot) = pre_existing_resources.get(&binding_index) {
                allocated_resource_slots.insert(binding_index, pre_existing_resource_slot);

                match owned_binding_resource {
                    OwnedBindingResource::Buffer(_) => {
                        self.buffers
                            .get_mut(&binding_index)
                            .expect("Buffer binding array should exist")
                            .bindings
                            .get_mut(pre_existing_resource_slot)
                            .expect("Slot should exist")
                            .ref_count += 1;
                    }
                    OwnedBindingResource::TextureView(texture_view_dimension, _) => {
                        let bindless_resource_type =
                            BindlessResourceType::from(texture_view_dimension);
                        self.textures
                            .get_mut(&bindless_resource_type)
                            .expect("Texture binding array should exist")
                            .bindings
                            .get_mut(pre_existing_resource_slot)
                            .expect("Slot should exist")
                            .ref_count += 1;
                    }
                    OwnedBindingResource::Sampler(sampler_binding_type, _) => {
                        let bindless_resource_type =
                            BindlessResourceType::from(sampler_binding_type);
                        self.samplers
                            .get_mut(&bindless_resource_type)
                            .expect("Sampler binding array should exist")
                            .bindings
                            .get_mut(pre_existing_resource_slot)
                            .expect("Slot should exist")
                            .ref_count += 1;
                    }
                }

                continue;
            }

            // Otherwise, we need to insert it anew.
            match owned_binding_resource {
                OwnedBindingResource::Buffer(buffer) => {
                    let slot = self
                        .buffers
                        .get_mut(&binding_index)
                        .expect("Buffer binding array should exist")
                        .insert(buffer);
                    allocated_resource_slots.insert(binding_index, slot);
                }
                OwnedBindingResource::TextureView(texture_view_dimension, texture_view) => {
                    let bindless_resource_type = BindlessResourceType::from(texture_view_dimension);
                    let slot = self
                        .textures
                        .get_mut(&bindless_resource_type)
                        .expect("Texture array should exist")
                        .insert(texture_view);
                }
                OwnedBindingResource::Sampler(sampler_binding_type, sampler) => {
                    let bindless_resource_type = BindlessResourceType::from(sampler_binding_type);
                    let slot = self
                        .samplers
                        .get_mut(&bindless_resource_type)
                        .expect("Sampler should exist")
                        .insert(sampler);
                }
            }
        }

        // Serialize the allocated resource slots.
        self.bindings.set(slot, allocated_resource_slots);

        // Invalidate the cached bind group.
        self.bind_group = None;

        Ok(MaterialBindGroupSlot(slot))
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
        binding_numbers: &'a HashMap<BindlessResourceType, u32>,
        bindless_descriptor: &'a BindlessDescriptor,
        fallback_buffers: &'a MaterialFallbackBuffers,
        fallback_sampler: &'a Sampler,
        fallback_image: &'a FallbackImage,
    ) -> Vec<(&'a u32, BindingResourceArray<'a>)> {
        let bindless_descriptor = M::bindless_descriptor().expect("Need a bindless descriptor");

        let mut binding_resource_arrays = vec![];
        let mut binding_number_iterator = binding_numbers.iter();

        // Build sampler bindings.
        for (bindless_resource_type, sampler_bindless_binding_array) in self.samplers.iter() {
            let sampler_bindings = sampler_bindless_binding_array
                .bindings
                .iter()
                .map(|maybe_bindless_binding| match *maybe_bindless_binding {
                    Some(ref bindless_binding) => &bindless_binding.resource,
                    None => &**fallback_sampler,
                })
                .collect();
            binding_resource_arrays.push((
                binding_numbers
                    .get(bindless_resource_type)
                    .expect("Binding number not present"),
                BindingResourceArray::Samplers(sampler_bindings),
            ));
        }

        // Build texture bindings.
        for (bindless_resource_type, fallback_image, texture_view_dimension) in [
            (
                BindlessResourceType::Texture1d,
                &fallback_image.d1,
                TextureViewDimension::D1,
            ),
            (
                BindlessResourceType::Texture2d,
                &fallback_image.d2,
                TextureViewDimension::D2,
            ),
            (
                BindlessResourceType::Texture2dArray,
                &fallback_image.d2_array,
                TextureViewDimension::D2Array,
            ),
            (
                BindlessResourceType::Texture3d,
                &fallback_image.d3,
                TextureViewDimension::D3,
            ),
            (
                BindlessResourceType::TextureCube,
                &fallback_image.cube,
                TextureViewDimension::Cube,
            ),
            (
                BindlessResourceType::TextureCubeArray,
                &fallback_image.cube_array,
                TextureViewDimension::CubeArray,
            ),
        ] {
            let Some(texture_bindless_binding_array) = self.textures.get(&bindless_resource_type)
            else {
                continue;
            };
            let texture_bindings = texture_bindless_binding_array
                .bindings
                .iter()
                .map(|maybe_bindless_binding| match *maybe_bindless_binding {
                    Some(ref bindless_binding) => &*bindless_binding.resource,
                    None => &*fallback_image.texture_view,
                })
                .collect();
            binding_resource_arrays.push((
                binding_numbers
                    .get(&bindless_resource_type)
                    .expect("Binding number not present"),
                BindingResourceArray::TextureViews(texture_view_dimension, texture_bindings),
            ));
        }

        // Build buffer bindings.
        for bindless_buffer_descriptor in bindless_descriptor.buffers.iter() {
            let Some(buffer_bindless_binding_array) =
                self.buffers.get(&bindless_buffer_descriptor.index)
            else {
                error!(
                    "Slab didn't contain a binding array for buffer {}",
                    bindless_buffer_descriptor.index
                );
                continue;
            };
            let buffer_bindings = buffer_bindless_binding_array
                .bindings
                .iter()
                .map(|maybe_bindless_binding| {
                    let buffer = match *maybe_bindless_binding {
                        None => {
                            todo!("TODO: populate fallback buffers")
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
                &bindless_buffer_descriptor.index,
                BindingResourceArray::Buffers(buffer_bindings),
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
