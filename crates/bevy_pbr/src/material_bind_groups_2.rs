use core::{marker::PhantomData, ops::Index};

use bevy_derive::{Deref, DerefMut};
use bevy_ecs::{
    resource::Resource,
    world::{FromWorld, World},
};
use bevy_platform_support::collections::HashMap;
use bevy_reflect::{prelude::ReflectDefault, Reflect};
use bevy_render::{
    render_resource::{
        BindGroup, BindGroupEntry, BindGroupLayout, BindingNumber, BindingResource,
        BindlessDescriptor, BindlessIndex, BindlessResourceType, Buffer, BufferBinding, BufferId,
        BufferUsages, CompareFunction, FilterMode, OwnedBindingResource, RawBufferVec, Sampler,
        SamplerDescriptor, SamplerId, TextureView, TextureViewDimension, TextureViewId,
        UnpreparedBindGroup, WgpuSampler, WgpuTextureView,
    },
    renderer::{RenderDevice, RenderQueue},
    texture::FallbackImage,
};
use bevy_utils::default;
use tracing::{error, trace};

use crate::Material;

// FIXME: Make this something that can be specified in `AsBindGroup`.
const BINDINGS_SIZE: u32 = 32;

#[derive(Resource)]
pub struct MaterialBindGroupAllocator<M>
where
    M: Material,
{
    slabs: Vec<MaterialBindlessSlab<M>>,
    bind_group_layout: BindGroupLayout,
    bindless_descriptor: BindlessDescriptor,
    slab_capacity: u32,
}

pub struct MaterialBindlessSlab<M>
where
    M: Material,
{
    /// Bind group.
    ///
    /// If this is `None`, then the bind group is dirty and needs to be
    /// regenerated.
    bind_group: Option<BindGroup>,

    bindings: MaterialBindlessSlabBindings<M>,

    samplers: HashMap<BindlessResourceType, MaterialBindlessBindingArray<Sampler>>,
    textures: HashMap<BindlessResourceType, MaterialBindlessBindingArray<TextureView>>,
    buffers: HashMap<BindlessIndex, MaterialBindlessBindingArray<Buffer>>,

    extra_data: Vec<Option<M::Data>>,

    live_allocation_count: u32,
}

struct MaterialBindlessSlabBindings<M>
where
    M: Material,
{
    buffer: RawBufferVec<u32>,
    buffer_dirty: BufferDirtyState,
    free_slots: Vec<MaterialBindGroupSlot>,
    phantom: PhantomData<M>,
}

struct MaterialBindlessBindingArray<R>
where
    R: GetBindingResourceId,
{
    // This is necessary because of the `wgpu` API.
    binding_number: BindingNumber,
    bindings: Vec<Option<MaterialBindlessBinding<R>>>,
    resource_type: BindlessResourceType,
    resource_to_slot: HashMap<BindingResourceId, u32>,
    free_slots: Vec<u32>,
    len: u32,
    capacity: u32,
}

struct MaterialBindlessBinding<R>
where
    R: GetBindingResourceId,
{
    resource: R,
    ref_count: u32,
}

#[derive(Resource)]
pub struct FallbackBindlessResources {
    /// A dummy sampler that we fill unused slots in bindless sampler arrays
    /// with.
    filtering_sampler: Sampler,
    non_filtering_sampler: Sampler,
    comparison_sampler: Sampler,
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
    TextureViews(Vec<&'a WgpuTextureView>),
    Samplers(Vec<&'a WgpuSampler>),
}

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

enum BufferDirtyState {
    Clean,
    NeedsReserve,
    NeedsUpload,
}

trait GetBindingResourceId {
    fn binding_resource_id(&self, resource_type: BindlessResourceType) -> BindingResourceId;
}

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

impl GetBindingResourceId for Buffer {
    fn binding_resource_id(&self, _: BindlessResourceType) -> BindingResourceId {
        BindingResourceId::Buffer(self.id())
    }
}

impl GetBindingResourceId for Sampler {
    fn binding_resource_id(&self, _: BindlessResourceType) -> BindingResourceId {
        BindingResourceId::Sampler(self.id())
    }
}

impl GetBindingResourceId for TextureView {
    fn binding_resource_id(&self, resource_type: BindlessResourceType) -> BindingResourceId {
        let texture_view_dimension = match resource_type {
            BindlessResourceType::Texture1d => TextureViewDimension::D1,
            BindlessResourceType::Texture2d => TextureViewDimension::D2,
            BindlessResourceType::Texture2dArray => TextureViewDimension::D2Array,
            BindlessResourceType::Texture3d => TextureViewDimension::D3,
            BindlessResourceType::TextureCube => TextureViewDimension::Cube,
            BindlessResourceType::TextureCubeArray => TextureViewDimension::CubeArray,
            _ => panic!("Resource type is not a texture"),
        };
        BindingResourceId::TextureView(texture_view_dimension, self.id())
    }
}

impl<M> MaterialBindlessSlabBindings<M>
where
    M: Material,
{
    fn new(bindless_descriptor: &BindlessDescriptor) -> MaterialBindlessSlabBindings<M> {
        // Preallocate space for one bindings table, so that there will always be a buffer.
        let mut buffer = RawBufferVec::new(BufferUsages::STORAGE);
        for _ in 0..bindless_descriptor.resources.len() {
            buffer.push(0);
        }

        MaterialBindlessSlabBindings {
            buffer,
            free_slots: vec![],
            buffer_dirty: BufferDirtyState::NeedsReserve,
            phantom: PhantomData,
        }
    }

    fn get(&self, slot: MaterialBindGroupSlot, bindless_descriptor: &BindlessDescriptor) -> &[u32] {
        let struct_size = bindless_descriptor.resources.len();
        let start = struct_size * slot.0 as usize;
        &self.buffer.values()[start..(start + struct_size)]
    }

    fn set(
        &mut self,
        slot: MaterialBindGroupSlot,
        allocated_resource_slots: &HashMap<BindlessIndex, u32>,
        bindless_descriptor: &BindlessDescriptor,
    ) {
        let table_len = bindless_descriptor.resources.len();
        let range = (slot.0 as usize * table_len)..((slot.0 as usize + 1) * table_len);
        while self.buffer.len() < range.end {
            self.buffer.push(0);
        }

        for (&bindless_index, &resource_slot) in allocated_resource_slots {
            self.buffer
                .set(*bindless_index + range.start as u32, resource_slot);
        }

        self.buffer_dirty = BufferDirtyState::NeedsReserve;
    }

    fn prepare_buffer(&mut self, render_device: &RenderDevice) {
        match self.buffer_dirty {
            BufferDirtyState::Clean | BufferDirtyState::NeedsUpload => {}
            BufferDirtyState::NeedsReserve => {
                let capacity = self.buffer.len();
                self.buffer.reserve(capacity, render_device);
                self.buffer_dirty = BufferDirtyState::NeedsUpload;
            }
        }
    }

    fn write_buffer(&mut self, render_device: &RenderDevice, render_queue: &RenderQueue) {
        match self.buffer_dirty {
            BufferDirtyState::Clean => {}
            BufferDirtyState::NeedsReserve | BufferDirtyState::NeedsUpload => {
                self.buffer.write_buffer(render_device, render_queue);
                self.buffer_dirty = BufferDirtyState::Clean;
            }
        }
    }

    fn free(&mut self, slot: MaterialBindGroupSlot) {
        // Note that this doesn't actually dirty the buffer. We just leave the
        // old contents there.
        self.free_slots.push(slot);
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
        for (slab_index, slab) in self.slabs.iter_mut().enumerate() {
            trace!("Trying to allocate in slab {}", slab_index);
            match slab.try_allocate(unprepared_bind_group, &self.bindless_descriptor) {
                Ok(slot) => {
                    return MaterialBindingId {
                        group: MaterialBindGroupIndex(slab_index as u32),
                        slot,
                    };
                }
                Err(bind_group) => unprepared_bind_group = bind_group,
            }
        }

        let group = MaterialBindGroupIndex(self.slabs.len() as u32);
        //println!("Allocation in slab failed, creating new slab {:?}", group);
        self.slabs.push(MaterialBindlessSlab::new(
            self.slab_capacity,
            &self.bindless_descriptor,
        ));

        let Ok(slot) = self
            .slabs
            .last_mut()
            .expect("We just pushed a slab")
            .try_allocate(unprepared_bind_group, &self.bindless_descriptor)
        else {
            panic!("An allocation into an empty slab should always succeed")
        };

        MaterialBindingId { group, slot }
    }

    pub fn free(&mut self, material_binding_id: MaterialBindingId) {
        self.slabs
            .get_mut(material_binding_id.group.0 as usize)
            .expect("Slab should exist")
            .free(material_binding_id.slot, &self.bindless_descriptor);
    }

    pub fn get(&self, group: MaterialBindGroupIndex) -> Option<&MaterialBindlessSlab<M>> {
        self.slabs.get(group.0 as usize)
    }

    pub fn prepare_bind_groups(
        &mut self,
        render_device: &RenderDevice,
        fallback_bindless_resources: &FallbackBindlessResources,
        fallback_image: &FallbackImage,
    ) {
        for slab in &mut self.slabs {
            slab.prepare(
                render_device,
                &self.bind_group_layout,
                fallback_bindless_resources,
                fallback_image,
                &self.bindless_descriptor,
            );
        }
    }

    pub fn write_buffers(&mut self, render_device: &RenderDevice, render_queue: &RenderQueue) {
        for slab in &mut self.slabs {
            slab.write_buffer(render_device, render_queue);
        }
    }
}

impl<M> FromWorld for MaterialBindGroupAllocator<M>
where
    M: Material,
{
    fn from_world(world: &mut World) -> Self {
        let render_device = world.resource::<RenderDevice>();
        MaterialBindGroupAllocator::new(render_device)
    }
}

impl<M> MaterialBindlessSlab<M>
where
    M: Material,
{
    fn try_allocate(
        &mut self,
        mut unprepared_bind_group: UnpreparedBindGroup<M::Data>,
        bindless_descriptor: &BindlessDescriptor,
    ) -> Result<MaterialBindGroupSlot, UnpreparedBindGroup<M::Data>> {
        let mut pre_existing_resources = HashMap::new();
        let mut needed_free_buffer_slots = vec![];
        let mut needed_free_fixed_resource_slots = HashMap::new();

        // Locate pre-existing resources.

        for &(bindless_index, ref owned_binding_resource) in unprepared_bind_group.bindings.iter() {
            let bindless_index = BindlessIndex(bindless_index);
            match *owned_binding_resource {
                OwnedBindingResource::Buffer(ref buffer) => {
                    let Some(binding_array) = self.buffers.get(&bindless_index) else {
                        error!(
                            "Binding array wasn't present for buffer at index {:?}",
                            bindless_index
                        );
                        return Err(unprepared_bind_group);
                    };
                    match binding_array.find(BindingResourceId::Buffer(buffer.id())) {
                        Some(slot) => {
                            pre_existing_resources.insert(bindless_index, slot);
                        }
                        None => needed_free_buffer_slots.push(bindless_index),
                    }
                }

                OwnedBindingResource::TextureView(texture_view_dimension, ref texture_view) => {
                    let bindless_resource_type = BindlessResourceType::from(texture_view_dimension);
                    match self
                        .textures
                        .get(&bindless_resource_type)
                        .expect("Missing binding array for texture")
                        .find(BindingResourceId::TextureView(
                            texture_view_dimension,
                            texture_view.id(),
                        )) {
                        Some(slot) => {
                            pre_existing_resources.insert(bindless_index, slot);
                        }
                        None => {
                            *needed_free_fixed_resource_slots
                                .entry(bindless_resource_type)
                                .or_default() += 1;
                        }
                    }
                }

                OwnedBindingResource::Sampler(sampler_binding_type, ref sampler) => {
                    let bindless_resource_type = BindlessResourceType::from(sampler_binding_type);
                    match self
                        .samplers
                        .get(&bindless_resource_type)
                        .expect("Missing binding array for sampler")
                        .find(BindingResourceId::Sampler(sampler.id()))
                    {
                        Some(slot) => {
                            pre_existing_resources.insert(bindless_index, slot);
                        }
                        None => {
                            *needed_free_fixed_resource_slots
                                .entry(bindless_resource_type)
                                .or_default() += 1;
                        }
                    }
                }
            }
        }

        // Check to see if we have enough free space in buffer binding arrays.
        for &binding_index in &needed_free_buffer_slots {
            if !self
                .buffers
                .get(&binding_index)
                .expect("Buffer should be present")
                .has_free_slots(1)
            {
                trace!(
                    "Buffer at binding {:?} is full, can't allocate",
                    binding_index
                );
                return Err(unprepared_bind_group);
            }
        }

        // Check to see if we have enough free space in fixed resource binding
        // arrays.
        for (bindless_resource_type, &needed_slot_count) in &needed_free_fixed_resource_slots {
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
        let slot = self
            .bindings
            .free_slots
            .pop()
            .unwrap_or(MaterialBindGroupSlot(self.live_allocation_count));
        self.live_allocation_count += 1;

        let mut allocated_resource_slots = HashMap::default();

        for (bindless_index, owned_binding_resource) in unprepared_bind_group.bindings.drain(..) {
            let bindless_index = BindlessIndex(bindless_index);
            // If this is an other reference to an object we've already
            // allocated, just bump its reference count.
            if let Some(pre_existing_resource_slot) = pre_existing_resources.get(&bindless_index) {
                allocated_resource_slots.insert(bindless_index, *pre_existing_resource_slot);

                match owned_binding_resource {
                    OwnedBindingResource::Buffer(_) => {
                        self.buffers
                            .get_mut(&bindless_index)
                            .expect("Buffer binding array should exist")
                            .bindings
                            .get_mut(*pre_existing_resource_slot as usize)
                            .and_then(|binding| binding.as_mut())
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
                            .get_mut(*pre_existing_resource_slot as usize)
                            .and_then(|binding| binding.as_mut())
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
                            .get_mut(*pre_existing_resource_slot as usize)
                            .and_then(|binding| binding.as_mut())
                            .expect("Slot should exist")
                            .ref_count += 1;
                    }
                }

                continue;
            }

            // Otherwise, we need to insert it anew.
            let binding_resource_id = BindingResourceId::from(&owned_binding_resource);
            match owned_binding_resource {
                OwnedBindingResource::Buffer(buffer) => {
                    let slot = self
                        .buffers
                        .get_mut(&bindless_index)
                        .expect("Buffer binding array should exist")
                        .insert(binding_resource_id, buffer);
                    allocated_resource_slots.insert(bindless_index, slot);
                    //println!("inserted buffer at {:?} = {:?}", slot, id);
                }
                OwnedBindingResource::TextureView(texture_view_dimension, texture_view) => {
                    let bindless_resource_type = BindlessResourceType::from(texture_view_dimension);
                    let slot = self
                        .textures
                        .get_mut(&bindless_resource_type)
                        .expect("Texture array should exist")
                        .insert(binding_resource_id, texture_view);
                    allocated_resource_slots.insert(bindless_index, slot);
                }
                OwnedBindingResource::Sampler(sampler_binding_type, sampler) => {
                    let bindless_resource_type = BindlessResourceType::from(sampler_binding_type);
                    let slot = self
                        .samplers
                        .get_mut(&bindless_resource_type)
                        .expect("Sampler should exist")
                        .insert(binding_resource_id, sampler);
                    allocated_resource_slots.insert(bindless_index, slot);
                }
            }
        }

        // Serialize the allocated resource slots.
        self.bindings
            .set(slot, &allocated_resource_slots, bindless_descriptor);

        // Insert extra data.
        if self.extra_data.len() < (*slot as usize + 1) {
            self.extra_data.resize_with(*slot as usize + 1, || None);
        }
        self.extra_data[*slot as usize] = Some(unprepared_bind_group.data);

        // Invalidate the cached bind group.
        self.bind_group = None;

        Ok(slot)
    }

    fn free(&mut self, slot: MaterialBindGroupSlot, bindless_descriptor: &BindlessDescriptor) {
        for (bindless_index, (bindless_resource_type, &bindless_binding)) in bindless_descriptor
            .resources
            .iter()
            .zip(self.bindings.get(slot, bindless_descriptor))
            .enumerate()
        {
            let bindless_index = BindlessIndex::from(bindless_index as u32);
            match *bindless_resource_type {
                BindlessResourceType::None => {}
                BindlessResourceType::Buffer => {
                    self.buffers
                        .get_mut(&bindless_index)
                        .expect("Buffer should exist with that bindless index")
                        .remove(bindless_binding);
                }
                BindlessResourceType::SamplerFiltering
                | BindlessResourceType::SamplerNonFiltering
                | BindlessResourceType::SamplerComparison => {
                    self.samplers
                        .get_mut(bindless_resource_type)
                        .expect("Sampler array should exist")
                        .remove(bindless_binding);
                }
                BindlessResourceType::Texture1d
                | BindlessResourceType::Texture2d
                | BindlessResourceType::Texture2dArray
                | BindlessResourceType::Texture3d
                | BindlessResourceType::TextureCube
                | BindlessResourceType::TextureCubeArray => {
                    self.textures
                        .get_mut(bindless_resource_type)
                        .expect("Texture array should exist")
                        .remove(bindless_binding);
                }
            }
        }

        self.bindings.free(slot);
        self.extra_data[slot.0 as usize] = None;

        // Invalidate the cached bind group.
        self.bind_group = None;

        self.live_allocation_count -= 1;
    }

    fn prepare(
        &mut self,
        render_device: &RenderDevice,
        bind_group_layout: &BindGroupLayout,
        fallback_bindless_resources: &FallbackBindlessResources,
        fallback_image: &FallbackImage,
        bindless_descriptor: &BindlessDescriptor,
    ) {
        self.bindings.prepare_buffer(render_device);

        if self.bind_group.is_some() {
            return;
        }

        let binding_resource_arrays = self.create_binding_resource_arrays(
            fallback_bindless_resources,
            fallback_image,
            bindless_descriptor,
        );

        let mut bind_group_entries = vec![BindGroupEntry {
            binding: 0,
            resource: self
                .bindings
                .buffer
                .buffer()
                .expect("Bindings buffer must exist")
                .as_entire_binding(),
        }];

        for &(&binding, ref binding_resource_array) in binding_resource_arrays.iter() {
            bind_group_entries.push(BindGroupEntry {
                binding,
                resource: match *binding_resource_array {
                    BindingResourceArray::Buffers(ref buffer_bindings) => {
                        //println!("setting buffer array binding {}", binding);
                        BindingResource::BufferArray(&buffer_bindings[..])
                    }
                    BindingResourceArray::TextureViews(ref texture_views) => {
                        //println!("setting texture view array binding {}", binding);
                        BindingResource::TextureViewArray(&texture_views[..])
                    }
                    BindingResourceArray::Samplers(ref samplers) => {
                        //println!("setting sampler array binding {}", binding);
                        BindingResource::SamplerArray(&samplers[..])
                    }
                },
            });
        }

        self.bind_group = Some(render_device.create_bind_group(
            M::label(),
            bind_group_layout,
            &bind_group_entries,
        ));
    }

    fn write_buffer(&mut self, render_device: &RenderDevice, render_queue: &RenderQueue) {
        self.bindings.write_buffer(render_device, render_queue);
    }

    fn create_binding_resource_arrays<'a>(
        &'a self,
        fallback_bindless_resources: &'a FallbackBindlessResources,
        fallback_image: &'a FallbackImage,
        bindless_descriptor: &'a BindlessDescriptor,
    ) -> Vec<(&'a u32, BindingResourceArray<'a>)> {
        let mut binding_resource_arrays = vec![];

        // Build sampler bindings.
        for (bindless_resource_type, fallback_sampler) in [
            (
                BindlessResourceType::SamplerFiltering,
                &fallback_bindless_resources.filtering_sampler,
            ),
            (
                BindlessResourceType::SamplerNonFiltering,
                &fallback_bindless_resources.non_filtering_sampler,
            ),
            (
                BindlessResourceType::SamplerComparison,
                &fallback_bindless_resources.comparison_sampler,
            ),
        ] {
            match self.samplers.get(&bindless_resource_type) {
                Some(sampler_bindless_binding_array) => {
                    let sampler_bindings = sampler_bindless_binding_array
                        .bindings
                        .iter()
                        .map(|maybe_bindless_binding| match *maybe_bindless_binding {
                            Some(ref bindless_binding) => &bindless_binding.resource,
                            None => &**fallback_sampler,
                        })
                        .collect();
                    binding_resource_arrays.push((
                        &*sampler_bindless_binding_array.binding_number,
                        BindingResourceArray::Samplers(sampler_bindings),
                    ));
                }

                // Fill with a single fallback sampler.
                None => {
                    let binding_number = bindless_resource_type
                        .binding_number()
                        .expect("Sampler bindless resource type must have a binding number");

                    binding_resource_arrays.push((
                        &**binding_number,
                        BindingResourceArray::Samplers(vec![&**fallback_sampler]),
                    ));
                }
            }
        }

        // Build texture bindings.
        for (bindless_resource_type, fallback_image) in [
            (BindlessResourceType::Texture1d, &fallback_image.d1),
            (BindlessResourceType::Texture2d, &fallback_image.d2),
            (
                BindlessResourceType::Texture2dArray,
                &fallback_image.d2_array,
            ),
            (BindlessResourceType::Texture3d, &fallback_image.d3),
            (BindlessResourceType::TextureCube, &fallback_image.cube),
            (
                BindlessResourceType::TextureCubeArray,
                &fallback_image.cube_array,
            ),
        ] {
            match self.textures.get(&bindless_resource_type) {
                Some(texture_bindless_binding_array) => {
                    let texture_bindings = texture_bindless_binding_array
                        .bindings
                        .iter()
                        .map(|maybe_bindless_binding| match *maybe_bindless_binding {
                            Some(ref bindless_binding) => &*bindless_binding.resource,
                            None => &*fallback_image.texture_view,
                        })
                        .collect();
                    binding_resource_arrays.push((
                        &*texture_bindless_binding_array.binding_number,
                        BindingResourceArray::TextureViews(texture_bindings),
                    ));
                }

                // Fill with a single fallback image.
                None => {
                    let binding_number = bindless_resource_type
                        .binding_number()
                        .expect("Texture bindless resource type must have a binding number");

                    binding_resource_arrays.push((
                        binding_number,
                        BindingResourceArray::TextureViews(vec![&*fallback_image.texture_view]),
                    ));
                }
            }
        }

        // Build buffer bindings.
        // FIXME: O(n^2), may want to mandate that `BindlessDescriptor::buffers` is sorted
        for bindless_buffer_descriptor in bindless_descriptor.buffers.iter() {
            let Some(buffer_bindless_binding_array) =
                self.buffers.get(&bindless_buffer_descriptor.bindless_index)
            else {
                error!(
                    "Slab didn't contain a binding array for buffer binding {:?}, bindless {:?}",
                    bindless_buffer_descriptor.binding_number,
                    bindless_buffer_descriptor.bindless_index,
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
                &*buffer_bindless_binding_array.binding_number,
                BindingResourceArray::Buffers(buffer_bindings),
            ));
        }

        binding_resource_arrays
    }

    pub fn bind_group(&self) -> Option<&BindGroup> {
        self.bind_group.as_ref()
    }

    pub fn get_extra_data(&self, slot: MaterialBindGroupSlot) -> &M::Data {
        self.extra_data
            .get(slot.0 as usize)
            .and_then(|data| data.as_ref())
            .expect("Extra data not present")
    }
}

impl<R> MaterialBindlessBindingArray<R>
where
    R: GetBindingResourceId,
{
    fn new(
        binding_number: BindingNumber,
        resource_type: BindlessResourceType,
        capacity: u32,
    ) -> MaterialBindlessBindingArray<R> {
        MaterialBindlessBindingArray {
            binding_number,
            bindings: vec![],
            resource_type,
            resource_to_slot: HashMap::default(),
            free_slots: vec![],
            len: 0,
            capacity,
        }
    }

    fn has_free_slots(&self, slot_count: u32) -> bool {
        self.len + slot_count < self.capacity
    }

    fn find(&self, binding_resource_id: BindingResourceId) -> Option<u32> {
        self.resource_to_slot.get(&binding_resource_id).copied()
    }

    fn insert(&mut self, binding_resource_id: BindingResourceId, resource: R) -> u32 {
        debug_assert_ne!(self.len, self.capacity);
        let slot = self.free_slots.pop().unwrap_or(self.len);
        self.resource_to_slot.insert(binding_resource_id, slot);

        if self.bindings.len() < slot as usize + 1 {
            self.bindings.resize_with(slot as usize + 1, || None);
        }
        self.bindings[slot as usize] = Some(MaterialBindlessBinding::new(resource));

        self.len += 1;
        slot
    }

    fn remove(&mut self, slot: u32) {
        let maybe_binding = &mut self.bindings[slot as usize];
        let binding = maybe_binding
            .as_mut()
            .expect("Attempted to free an already-freed binding");

        binding.ref_count -= 1;
        if binding.ref_count != 0 {
            return;
        }

        let binding_resource_id = binding.resource.binding_resource_id(self.resource_type);
        self.resource_to_slot.remove(&binding_resource_id);

        *maybe_binding = None;
        self.free_slots.push(slot);
        self.len -= 1;
    }
}

impl<R> MaterialBindlessBinding<R>
where
    R: GetBindingResourceId,
{
    fn new(resource: R) -> MaterialBindlessBinding<R> {
        MaterialBindlessBinding {
            resource,
            ref_count: 1,
        }
    }
}

/// Returns true if the material will *actually* use bindless resources or false
/// if it won't.
///
/// This takes the platform support (or lack thereof) for bindless resources
/// into account.
pub fn material_uses_bindless_resources<M>(render_device: &RenderDevice) -> bool
where
    M: Material,
{
    M::bindless_slot_count().is_some() && M::bindless_supported(render_device)
}

impl<M> MaterialBindlessSlab<M>
where
    M: Material,
{
    fn new(
        slab_capacity: u32,
        bindless_descriptor: &BindlessDescriptor,
    ) -> MaterialBindlessSlab<M> {
        let mut buffers = HashMap::default();
        let mut samplers = HashMap::default();
        let mut textures = HashMap::default();

        for (bindless_index, bindless_resource_type) in
            bindless_descriptor.resources.iter().enumerate()
        {
            let bindless_index = BindlessIndex(bindless_index as u32);
            match *bindless_resource_type {
                BindlessResourceType::None => {}
                BindlessResourceType::Buffer => {
                    let binding_number = bindless_descriptor
                        .buffers
                        .iter()
                        .find(|bindless_buffer_descriptor| {
                            bindless_buffer_descriptor.bindless_index == bindless_index
                        })
                        .expect(
                            "Bindless buffer descriptor matching that bindless index should be \
                             present",
                        )
                        .binding_number;
                    buffers.insert(
                        bindless_index,
                        MaterialBindlessBindingArray::new(
                            binding_number,
                            *bindless_resource_type,
                            slab_capacity,
                        ),
                    );
                }
                BindlessResourceType::SamplerFiltering
                | BindlessResourceType::SamplerNonFiltering
                | BindlessResourceType::SamplerComparison => {
                    samplers.insert(
                        *bindless_resource_type,
                        MaterialBindlessBindingArray::new(
                            *bindless_resource_type.binding_number().unwrap(),
                            *bindless_resource_type,
                            slab_capacity,
                        ),
                    );
                }
                BindlessResourceType::Texture1d
                | BindlessResourceType::Texture2d
                | BindlessResourceType::Texture2dArray
                | BindlessResourceType::Texture3d
                | BindlessResourceType::TextureCube
                | BindlessResourceType::TextureCubeArray => {
                    textures.insert(
                        *bindless_resource_type,
                        MaterialBindlessBindingArray::new(
                            *bindless_resource_type.binding_number().unwrap(),
                            *bindless_resource_type,
                            slab_capacity,
                        ),
                    );
                }
            }
        }

        MaterialBindlessSlab {
            bind_group: None,
            bindings: MaterialBindlessSlabBindings::new(bindless_descriptor),
            samplers,
            textures,
            buffers,
            extra_data: vec![],
            live_allocation_count: 0,
        }
    }
}

impl FromWorld for FallbackBindlessResources {
    fn from_world(world: &mut World) -> Self {
        let render_device = world.resource::<RenderDevice>();
        FallbackBindlessResources {
            filtering_sampler: render_device.create_sampler(&SamplerDescriptor {
                label: Some("fallback filtering sampler"),
                ..default()
            }),
            non_filtering_sampler: render_device.create_sampler(&SamplerDescriptor {
                label: Some("fallback non-filtering sampler"),
                mag_filter: FilterMode::Nearest,
                min_filter: FilterMode::Nearest,
                mipmap_filter: FilterMode::Nearest,
                ..default()
            }),
            comparison_sampler: render_device.create_sampler(&SamplerDescriptor {
                label: Some("fallback comparison sampler"),
                compare: Some(CompareFunction::Always),
                ..default()
            }),
        }
    }
}
