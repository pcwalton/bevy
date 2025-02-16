//! Management of material bind groups and their resources.

use core::{marker::PhantomData, mem};

use bevy_derive::{Deref, DerefMut};
use bevy_ecs::{
    resource::Resource,
    world::{FromWorld, World},
};
use bevy_platform_support::collections::{HashMap, HashSet};
use bevy_reflect::{prelude::ReflectDefault, Reflect};
use bevy_render::{
    render_resource::{
        BindGroup, BindGroupEntry, BindGroupLayout, BindingNumber, BindingResource,
        BindlessDescriptor, BindlessIndex, BindlessResourceType, Buffer, BufferBinding,
        BufferDescriptor, BufferId, BufferUsages, CompareFunction, FilterMode,
        OwnedBindingResource, PreparedBindGroup, RawBufferVec, Sampler, SamplerDescriptor,
        SamplerId, TextureView, TextureViewDimension, TextureViewId, UnpreparedBindGroup,
        WgpuSampler, WgpuTextureView,
    },
    renderer::{RenderDevice, RenderQueue},
    texture::FallbackImage,
};
use bevy_utils::default;
use tracing::{error, trace};

use crate::Material;

/// A resource that places materials into bind groups and tracks their
/// resources.
#[derive(Resource)]
pub enum MaterialBindGroupAllocator<M>
where
    M: Material,
{
    Bindless(Box<MaterialBindGroupBindlessAllocator<M>>),
    NonBindless(Box<MaterialBindGroupNonBindlessAllocator<M>>),
}

pub struct MaterialBindGroupBindlessAllocator<M>
where
    M: Material,
{
    slabs: Vec<MaterialBindlessSlab<M>>,
    bind_group_layout: BindGroupLayout,
    bindless_descriptor: BindlessDescriptor,
    fallback_buffers: HashMap<BindlessIndex, Buffer>,
    slab_capacity: u32,
}

/// A single bind group and the bookkeeping necessary to allocate into it.
pub struct MaterialBindlessSlab<M>
where
    M: Material,
{
    /// The current bind group, if it's up to date.
    ///
    /// If this is `None`, then the bind group is dirty and needs to be
    /// regenerated.
    bind_group: Option<BindGroup>,

    /// A GPU-accessible buffer that holds the mapping from binding index to
    /// bindless slot.
    ///
    /// This is conventionally assigned to bind group binding 0.
    bindless_index_table: MaterialBindlessIndexTable<M>,

    /// The binding arrays containing samplers.
    samplers: HashMap<BindlessResourceType, MaterialBindlessBindingArray<Sampler>>,
    /// The binding arrays containing textures.
    textures: HashMap<BindlessResourceType, MaterialBindlessBindingArray<TextureView>>,
    /// The binding arrays containing data buffers.
    buffers: HashMap<BindlessIndex, MaterialBindlessBindingArray<Buffer>>,

    /// Holds extra CPU-accessible data that the material provides.
    ///
    /// Typically, this data is used for constructing the material key, for
    /// pipeline specialization purposes.
    extra_data: Vec<Option<M::Data>>,

    /// A free list of slot IDs.
    free_slots: Vec<MaterialBindGroupSlot>,
    /// The total number of materials currently allocated in this slab.
    live_allocation_count: u32,
    /// The total number of resources currently allocated in the binding arrays.
    allocated_resource_count: u32,
}

/// A GPU-accessible buffer that holds the mapping from binding index to
/// bindless slot.
///
/// This is conventionally assigned to bind group binding 0.
struct MaterialBindlessIndexTable<M>
where
    M: Material,
{
    buffer: RawBufferVec<u32>,
    buffer_dirty: BufferDirtyState,
    phantom: PhantomData<M>,
}

/// A single binding array for storing bindless resources and the bookkeeping
/// necessary to allocate into it.
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
}

struct MaterialBindlessBinding<R>
where
    R: GetBindingResourceId,
{
    resource: R,
    ref_count: u32,
}

pub struct MaterialBindGroupNonBindlessAllocator<M>
where
    M: Material,
{
    bind_groups: Vec<Option<MaterialNonBindlessAllocatedBindGroup<M>>>,
    to_prepare: HashSet<MaterialBindGroupIndex>,
    free_list: Vec<MaterialBindGroupIndex>,
    phantom: PhantomData<M>,
}

enum MaterialNonBindlessAllocatedBindGroup<M>
where
    M: Material,
{
    Unprepared {
        bind_group: UnpreparedBindGroup<M::Data>,
        layout: BindGroupLayout,
    },
    Prepared(PreparedBindGroup<M::Data>),
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
/// to satisfy `wgpu`'s lifetime requirements.
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
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Reflect, Deref, DerefMut)]
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
#[derive(Clone, Copy, Debug, Default, PartialEq, Reflect, Deref, DerefMut)]
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

/// The public interface to a slab, which represents a single bind group.
pub struct MaterialSlab<'a, M>(MaterialSlabImpl<'a, M>)
where
    M: Material;

/// The actual implementation of a material slab.
///
/// This has bindless and non-bindless variants.
enum MaterialSlabImpl<'a, M>
where
    M: Material,
{
    /// The implementation of the slab interface we use when the slab
    /// is bindless.
    Bindless(&'a MaterialBindlessSlab<M>),
    /// The implementation of the slab interface we use when the slab
    /// is non-bindless.
    NonBindless(MaterialNonBindlessSlab<'a, M>),
}

struct MaterialNonBindlessSlab<'a, M>(&'a PreparedBindGroup<M::Data>)
where
    M: Material;

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

impl<M> MaterialBindGroupAllocator<M>
where
    M: Material,
{
    fn new(render_device: &RenderDevice) -> MaterialBindGroupAllocator<M> {
        if M::bindless_slot_count()
            .is_some_and(|bindless_slot_count| bindless_slot_count.resolve() > 1)
        {
            MaterialBindGroupAllocator::Bindless(Box::new(MaterialBindGroupBindlessAllocator::new(
                render_device,
            )))
        } else {
            MaterialBindGroupAllocator::NonBindless(Box::new(
                MaterialBindGroupNonBindlessAllocator::new(),
            ))
        }
    }

    /// Returns the slab with the given index, if one exists.
    pub fn get(&self, group: MaterialBindGroupIndex) -> Option<MaterialSlab<M>> {
        match *self {
            MaterialBindGroupAllocator::Bindless(ref bindless_allocator) => bindless_allocator
                .get(group)
                .map(|bindless_slab| MaterialSlab(MaterialSlabImpl::Bindless(bindless_slab))),
            MaterialBindGroupAllocator::NonBindless(ref non_bindless_allocator) => {
                non_bindless_allocator.get(group).map(|non_bindless_slab| {
                    MaterialSlab(MaterialSlabImpl::NonBindless(non_bindless_slab))
                })
            }
        }
    }

    /// Allocates an [`UnpreparedBindGroup`] and returns the resulting binding ID.
    ///
    /// This method should generally be preferred over
    /// [`Self::allocate_prepared`], because this method supports both bindless
    /// and non-bindless bind groups. Only use [`Self::allocate_preferred`] if
    /// you need to prepare the bind group yourself.
    pub fn allocate_unprepared(
        &mut self,
        unprepared_bind_group: UnpreparedBindGroup<M::Data>,
        bind_group_layout: &BindGroupLayout,
    ) -> MaterialBindingId {
        match *self {
            MaterialBindGroupAllocator::Bindless(
                ref mut material_bind_group_bindless_allocator,
            ) => material_bind_group_bindless_allocator.allocate_unprepared(unprepared_bind_group),
            MaterialBindGroupAllocator::NonBindless(
                ref mut material_bind_group_non_bindless_allocator,
            ) => material_bind_group_non_bindless_allocator
                .allocate_unprepared(unprepared_bind_group, (*bind_group_layout).clone()),
        }
    }

    pub fn allocate_prepared(
        &mut self,
        prepared_bind_group: PreparedBindGroup<M::Data>,
    ) -> MaterialBindingId {
        match *self {
            MaterialBindGroupAllocator::Bindless(_) => {
                panic!(
                    "Bindless resources are incompatible with implementing `as_bind_group` \
                     directly; implement `unprepared_bind_group` instead or disable bindless"
                )
            }
            MaterialBindGroupAllocator::NonBindless(ref mut non_bindless_allocator) => {
                non_bindless_allocator.allocate_prepared(prepared_bind_group)
            }
        }
    }

    /// Deallocates the material with the given binding ID.
    ///
    /// Any resources that are no longer referenced are removed from the slab.
    pub fn free(&mut self, material_binding_id: MaterialBindingId) {
        match *self {
            MaterialBindGroupAllocator::Bindless(
                ref mut material_bind_group_bindless_allocator,
            ) => material_bind_group_bindless_allocator.free(material_binding_id),
            MaterialBindGroupAllocator::NonBindless(
                ref mut material_bind_group_non_bindless_allocator,
            ) => material_bind_group_non_bindless_allocator.free(material_binding_id),
        }
    }

    pub fn prepare_bind_groups(
        &mut self,
        render_device: &RenderDevice,
        fallback_bindless_resources: &FallbackBindlessResources,
        fallback_image: &FallbackImage,
    ) {
        match *self {
            MaterialBindGroupAllocator::Bindless(
                ref mut material_bind_group_bindless_allocator,
            ) => material_bind_group_bindless_allocator.prepare_bind_groups(
                render_device,
                fallback_bindless_resources,
                fallback_image,
            ),
            MaterialBindGroupAllocator::NonBindless(
                ref mut material_bind_group_non_bindless_allocator,
            ) => material_bind_group_non_bindless_allocator.prepare_bind_groups(render_device),
        }
    }

    pub fn write_buffers(&mut self, render_device: &RenderDevice, render_queue: &RenderQueue) {
        match *self {
            MaterialBindGroupAllocator::Bindless(
                ref mut material_bind_group_bindless_allocator,
            ) => material_bind_group_bindless_allocator.write_buffers(render_device, render_queue),
            MaterialBindGroupAllocator::NonBindless(_) => {
                // Not applicable.
            }
        }
    }
}

impl<M> MaterialBindlessIndexTable<M>
where
    M: Material,
{
    fn new(bindless_descriptor: &BindlessDescriptor) -> MaterialBindlessIndexTable<M> {
        // Preallocate space for one bindings table, so that there will always be a buffer.
        let mut buffer = RawBufferVec::new(BufferUsages::STORAGE);
        for _ in 0..bindless_descriptor.resources.len() {
            buffer.push(0);
        }

        MaterialBindlessIndexTable {
            buffer,
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
}

impl<M> MaterialBindGroupBindlessAllocator<M>
where
    M: Material,
{
    fn new(render_device: &RenderDevice) -> MaterialBindGroupBindlessAllocator<M> {
        let bindless_descriptor = M::bindless_descriptor()
            .expect("Non-bindless materials should use the non-bindless allocator");
        let fallback_buffers = bindless_descriptor
            .buffers
            .iter()
            .map(|bindless_buffer_descriptor| {
                (
                    bindless_buffer_descriptor.bindless_index,
                    render_device.create_buffer(&BufferDescriptor {
                        label: Some("bindless fallback buffer"),
                        size: bindless_buffer_descriptor.element_size as u64,
                        usage: BufferUsages::STORAGE,
                        mapped_at_creation: false,
                    }),
                )
            })
            .collect();

        MaterialBindGroupBindlessAllocator {
            slabs: vec![],
            bind_group_layout: M::bind_group_layout(render_device),
            bindless_descriptor,
            fallback_buffers,
            slab_capacity: M::bindless_slot_count()
                .expect("Non-bindless materials should use the non-bindless allocator")
                .resolve(),
        }
    }

    fn allocate_unprepared(
        &mut self,
        mut unprepared_bind_group: UnpreparedBindGroup<M::Data>,
    ) -> MaterialBindingId {
        for (slab_index, slab) in self.slabs.iter_mut().enumerate() {
            trace!("Trying to allocate in slab {}", slab_index);
            match slab.try_allocate(
                unprepared_bind_group,
                &self.bindless_descriptor,
                self.slab_capacity,
            ) {
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
        self.slabs
            .push(MaterialBindlessSlab::new(&self.bindless_descriptor));

        // Allocate into the newly-pushed slab.
        let Ok(slot) = self
            .slabs
            .last_mut()
            .expect("We just pushed a slab")
            .try_allocate(
                unprepared_bind_group,
                &self.bindless_descriptor,
                self.slab_capacity,
            )
        else {
            panic!("An allocation into an empty slab should always succeed")
        };

        MaterialBindingId { group, slot }
    }

    fn free(&mut self, material_binding_id: MaterialBindingId) {
        self.slabs
            .get_mut(material_binding_id.group.0 as usize)
            .expect("Slab should exist")
            .free(material_binding_id.slot, &self.bindless_descriptor);
    }

    fn get(&self, group: MaterialBindGroupIndex) -> Option<&MaterialBindlessSlab<M>> {
        self.slabs.get(group.0 as usize)
    }

    fn prepare_bind_groups(
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
                &self.fallback_buffers,
                fallback_image,
                &self.bindless_descriptor,
            );
        }
    }

    /// Writes any buffers that we're managing to the GPU.
    ///
    /// Currently, this only consists of the bindless index tables.
    fn write_buffers(&mut self, render_device: &RenderDevice, render_queue: &RenderQueue) {
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
    /// Attempts to allocate the given unprepared bind group in this slab.
    ///
    /// If the allocation succeeds, this method returns the slot that the
    /// allocation was placed in. If the allocation fails because the slab was
    /// full, this method returns the unprepared bind group back to the caller
    /// so that it can try to allocate again.
    fn try_allocate(
        &mut self,
        mut unprepared_bind_group: UnpreparedBindGroup<M::Data>,
        bindless_descriptor: &BindlessDescriptor,
        slot_capacity: u32,
    ) -> Result<MaterialBindGroupSlot, UnpreparedBindGroup<M::Data>> {
        let mut pre_existing_resources = HashMap::new();
        let mut needed_free_buffer_slots = vec![];
        let mut needed_free_fixed_resource_slots: HashMap<BindlessResourceType, u32> =
            HashMap::default();
        let mut total_needed_slots = 0;

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
                            total_needed_slots += 1;
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
                            total_needed_slots += 1;
                        }
                    }
                }
            }
        }

        // Check to see if we have enough free space.
        //
        // As a special case, note that if *nothing* is allocated in this slab,
        // then we always allow a material to be placed in it, regardless of the
        // number of bindings the material has. This is so that, if the
        // platform's maximum bindless count is set too low to hold even a
        // single material, we can still place each material into a separate
        // slab instead of failing outright.
        if self.allocated_resource_count > 0
            && self.allocated_resource_count + total_needed_slots > slot_capacity
        {
            trace!("Slab is full, can't allocate");
            return Err(unprepared_bind_group);
        }

        // OK, we can allocate in this slab. Assign a slot ID.
        let slot = self
            .free_slots
            .pop()
            .unwrap_or(MaterialBindGroupSlot(self.live_allocation_count));

        // Bump the live allocation count.
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

            // Bump the allocated resource count.
            self.allocated_resource_count += 1;
        }

        // Serialize the allocated resource slots.
        self.bindless_index_table
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

    /// Removes the material allocated in the given slot, with the given
    /// descriptor, from this slab.
    fn free(&mut self, slot: MaterialBindGroupSlot, bindless_descriptor: &BindlessDescriptor) {
        // Loop through each binding.
        for (bindless_index, (bindless_resource_type, &bindless_binding)) in bindless_descriptor
            .resources
            .iter()
            .zip(self.bindless_index_table.get(slot, bindless_descriptor))
            .enumerate()
        {
            let bindless_index = BindlessIndex::from(bindless_index as u32);

            // Free the binding.
            let resource_freed = match *bindless_resource_type {
                BindlessResourceType::None => false,
                BindlessResourceType::Buffer => self
                    .buffers
                    .get_mut(&bindless_index)
                    .expect("Buffer should exist with that bindless index")
                    .remove(bindless_binding),
                BindlessResourceType::SamplerFiltering
                | BindlessResourceType::SamplerNonFiltering
                | BindlessResourceType::SamplerComparison => self
                    .samplers
                    .get_mut(bindless_resource_type)
                    .expect("Sampler array should exist")
                    .remove(bindless_binding),
                BindlessResourceType::Texture1d
                | BindlessResourceType::Texture2d
                | BindlessResourceType::Texture2dArray
                | BindlessResourceType::Texture3d
                | BindlessResourceType::TextureCube
                | BindlessResourceType::TextureCubeArray => self
                    .textures
                    .get_mut(bindless_resource_type)
                    .expect("Texture array should exist")
                    .remove(bindless_binding),
            };

            // If the slot is now free, decrement the allocated resource
            // count.
            if resource_freed {
                self.allocated_resource_count -= 1;
            }
        }

        // Clear out the extra data.
        self.extra_data[slot.0 as usize] = None;

        // Invalidate the cached bind group.
        self.bind_group = None;

        // Release the slot ID.
        self.free_slots.push(slot);
        self.live_allocation_count -= 1;
    }

    fn prepare(
        &mut self,
        render_device: &RenderDevice,
        bind_group_layout: &BindGroupLayout,
        fallback_bindless_resources: &FallbackBindlessResources,
        fallback_buffers: &HashMap<BindlessIndex, Buffer>,
        fallback_image: &FallbackImage,
        bindless_descriptor: &BindlessDescriptor,
    ) {
        self.bindless_index_table.prepare_buffer(render_device);

        if self.bind_group.is_some() {
            return;
        }

        let binding_resource_arrays = self.create_binding_resource_arrays(
            fallback_bindless_resources,
            fallback_buffers,
            fallback_image,
            bindless_descriptor,
        );

        let mut bind_group_entries = vec![BindGroupEntry {
            binding: 0,
            resource: self
                .bindless_index_table
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

    /// Writes any buffers that we're managing to the GPU.
    ///
    /// Currently, this only consists of the bindless index table.
    fn write_buffer(&mut self, render_device: &RenderDevice, render_queue: &RenderQueue) {
        self.bindless_index_table
            .write_buffer(render_device, render_queue);
    }

    fn create_binding_resource_arrays<'a>(
        &'a self,
        fallback_bindless_resources: &'a FallbackBindlessResources,
        fallback_buffers: &'a HashMap<BindlessIndex, Buffer>,
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
                        None => fallback_buffers
                            .get(&bindless_buffer_descriptor.bindless_index)
                            .expect("Fallback buffer should exist"),
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

    fn bind_group(&self) -> Option<&BindGroup> {
        self.bind_group.as_ref()
    }

    fn get_extra_data(&self, slot: MaterialBindGroupSlot) -> &M::Data {
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
    ) -> MaterialBindlessBindingArray<R> {
        MaterialBindlessBindingArray {
            binding_number,
            bindings: vec![],
            resource_type,
            resource_to_slot: HashMap::default(),
            free_slots: vec![],
            len: 0,
        }
    }

    fn find(&self, binding_resource_id: BindingResourceId) -> Option<u32> {
        self.resource_to_slot.get(&binding_resource_id).copied()
    }

    fn insert(&mut self, binding_resource_id: BindingResourceId, resource: R) -> u32 {
        let slot = self.free_slots.pop().unwrap_or(self.len);
        self.resource_to_slot.insert(binding_resource_id, slot);

        if self.bindings.len() < slot as usize + 1 {
            self.bindings.resize_with(slot as usize + 1, || None);
        }
        self.bindings[slot as usize] = Some(MaterialBindlessBinding::new(resource));

        self.len += 1;
        slot
    }

    /// Removes a reference to an object from the slot.
    ///
    /// If the reference count dropped to 0 and the object was freed, this
    /// method returns true. If the object was still referenced after removing
    /// it, returns false.
    fn remove(&mut self, slot: u32) -> bool {
        let maybe_binding = &mut self.bindings[slot as usize];
        let binding = maybe_binding
            .as_mut()
            .expect("Attempted to free an already-freed binding");

        binding.ref_count -= 1;
        if binding.ref_count != 0 {
            return false;
        }

        let binding_resource_id = binding.resource.binding_resource_id(self.resource_type);
        self.resource_to_slot.remove(&binding_resource_id);

        *maybe_binding = None;
        self.free_slots.push(slot);
        self.len -= 1;
        true
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
    fn new(bindless_descriptor: &BindlessDescriptor) -> MaterialBindlessSlab<M> {
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
                        MaterialBindlessBindingArray::new(binding_number, *bindless_resource_type),
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
                        ),
                    );
                }
            }
        }

        MaterialBindlessSlab {
            bind_group: None,
            bindless_index_table: MaterialBindlessIndexTable::new(bindless_descriptor),
            samplers,
            textures,
            buffers,
            extra_data: vec![],
            free_slots: vec![],
            live_allocation_count: 0,
            allocated_resource_count: 0,
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

impl<M> MaterialBindGroupNonBindlessAllocator<M>
where
    M: Material,
{
    fn new() -> MaterialBindGroupNonBindlessAllocator<M> {
        MaterialBindGroupNonBindlessAllocator {
            bind_groups: vec![],
            to_prepare: HashSet::default(),
            free_list: vec![],
            phantom: PhantomData,
        }
    }

    fn allocate(
        &mut self,
        bind_group: MaterialNonBindlessAllocatedBindGroup<M>,
    ) -> MaterialBindingId {
        let group_id = self
            .free_list
            .pop()
            .unwrap_or(MaterialBindGroupIndex(self.bind_groups.len() as u32));
        if self.bind_groups.len() < *group_id as usize + 1 {
            self.bind_groups
                .resize_with(*group_id as usize + 1, || None);
        }

        self.bind_groups[*group_id as usize] = Some(bind_group);

        MaterialBindingId {
            group: group_id,
            slot: default(),
        }
    }

    fn allocate_unprepared(
        &mut self,
        unprepared_bind_group: UnpreparedBindGroup<M::Data>,
        bind_group_layout: BindGroupLayout,
    ) -> MaterialBindingId {
        self.allocate(MaterialNonBindlessAllocatedBindGroup::Unprepared {
            bind_group: unprepared_bind_group,
            layout: bind_group_layout,
        })
    }

    fn allocate_prepared(
        &mut self,
        prepared_bind_group: PreparedBindGroup<M::Data>,
    ) -> MaterialBindingId {
        self.allocate(MaterialNonBindlessAllocatedBindGroup::Prepared(
            prepared_bind_group,
        ))
    }

    fn free(&mut self, binding_id: MaterialBindingId) {
        debug_assert_eq!(binding_id.slot, MaterialBindGroupSlot(0));
        debug_assert!(self.bind_groups[*binding_id.group as usize].is_none());
        self.bind_groups[*binding_id.group as usize] = None;
        self.to_prepare.remove(&binding_id.group);
        self.free_list.push(binding_id.group);
    }

    fn get(&self, group: MaterialBindGroupIndex) -> Option<MaterialNonBindlessSlab<M>> {
        match self.bind_groups[group.0 as usize] {
            Some(MaterialNonBindlessAllocatedBindGroup::Prepared(ref prepared_bind_group)) => {
                Some(MaterialNonBindlessSlab(prepared_bind_group))
            }
            Some(MaterialNonBindlessAllocatedBindGroup::Unprepared { .. }) | None => None,
        }
    }

    fn prepare_bind_groups(&mut self, render_device: &RenderDevice) {
        for bind_group_index in mem::take(&mut self.to_prepare) {
            let Some(MaterialNonBindlessAllocatedBindGroup::Unprepared {
                bind_group: unprepared_bind_group,
                layout: bind_group_layout,
            }) = mem::take(&mut self.bind_groups[*bind_group_index as usize])
            else {
                panic!("Allocation didn't exist or was already prepared");
            };

            let entries: Vec<_> = unprepared_bind_group
                .bindings
                .iter()
                .map(|(index, binding)| BindGroupEntry {
                    binding: *index,
                    resource: binding.get_binding(),
                })
                .collect();

            let bind_group =
                render_device.create_bind_group(M::label(), &bind_group_layout, &entries);

            self.bind_groups[*bind_group_index as usize] = Some(
                MaterialNonBindlessAllocatedBindGroup::Prepared(PreparedBindGroup {
                    bindings: unprepared_bind_group.bindings,
                    bind_group,
                    data: unprepared_bind_group.data,
                }),
            );
        }
    }
}

impl<'a, M> MaterialSlab<'a, M>
where
    M: Material,
{
    pub fn get_extra_data(&self, slot: MaterialBindGroupSlot) -> &M::Data {
        match self.0 {
            MaterialSlabImpl::Bindless(material_bindless_slab) => {
                material_bindless_slab.get_extra_data(slot)
            }
            MaterialSlabImpl::NonBindless(ref prepared_bind_group) => &prepared_bind_group.0.data,
        }
    }

    pub fn bind_group(&self) -> Option<&'a BindGroup> {
        match self.0 {
            MaterialSlabImpl::Bindless(material_bindless_slab) => {
                material_bindless_slab.bind_group()
            }
            MaterialSlabImpl::NonBindless(ref prepared_bind_group) => {
                Some(&prepared_bind_group.0.bind_group)
            }
        }
    }
}
