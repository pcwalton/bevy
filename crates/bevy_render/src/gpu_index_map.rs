//! Hash tables that are sparsely synchronized between CPU and GPU.

use alloc::borrow::Cow;
use bevy_platform::collections::{HashMap, HashSet};
use core::{
    hash::{BuildHasher, Hash},
    marker::PhantomData,
    mem,
};

use crate::{
    render_resource::BufferUsages,
    slab_allocator::{AllocationStage, DeallocationStage},
};
use crate::{
    renderer::{RenderDevice, RenderQueue},
    slab_allocator::{SlabAllocator, SlabId, SlabItem, SlabItemLayout},
};

const PAGE_SIZE: u32 = 4096;

pub(crate) struct SlabBackedIndexMaps<MK, K, V, S>
where
    MK: Clone + Hash + Eq,
    K: Clone + Hash + Eq,
    S: BuildHasher,
{
    allocator: SlabAllocator<SlabBackedIndexMapValue<MK, K, V>>,
    tables: HashMap<MK, SlabBackedIndexMap<K, V, S>>,
    dirty_tables: HashSet<MK>,
    allocation_settings: SlabAllocatorSettings,
}

pub(crate) struct SlabBackedIndexMap<MK, K, V, S>
where
    MK: Clone + Hash + Eq,
    K: Clone + Hash + Eq,
    S: BuildHasher,
{
    indices: HashMap<K, u32, S>,
    keys: Vec<K>,
    values: Vec<V>,
    slab: SlabId<SlabBackedIndexMapValue<MK, K, V>>,
    capacity: u32,
    dirty_pages: Vec<u64>,
}

pub(crate) enum Entry<'a, MK, K, V, S>
where
    MK: Clone + Hash + Eq,
    K: Clone + Hash + Eq,
    S: BuildHasher,
{
    Occupied(OccupiedEntry<'a, MK, K, V, S>),
    Vacant(VacantEntry<'a, MK, K, V, S>),
}

pub(crate) struct OccupiedEntry<'a, MK, K, V, S>
where
    MK: Clone + Hash + Eq,
    K: Clone + Hash + Eq,
    S: BuildHasher,
{
    indices: &'a mut HashMap<K, u32, S>,
    key: (&'a MK, &'a K),
    slab: SlabId<SlabBackedIndexMapValue<MK, V>>,
    dirty_pages: &'a mut Vec<u64>,
    allocator: &'a mut SlabAllocator<SlabBackedIndexMapValue<MK, V>>,
}

pub(crate) struct VacantEntry<'a, MK, K, V, S>
where
    MK: Clone + Hash + Eq,
    K: Clone + Hash + Eq,
    S: BuildHasher,
{
    key: (&'a MK, K),
    indices: &'a mut HashMap<K, u32, S>,
    dirty_pages: &'a mut Vec<u64>,
    allocator: &'a mut SlabAllocator<SlabBackedIndexMapValue<MK, V>>,
}

pub(crate) struct SlabBackedIndexMapValue<MK, V> {
    phantom: PhantomData<(MK, V)>,
}

pub(crate) struct SlabBackedIndexMapValueLayout<MK, K, V> {
    phantom: PhantomData<(MK, K, V)>,
}

impl<MK, K, V, S> SlabBackedIndexMaps<MK, K, V, S>
where
    MK: Clone + Hash + Eq,
    K: Clone + Hash + Eq,
    S: BuildHasher,
{
    pub(crate) fn get_mut(&mut self, map_key: &MK) -> Option<&mut SlabBackedIndexMap<MK, K, V, S>> {
        let mut index_map = self.tables.get_mut(map_key)?;
        self.dirty_tables.insert(map_key.clone());
        Some(index_map)
    }

    /// Syncs changes to the GPU.
    pub(crate) fn write(&mut self, render_device: &RenderDevice, render_queue: &RenderQueue) {
        let mut deallocation_stage = self.allocator.stage_deallocation();
        for dirty_table_key in &self.dirty_tables {
            self.tables[dirty_table_key]
                .free_allocation_if_necessary(&mut deallocation_stage, dirty_table_key);
        }
        deallocation_stage.commit();

        let mut allocation_stage = self.allocator.stage_allocation();
        for dirty_table_key in &self.dirty_tables {
            self.tables[dirty_table_key].allocate_if_necessary(
                &mut allocation_stage,
                &self.allocation_settings,
                dirty_table_key,
            );
        }
        allocation_stage.commit(render_device, render_queue);

        for dirty_table_key in self.dirty_tables.drain() {
            self.tables[dirty_table_key].write_if_necessary(
                &mut allocation_stage,
                render_device,
                render_queue,
                &dirty_table_key,
            );
        }
    }
}

impl<MK, K, V, S> SlabBackedIndexMap<MK, K, V, S>
where
    MK: Clone + Hash + Eq,
    K: Clone + Hash + Eq,
    S: BuildHasher,
{
    pub(crate) fn entry<'a>(
        &'a mut self,
        allocator: &'a mut SlabAllocator<SlabBackedIndexMapValue<MK, V>>,
        key: K,
    ) -> Entry<'_, MK, K, V, S> {
        let hash = self.hash(&key);
        // TODO
        Entry::new()
    }

    pub(crate) fn swap_remove(
        &mut self,
        allocator: &mut SlabAllocator<SlabBackedIndexMapValue<MK, V>>,
        map_key: &MK,
        key: &K,
    ) -> Option<V> {
        let index = self.indices.remove(key)?;
        let moved_index = (self.keys.len() - 1) as u32;
        let moved_key = self.keys.swap_remove(index as usize);
        let old_value = mem::replace(&mut self.values[index as usize], self.values[moved_index]);
        self.indices[&moved_key] = index;
        self.note_changed_index(index);
        self.note_changed_index(moved_index);
        Some(old_value)
    }

    pub fn free_allocation_if_necessary(
        &mut self,
        deallocation_stage: &mut DeallocationStage<SlabBackedIndexMapValue<MK, V>>,
        map_key: &MK,
    ) {
        // Only free the allocation if it's grown.
        if self.values.len() > self.capacity as usize {
            deallocation_stage.free(map_key);
        }
    }

    pub fn allocate_if_necessary(
        &mut self,
        allocation_stage: &mut AllocationStage<SlabBackedIndexMapValue<MK, V>>,
        settings: &SlabAllocatorSettings,
        map_key: &MK,
    ) {
        if self.values.len() <= self.capacity as usize {
            return;
        }

        // FIXME: round up to 1.5
        let new_capacity = (self.capacity + 1).next_power_of_two();
        allocation_stage.allocate(
            map_key,
            self.values.len() * size_of::<V>(),
            self.layout,
            SlabBackedIndexMapValueLayout::<MK, K, V>::new(),
            settings,
        );
    }

    pub fn write_if_necessary(
        &mut self,
        allocator: &mut SlabAllocator<SlabBackedIndexMapValue<MK, V>>,
        render_device: &RenderDevice,
        render_queue: &RenderQueue,
        map_key: &MK,
    ) {
        for (page_word_index, page_word) in self.dirty_pages.iter_mut().enumerate() {
            let page_word = mem::take(page_word);
            while page_word != 0 {
                let page_in_word = page_word.trailing_zeros();
                page_word &= !(1u64 << page_in_word);
                let page = page_word_index * 64 + page_in_word;
                // TODO: copy sparsely. We shouldn't use this API. We should
                // make a new one that lets the sparse buffer vec do its thing.
                allocator.copy_element_data(
                    map_key,
                    self.values.len(),
                    |dest| dest.copy_from_slice(bytemuck::cast_slice(&self.values[..])),
                    render_device,
                    render_queue,
                );
            }
        }
    }

    fn note_changed_index(&mut self, index: u32) {
        note_changed_index(&mut self.dirty_pages, index)
    }
}

impl<MK, K, V> SlabItem for SlabBackedIndexMapValue<MK, K, V>
where
    MK: Clone + Hash + Eq,
    K: Clone + Hash + Eq,
{
    type Key = MK;
    type Layout = SlabBackedIndexMapValueLayout<V>;
    fn label() -> Cow<'static, str> {
        "slab-backed index map".into()
    }
}

impl<MK, K, V> SlabItemLayout for SlabBackedIndexMapValueLayout<MK, K, V>
where
    MK: Clone + Hash + Eq,
    K: Clone + Hash + Eq,
{
    fn size(&self) -> u64 {
        size_of::<V>() as u64
    }

    fn elements_per_slot(&self) -> u32 {
        1
    }

    fn buffer_usages(&self) -> BufferUsages {
        BufferUsages::STORAGE
    }
}

impl<'a, MK, K, V, S> OccupiedEntry<'a, MK, K, V, S>
where
    MK: Clone + Hash + Eq,
    K: Clone + Hash + Eq,
    S: BuildHasher,
{
    pub(crate) fn insert(&mut self, value: V) -> V {
        let index = self
            .indices
            .get(self.key.1)
            .expect("Index should be present in map");
        note_changed_index(self.dirty_pages, *index);
        mem::replace(&mut self.map.values[*index as usize], value)
    }
}

impl<'a, MK, K, V, S> VacantEntry<'a, MK, K, V, S>
where
    MK: Clone + Hash + Eq,
    K: Clone + Hash + Eq,
    S: BuildHasher,
{
    pub(crate) fn insert(&mut self, value: V) {
        let index = self.keys.len() as u32;
        debug_assert_eq!(index, self.values.len() as u32);
        self.keys.push(self.key.1);
        self.indices.insert(self.key.1, index);
        self.values.push(value);
        note_changed_index(self.dirty_pages, index);
    }
}

fn note_changed_index(dirty_pages: &mut Vec<u64>, index: u32) {
    let page = index / PAGE_SIZE;
    let (page_word_index, index_in_page_word) = (page / 64, page % 64);
    while page >= dirty_pages.len() {
        dirty_pages.push(0);
    }
    dirty_pages[page_word_index as usize] |= 1u64 << index_in_page_word;
}
