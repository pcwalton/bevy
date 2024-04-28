//! GPU allocators.

use std::fmt::{self, Debug, Display, Formatter, Write};
use std::mem;

use bevy_app::{App, Plugin};
use bevy_ecs::system::Resource;
use bevy_utils::HashMap;
use offset_allocator::Allocator;
use slotmap::{new_key_type, SlotMap};
use wgpu::util::BufferInitDescriptor;
use wgpu::{BufferDescriptor, BufferUsages};

use crate::mesh::MeshVertexBufferLayoutRef;
use crate::render_resource::{Buffer, BufferSlice};
use crate::renderer::{RenderDevice, RenderQueue};
use crate::RenderApp;

/// 256MB slabs.
pub const SLAB_SIZE: u64 = 256 * 1024 * 1024;

pub struct GpuAllocatorPlugin;

#[derive(Resource, Default)]
pub struct GpuAllocator {
    suballocators: SlotMap<GpuSuballocatorKey, GpuSuballocator>,
    class_to_suballocator_key: HashMap<GpuAllocationClass, GpuSuballocatorKey>,
}

pub struct GpuSuballocator {
    main_slabs: Vec<GpuSlab>,
    large_slabs: Vec<Buffer>,
    class: GpuAllocationClass,
}

new_key_type! {
    struct GpuSuballocatorKey;
}

#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub enum GpuAllocationClass {
    Vertex(MeshVertexBufferLayoutRef),
    Index,
}

pub struct GpuSlab {
    buffer: Buffer,
    allocator: Allocator,
}

#[derive(Clone, Debug)]
pub struct GpuAllocation {
    key: GpuSuballocatorKey,
    slab: u32,
    /// This is in elements, not bytes.
    ///
    /// If not present, this is a large allocation.
    offset: Option<u64>,
    /// Size in elements.
    size: u64,
}

impl Plugin for GpuAllocatorPlugin {
    fn build(&self, app: &mut App) {
        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };

        render_app.init_resource::<GpuAllocator>();
    }
}

impl GpuAllocator {
    pub fn get_buffer_slice<'a>(&'a self, allocation: &GpuAllocation) -> BufferSlice<'a> {
        self.suballocators[allocation.key].get_buffer_slice(allocation)
    }

    pub fn get_buffer_and_offset<'a>(&'a self, allocation: &GpuAllocation) -> (&'a Buffer, u64) {
        self.suballocators[allocation.key].get_buffer_and_offset(allocation)
    }

    pub fn allocate_with_data(
        &mut self,
        render_device: &RenderDevice,
        render_queue: &RenderQueue,
        class: GpuAllocationClass,
        contents: &[u32],
        label: Option<&str>,
    ) -> GpuAllocation {
        let suballocator_key = self
            .class_to_suballocator_key
            .entry(class.clone())
            .or_insert_with(|| {
                self.suballocators
                    .insert(GpuSuballocator::new(class.clone()))
            });

        let suballocator = &mut self.suballocators[*suballocator_key];

        if contents.len() as u64 * 4 * class.element_size() <= SLAB_SIZE {
            suballocator.allocate_with_data(
                render_device,
                render_queue,
                *suballocator_key,
                contents,
                label,
            )
        } else {
            suballocator.allocate_large_with_data(render_device, *suballocator_key, contents, label)
        }
    }
}

impl GpuSuballocator {
    fn new(class: GpuAllocationClass) -> Self {
        GpuSuballocator {
            main_slabs: vec![],
            large_slabs: vec![],
            class,
        }
    }

    fn allocate_with_data(
        &mut self,
        render_device: &RenderDevice,
        render_queue: &RenderQueue,
        key: GpuSuballocatorKey,
        contents: &[u32],
        label: Option<&str>,
    ) -> GpuAllocation {
        // First-fit.
        for (slab_index, slab) in self.main_slabs.iter_mut().enumerate() {
            if let Some(allocation) =
                slab.allocate_with_data(render_queue, slab_index as u32, &self.class, key, contents)
            {
                return allocation;
            }
        }

        println!("allocating new slab, class={:?}", self.class);
        let new_slab_index = self.main_slabs.len();
        let new_slab_label = self.create_slab_label(new_slab_index, label);
        let mut new_slab = GpuSlab::new(render_device, self.class.clone(), new_slab_label);
        let allocation = new_slab
            .allocate_with_data(
                render_queue,
                new_slab_index as u32,
                &self.class,
                key,
                contents,
            )
            .expect("This allocation should never fail");

        self.main_slabs.push(new_slab);
        allocation
    }

    fn allocate_large_with_data(
        &mut self,
        render_device: &RenderDevice,
        key: GpuSuballocatorKey,
        contents: &[u32],
        label: Option<&str>,
    ) -> GpuAllocation {
        let slab_index = self.large_slabs.len();
        let slab_label = self.create_slab_label(slab_index, label);

        self.large_slabs.push(
            render_device.create_buffer_with_data(&BufferInitDescriptor {
                usage: self.class.buffer_usages(),
                contents: bytemuck::cast_slice(contents),
                label: Some(&slab_label),
            }),
        );

        GpuAllocation {
            key,
            slab: slab_index as u32,
            offset: None,
            size: (contents.len() as u64 * 4) / self.class.element_size(),
        }
    }

    fn create_slab_label(&self, slab_index: usize, label: Option<&str>) -> String {
        let mut slab_label = format!("{} buffer {}", self.class, slab_index);
        if let Some(label) = label {
            let _ = write!(&mut slab_label, " \"{}\"", label);
        }
        slab_label
    }

    fn get_buffer_slice<'a>(&'a self, allocation: &GpuAllocation) -> BufferSlice<'a> {
        match allocation.offset {
            None => self.large_slabs[allocation.slab as usize].slice(..),
            Some(offset) => self.main_slabs[allocation.slab as usize]
                .buffer
                .slice(offset..(offset + allocation.size)),
        }
    }

    fn get_buffer_and_offset<'a>(&'a self, allocation: &GpuAllocation) -> (&'a Buffer, u64) {
        match allocation.offset {
            None => (&self.large_slabs[allocation.slab as usize], 0),
            Some(offset) => (&self.main_slabs[allocation.slab as usize].buffer, offset),
        }
    }
}

impl GpuAllocationClass {
    fn buffer_usages(&self) -> BufferUsages {
        match *self {
            GpuAllocationClass::Vertex(_) => BufferUsages::VERTEX,
            GpuAllocationClass::Index => BufferUsages::INDEX,
        }
    }

    fn element_size(&self) -> u64 {
        match self {
            GpuAllocationClass::Vertex(layout) => layout.0.layout().array_stride,
            GpuAllocationClass::Index => mem::size_of::<u32>() as u64,
        }
    }
}

impl GpuAllocation {
    #[inline]
    pub fn offset(&self) -> u64 {
        self.offset.unwrap_or_default()
    }
}

impl GpuSlab {
    fn new(render_device: &RenderDevice, class: GpuAllocationClass, slab_label: String) -> GpuSlab {
        GpuSlab {
            buffer: render_device.create_buffer(&BufferDescriptor {
                label: Some(&slab_label),
                size: SLAB_SIZE,
                // We have to be able to copy to the buffer!
                usage: class.buffer_usages() | BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
            allocator: Allocator::new((SLAB_SIZE / class.element_size()) as u32, 65534),
        }
    }

    fn allocate_with_data(
        &mut self,
        render_queue: &RenderQueue,
        slab_index: u32,
        class: &GpuAllocationClass,
        key: GpuSuballocatorKey,
        contents: &[u32],
    ) -> Option<GpuAllocation> {
        let element_size = class.element_size();
        let size = (contents.len() as u64 * 4).div_ceil(element_size) as u32;
        let allocation = self.allocator.allocate(size)?;
        let offset = u32::from(allocation.offset.unwrap_or_default()) as u64;
        render_queue.write_buffer(
            &self.buffer,
            offset * element_size,
            bytemuck::cast_slice(contents),
        );

        Some(GpuAllocation {
            key,
            slab: slab_index,
            offset: Some(offset),
            size: size as u64,
        })
    }
}

impl Display for GpuAllocationClass {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match *self {
            GpuAllocationClass::Vertex(_) => write!(f, "vertex"),
            GpuAllocationClass::Index => write!(f, "index"),
        }
    }
}
