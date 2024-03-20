use std::marker::PhantomData;

use super::Buffer;
use crate::renderer::{RenderDevice, RenderQueue};
use encase::{
    internal::{ReadFrom, WriteInto, Writer},
    DynamicStorageBuffer as DynamicStorageBufferWrapper, ShaderType,
    StorageBuffer as StorageBufferWrapper,
};
use wgpu::{util::BufferInitDescriptor, BindingResource, BufferBinding, BufferUsages};

/// Stores data to be transferred to the GPU and made accessible to shaders as a storage buffer.
///
/// Storage buffers can be made available to shaders in some combination of read/write mode, and can store large amounts of data.
/// Note however that WebGL2 does not support storage buffers, so consider alternative options in this case.
///
/// Storage buffers store an array of instances of type `T`, which must
/// implement [`ShaderType`].
///
/// The contained data is stored in system RAM. [`write_buffer`](StorageBuffer::write_buffer) queues
/// copying of the data from system RAM to VRAM. Storage buffers must conform to [std430 alignment/padding requirements], which
/// is automatically enforced by this structure.
///
/// Other options for storing GPU-accessible data are:
/// * [`DynamicStorageBuffer`]
/// * [`UniformBuffer`](crate::render_resource::UniformBuffer)
/// * [`DynamicUniformBuffer`](crate::render_resource::DynamicUniformBuffer)
/// * [`GpuArrayBuffer`](crate::render_resource::GpuArrayBuffer)
/// * [`BufferVec`](crate::render_resource::BufferVec)
/// * [`Texture`](crate::render_resource::Texture)
///
/// [std430 alignment/padding requirements]: https://www.w3.org/TR/WGSL/#address-spaces-storage
pub struct StorageBuffer<T>
where
    T: ShaderType + ReadFrom + WriteInto + Default,
{
    scratch: StorageBufferWrapper<Vec<u8>>,
    buffer: Option<Buffer>,
    label: Option<String>,
    changed: bool,
    buffer_usage: BufferUsages,
    phantom: PhantomData<T>,
}

impl<T> From<T> for StorageBuffer<T>
where
    T: ShaderType + ReadFrom + WriteInto + Default,
{
    fn from(value: T) -> Self {
        let mut scratch = StorageBufferWrapper::new(vec![]);
        scratch.write(&value).unwrap();
        Self {
            scratch,
            buffer: None,
            label: None,
            changed: false,
            buffer_usage: BufferUsages::COPY_DST | BufferUsages::STORAGE,
            phantom: PhantomData,
        }
    }
}

impl<T> Default for StorageBuffer<T>
where
    T: ShaderType + Default + ReadFrom + WriteInto,
{
    fn default() -> Self {
        let mut scratch = StorageBufferWrapper::new(vec![]);
        scratch.write(&T::default()).unwrap();
        Self {
            scratch,
            buffer: None,
            label: None,
            changed: false,
            buffer_usage: BufferUsages::COPY_DST | BufferUsages::STORAGE,
            phantom: PhantomData,
        }
    }
}

impl<T> StorageBuffer<T>
where
    T: ShaderType + WriteInto + ReadFrom + Default,
{
    #[inline]
    pub fn buffer(&self) -> Option<&Buffer> {
        self.buffer.as_ref()
    }

    #[inline]
    pub fn binding(&self) -> Option<BindingResource> {
        Some(BindingResource::Buffer(
            self.buffer()?.as_entire_buffer_binding(),
        ))
    }

    /// Deletes all data from this storage buffer.
    pub fn clear(&mut self) {
        self.scratch.as_mut().clear();
    }

    /// Returns the number of instances of `T` in this buffer.
    ///
    /// This is not the byte size.
    pub fn len(&self) -> u64 {
        self.scratch.as_ref().len() as u64 / u64::from(T::min_size())
    }

    /// True if no instances of `T` are present in this buffer.
    pub fn is_empty(&self) -> bool {
        self.scratch.as_ref().is_empty()
    }

    /// Adds a new instance of type `T` to the buffer.
    pub fn push(&mut self, item: T) {
        let backing_store = self.scratch.as_mut();
        let backing_len = backing_store.len();
        item.write_into(&mut Writer::new(&item, backing_store, backing_len).unwrap());
    }

    /// Adds many instances of type `T` to the buffer.
    pub fn extend(&mut self, items: impl Iterator<Item = T>) {
        for item in items {
            self.push(item);
        }
    }

    pub fn set_label(&mut self, label: Option<&str>) {
        let label = label.map(str::to_string);

        if label != self.label {
            self.changed = true;
        }

        self.label = label;
    }

    pub fn get_label(&self) -> Option<&str> {
        self.label.as_deref()
    }

    /// Add more [`BufferUsages`] to the buffer.
    ///
    /// This method only allows addition of flags to the default usage flags.
    ///
    /// The default values for buffer usage are `BufferUsages::COPY_DST` and `BufferUsages::STORAGE`.
    pub fn add_usages(&mut self, usage: BufferUsages) {
        self.buffer_usage |= usage;
        self.changed = true;
    }

    /// Queues writing of data from system RAM to VRAM using the [`RenderDevice`]
    /// and the provided [`RenderQueue`].
    ///
    /// If there is no GPU-side buffer allocated to hold the data currently stored, or if a GPU-side buffer previously
    /// allocated does not have enough capacity, a new GPU-side buffer is created.
    pub fn write_buffer(&mut self, device: &RenderDevice, queue: &RenderQueue) {
        let capacity = self.buffer.as_deref().map(wgpu::Buffer::size).unwrap_or(0);

        let size = self.scratch.as_ref().len() as u64;

        if capacity < size || self.changed {
            self.buffer = Some(device.create_buffer_with_data(&BufferInitDescriptor {
                label: self.label.as_deref(),
                usage: self.buffer_usage,
                contents: self.scratch.as_ref(),
            }));
            self.changed = false;
        } else if let Some(buffer) = &self.buffer {
            queue.write_buffer(buffer, 0, self.scratch.as_ref());
        }
    }

    /// Like [`StorageBuffer::write_buffer`], but if the buffer is empty,
    /// forcibly creates one with a single default entry.
    ///
    /// Ideally we shouldn't be doing this, but there are many places throughout
    /// rendering where we expect even zero-sized buffers to be manifested on
    /// GPU, and this hack keeps that working.
    pub fn force_write_buffer(&mut self, device: &RenderDevice, queue: &RenderQueue) {
        if self.is_empty() {
            self.push(T::default());
        }

        self.write_buffer(device, queue);
    }
}

/// Stores data to be transferred to the GPU and made accessible to shaders as a dynamic storage buffer.
///
/// Dynamic storage buffers can be made available to shaders in some combination of read/write mode, and can store large amounts
/// of data. Note however that WebGL2 does not support storage buffers, so consider alternative options in this case. Dynamic
/// storage buffers support multiple separate bindings at dynamic byte offsets and so have a
/// [`push`](DynamicStorageBuffer::push) method.
///
/// The contained data is stored in system RAM. [`write_buffer`](DynamicStorageBuffer::write_buffer)
/// queues copying of the data from system RAM to VRAM. The data within a storage buffer binding must conform to
/// [std430 alignment/padding requirements]. `DynamicStorageBuffer` takes care of serialising the inner type to conform to
/// these requirements. Each item [`push`](DynamicStorageBuffer::push)ed into this structure
/// will additionally be aligned to meet dynamic offset alignment requirements.
///
/// Other options for storing GPU-accessible data are:
/// * [`StorageBuffer`]
/// * [`UniformBuffer`](crate::render_resource::UniformBuffer)
/// * [`DynamicUniformBuffer`](crate::render_resource::DynamicUniformBuffer)
/// * [`GpuArrayBuffer`](crate::render_resource::GpuArrayBuffer)
/// * [`BufferVec`](crate::render_resource::BufferVec)
/// * [`Texture`](crate::render_resource::Texture)
///
/// [std430 alignment/padding requirements]: https://www.w3.org/TR/WGSL/#address-spaces-storage
pub struct DynamicStorageBuffer<T: ShaderType> {
    scratch: DynamicStorageBufferWrapper<Vec<u8>>,
    buffer: Option<Buffer>,
    label: Option<String>,
    changed: bool,
    buffer_usage: BufferUsages,
    _marker: PhantomData<fn() -> T>,
}

impl<T: ShaderType> Default for DynamicStorageBuffer<T> {
    fn default() -> Self {
        Self {
            scratch: DynamicStorageBufferWrapper::new(Vec::new()),
            buffer: None,
            label: None,
            changed: false,
            buffer_usage: BufferUsages::COPY_DST | BufferUsages::STORAGE,
            _marker: PhantomData,
        }
    }
}

impl<T: ShaderType + WriteInto> DynamicStorageBuffer<T> {
    #[inline]
    pub fn buffer(&self) -> Option<&Buffer> {
        self.buffer.as_ref()
    }

    #[inline]
    pub fn binding(&self) -> Option<BindingResource> {
        Some(BindingResource::Buffer(BufferBinding {
            buffer: self.buffer()?,
            offset: 0,
            size: Some(T::min_size()),
        }))
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.scratch.as_ref().is_empty()
    }

    #[inline]
    pub fn push(&mut self, value: T) -> u32 {
        self.scratch.write(&value).unwrap() as u32
    }

    pub fn set_label(&mut self, label: Option<&str>) {
        let label = label.map(str::to_string);

        if label != self.label {
            self.changed = true;
        }

        self.label = label;
    }

    pub fn get_label(&self) -> Option<&str> {
        self.label.as_deref()
    }

    /// Add more [`BufferUsages`] to the buffer.
    ///
    /// This method only allows addition of flags to the default usage flags.
    ///
    /// The default values for buffer usage are `BufferUsages::COPY_DST` and `BufferUsages::STORAGE`.
    pub fn add_usages(&mut self, usage: BufferUsages) {
        self.buffer_usage |= usage;
        self.changed = true;
    }

    #[inline]
    pub fn write_buffer(&mut self, device: &RenderDevice, queue: &RenderQueue) {
        let capacity = self.buffer.as_deref().map(wgpu::Buffer::size).unwrap_or(0);
        let size = self.scratch.as_ref().len() as u64;
        if capacity < size || self.changed {
            self.buffer = Some(device.create_buffer_with_data(&BufferInitDescriptor {
                label: self.label.as_deref(),
                usage: self.buffer_usage,
                contents: self.scratch.as_ref(),
            }));
            self.changed = false;
        } else if let Some(buffer) = &self.buffer {
            queue.write_buffer(buffer, 0, self.scratch.as_ref());
        }
    }

    #[inline]
    pub fn clear(&mut self) {
        self.scratch.as_mut().clear();
        self.scratch.set_offset(0);
    }
}
