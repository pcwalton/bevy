use crate::{
    define_atomic_id,
    render_asset::RenderAssets,
    render_resource::{resource_macros::*, BindGroupLayout, Buffer, Sampler, TextureView},
    renderer::{RenderDevice, RenderQueue},
    texture::{FallbackImage, GpuImage},
    RenderApp,
};
use bevy_app::{App, Plugin};
use bevy_derive::{Deref, DerefMut};
use bevy_ecs::system::Resource;
use bevy_math::UVec4;
pub use bevy_render_macros::AsBindGroup;
use bevy_utils::{hashbrown::HashMap, prelude::default, tracing::error};
use encase::ShaderType;
use nonmax::NonMaxU32;
use std::{collections::BTreeMap, mem, ops::Deref};
use thiserror::Error;
use wgpu::{
    BindGroupEntry, BindGroupLayoutEntry, BindingResource, BufferDescriptor, BufferUsages, Features,
};

define_atomic_id!(BindGroupId);
render_resource_wrapper!(ErasedBindGroup, wgpu::BindGroup);

pub struct RenderResourcePlugin {
    pub enable_bindless_textures: bool,
}

impl Plugin for RenderResourcePlugin {
    fn build(&self, _: &mut App) {}

    fn finish(&self, app: &mut App) {
        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };

        let render_device = render_app.world().resource::<RenderDevice>();
        render_app.insert_resource(RenderBindGroupStore::new(
            self.enable_bindless_textures,
            render_device,
        ));
    }
}

impl Default for RenderResourcePlugin {
    fn default() -> Self {
        Self {
            enable_bindless_textures: true,
        }
    }
}

/// Bind groups are responsible for binding render resources (e.g. buffers, textures, samplers)
/// to a [`TrackedRenderPass`](crate::render_phase::TrackedRenderPass).
/// This makes them accessible in the pipeline (shaders) as uniforms.
///
/// May be converted from and dereferences to a wgpu [`BindGroup`](wgpu::BindGroup).
/// Can be created via [`RenderDevice::create_bind_group`](RenderDevice::create_bind_group).
#[derive(Clone, Debug)]
pub struct BindGroup {
    id: BindGroupId,
    value: ErasedBindGroup,
}

impl BindGroup {
    /// Returns the [`BindGroupId`].
    #[inline]
    pub fn id(&self) -> BindGroupId {
        self.id
    }
}

impl From<wgpu::BindGroup> for BindGroup {
    fn from(value: wgpu::BindGroup) -> Self {
        BindGroup {
            id: BindGroupId::new(),
            value: ErasedBindGroup::new(value),
        }
    }
}

impl Deref for BindGroup {
    type Target = wgpu::BindGroup;

    #[inline]
    fn deref(&self) -> &Self::Target {
        &self.value
    }
}

pub enum BindingRefArray<'a> {
    Sampler(Vec<&'a wgpu::Sampler>),
    TextureView(Vec<&'a wgpu::TextureView>),
}

/// Converts a value to a [`BindGroup`] with a given [`BindGroupLayout`], which can then be used in Bevy shaders.
/// This trait can be derived (and generally should be). Read on for details and examples.
///
/// This is an opinionated trait that is intended to make it easy to generically
/// convert a type into a [`BindGroup`]. It provides access to specific render resources,
/// such as [`RenderAssets<GpuImage>`] and [`FallbackImage`]. If a type has a [`Handle<Image>`](bevy_asset::Handle),
/// these can be used to retrieve the corresponding [`Texture`](crate::render_resource::Texture) resource.
///
/// [`AsBindGroup::as_bind_group`] is intended to be called once, then the result cached somewhere. It is generally
/// ok to do "expensive" work here, such as creating a [`Buffer`] for a uniform.
///
/// If for some reason a [`BindGroup`] cannot be created yet (for example, the [`Texture`](crate::render_resource::Texture)
/// for an [`Image`](crate::texture::Image) hasn't loaded yet), just return [`AsBindGroupError::RetryNextUpdate`], which signals that the caller
/// should retry again later.
///
/// # Deriving
///
/// This trait can be derived. Field attributes like `uniform` and `texture` are used to define which fields should be bindings,
/// what their binding type is, and what index they should be bound at:
///
/// ```
/// # use bevy_render::{render_resource::*, texture::Image};
/// # use bevy_color::LinearRgba;
/// # use bevy_asset::Handle;
/// #[derive(AsBindGroup)]
/// struct CoolMaterial {
///     #[uniform(0)]
///     color: LinearRgba,
///     #[texture(1)]
///     #[sampler(2)]
///     color_texture: Handle<Image>,
///     #[storage(3, read_only)]
///     values: Vec<f32>,
///     #[storage(4, read_only, buffer)]
///     buffer: Buffer,
///     #[storage_texture(5)]
///     storage_texture: Handle<Image>,
/// }
/// ```
///
/// In WGSL shaders, the binding would look like this:
///
/// ```wgsl
/// @group(2) @binding(0) var<uniform> color: vec4<f32>;
/// @group(2) @binding(1) var color_texture: texture_2d<f32>;
/// @group(2) @binding(2) var color_sampler: sampler;
/// @group(2) @binding(3) var<storage> values: array<f32>;
/// @group(2) @binding(5) var storage_texture: texture_storage_2d<rgba8unorm, read_write>;
/// ```
/// Note that the "group" index is determined by the usage context. It is not defined in [`AsBindGroup`]. For example, in Bevy material bind groups
/// are generally bound to group 2.
///
/// The following field-level attributes are supported:
///
/// * `uniform(BINDING_INDEX)`
///     * The field will be converted to a shader-compatible type using the [`ShaderType`] trait, written to a [`Buffer`], and bound as a uniform.
///         [`ShaderType`] is implemented for most math types already, such as [`f32`], [`Vec4`](bevy_math::Vec4), and
///         [`LinearRgba`](bevy_color::LinearRgba). It can also be derived for custom structs.
///
/// * `texture(BINDING_INDEX, arguments)`
///     * This field's [`Handle<Image>`](bevy_asset::Handle) will be used to look up the matching [`Texture`](crate::render_resource::Texture)
///         GPU resource, which will be bound as a texture in shaders. The field will be assumed to implement [`Into<Option<Handle<Image>>>`]. In practice,
///         most fields should be a [`Handle<Image>`](bevy_asset::Handle) or [`Option<Handle<Image>>`]. If the value of an [`Option<Handle<Image>>`] is
///         [`None`], the [`FallbackImage`] resource will be used instead. This attribute can be used in conjunction with a `sampler` binding attribute
///         (with a different binding index) if a binding of the sampler for the [`Image`](crate::texture::Image) is also required.
///
/// | Arguments             | Values                                                                  | Default              |
/// |-----------------------|-------------------------------------------------------------------------|----------------------|
/// | `dimension` = "..."   | `"1d"`, `"2d"`, `"2d_array"`, `"3d"`, `"cube"`, `"cube_array"`          | `"2d"`               |
/// | `sample_type` = "..." | `"float"`, `"depth"`, `"s_int"` or `"u_int"`                            | `"float"`            |
/// | `filterable` = ...    | `true`, `false`                                                         | `true`               |
/// | `multisampled` = ...  | `true`, `false`                                                         | `false`              |
/// | `visibility(...)`     | `all`, `none`, or a list-combination of `vertex`, `fragment`, `compute` | `vertex`, `fragment` |
///
/// * `storage_texture(BINDING_INDEX, arguments)`
///     * This field's [`Handle<Image>`](bevy_asset::Handle) will be used to look up the matching [`Texture`](crate::render_resource::Texture)
///         GPU resource, which will be bound as a storage texture in shaders. The field will be assumed to implement [`Into<Option<Handle<Image>>>`]. In practice,
///         most fields should be a [`Handle<Image>`](bevy_asset::Handle) or [`Option<Handle<Image>>`]. If the value of an [`Option<Handle<Image>>`] is
///         [`None`], the [`FallbackImage`] resource will be used instead.
///
/// | Arguments              | Values                                                                                     | Default       |
/// |------------------------|--------------------------------------------------------------------------------------------|---------------|
/// | `dimension` = "..."    | `"1d"`, `"2d"`, `"2d_array"`, `"3d"`, `"cube"`, `"cube_array"`                             | `"2d"`        |
/// | `image_format` = ...   | any member of [`TextureFormat`](crate::render_resource::TextureFormat)                     | `Rgba8Unorm`  |
/// | `access` = ...         | any member of [`StorageTextureAccess`](crate::render_resource::StorageTextureAccess)       | `ReadWrite`   |
/// | `visibility(...)`      | `all`, `none`, or a list-combination of `vertex`, `fragment`, `compute`                    | `compute`     |
///
/// * `sampler(BINDING_INDEX, arguments)`
///     * This field's [`Handle<Image>`](bevy_asset::Handle) will be used to look up the matching [`Sampler`] GPU
///         resource, which will be bound as a sampler in shaders. The field will be assumed to implement [`Into<Option<Handle<Image>>>`]. In practice,
///         most fields should be a [`Handle<Image>`](bevy_asset::Handle) or [`Option<Handle<Image>>`]. If the value of an [`Option<Handle<Image>>`] is
///         [`None`], the [`FallbackImage`] resource will be used instead. This attribute can be used in conjunction with a `texture` binding attribute
///         (with a different binding index) if a binding of the texture for the [`Image`](crate::texture::Image) is also required.
///
/// | Arguments              | Values                                                                  | Default                |
/// |------------------------|-------------------------------------------------------------------------|------------------------|
/// | `sampler_type` = "..." | `"filtering"`, `"non_filtering"`, `"comparison"`.                       |  `"filtering"`         |
/// | `visibility(...)`      | `all`, `none`, or a list-combination of `vertex`, `fragment`, `compute` |   `vertex`, `fragment` |
///
/// * `storage(BINDING_INDEX, arguments)`
///     * The field will be converted to a shader-compatible type using the [`ShaderType`] trait, written to a [`Buffer`], and bound as a storage buffer.
///     * It supports and optional `read_only` parameter. Defaults to false if not present.
///
/// | Arguments              | Values                                                                  | Default              |
/// |------------------------|-------------------------------------------------------------------------|----------------------|
/// | `visibility(...)`      | `all`, `none`, or a list-combination of `vertex`, `fragment`, `compute` | `vertex`, `fragment` |
/// | `read_only`            | if present then value is true, otherwise false                          | `false`              |
///
/// Note that fields without field-level binding attributes will be ignored.
/// ```
/// # use bevy_render::{render_resource::AsBindGroup};
/// # use bevy_color::LinearRgba;
/// # use bevy_asset::Handle;
/// #[derive(AsBindGroup)]
/// struct CoolMaterial {
///     #[uniform(0)]
///     color: LinearRgba,
///     this_field_is_ignored: String,
/// }
/// ```
///
///  As mentioned above, [`Option<Handle<Image>>`] is also supported:
/// ```
/// # use bevy_render::{render_resource::AsBindGroup, texture::Image};
/// # use bevy_color::LinearRgba;
/// # use bevy_asset::Handle;
/// #[derive(AsBindGroup)]
/// struct CoolMaterial {
///     #[uniform(0)]
///     color: LinearRgba,
///     #[texture(1)]
///     #[sampler(2)]
///     color_texture: Option<Handle<Image>>,
/// }
/// ```
/// This is useful if you want a texture to be optional. When the value is [`None`], the [`FallbackImage`] will be used for the binding instead, which defaults
/// to "pure white".
///
/// Field uniforms with the same index will be combined into a single binding:
/// ```
/// # use bevy_render::{render_resource::AsBindGroup};
/// # use bevy_color::LinearRgba;
/// #[derive(AsBindGroup)]
/// struct CoolMaterial {
///     #[uniform(0)]
///     color: LinearRgba,
///     #[uniform(0)]
///     roughness: f32,
/// }
/// ```
///
/// In WGSL shaders, the binding would look like this:
/// ```wgsl
/// struct CoolMaterial {
///     color: vec4<f32>,
///     roughness: f32,
/// };
///
/// @group(2) @binding(0) var<uniform> material: CoolMaterial;
/// ```
///
/// Some less common scenarios will require "struct-level" attributes. These are the currently supported struct-level attributes:
/// * `uniform(BINDING_INDEX, ConvertedShaderType)`
///     * This also creates a [`Buffer`] using [`ShaderType`] and binds it as a uniform, much
///         like the field-level `uniform` attribute. The difference is that the entire [`AsBindGroup`] value is converted to `ConvertedShaderType`,
///         which must implement [`ShaderType`], instead of a specific field implementing [`ShaderType`]. This is useful if more complicated conversion
///         logic is required. The conversion is done using the [`AsBindGroupShaderType<ConvertedShaderType>`] trait, which is automatically implemented
///         if `&Self` implements [`Into<ConvertedShaderType>`]. Only use [`AsBindGroupShaderType`] if access to resources like [`RenderAssets<GpuImage>`] is
///         required.
/// * `bind_group_data(DataType)`
///     * The [`AsBindGroup`] type will be converted to some `DataType` using [`Into<DataType>`] and stored
///         as [`AsBindGroup::Data`] as part of the [`AsBindGroup::as_bind_group`] call. This is useful if data needs to be stored alongside
///         the generated bind group, such as a unique identifier for a material's bind group. The most common use case for this attribute
///         is "shader pipeline specialization". See [`SpecializedRenderPipeline`](crate::render_resource::SpecializedRenderPipeline).
///
/// The previous `CoolMaterial` example illustrating "combining multiple field-level uniform attributes with the same binding index" can
/// also be equivalently represented with a single struct-level uniform attribute:
/// ```
/// # use bevy_render::{render_resource::{AsBindGroup, ShaderType}};
/// # use bevy_color::LinearRgba;
/// #[derive(AsBindGroup)]
/// #[uniform(0, CoolMaterialUniform)]
/// struct CoolMaterial {
///     color: LinearRgba,
///     roughness: f32,
/// }
///
/// #[derive(ShaderType)]
/// struct CoolMaterialUniform {
///     color: LinearRgba,
///     roughness: f32,
/// }
///
/// impl From<&CoolMaterial> for CoolMaterialUniform {
///     fn from(material: &CoolMaterial) -> CoolMaterialUniform {
///         CoolMaterialUniform {
///             color: material.color,
///             roughness: material.roughness,
///         }
///     }
/// }
/// ```
///
/// Setting `bind_group_data` looks like this:
/// ```
/// # use bevy_render::{render_resource::AsBindGroup};
/// # use bevy_color::LinearRgba;
/// #[derive(AsBindGroup)]
/// #[bind_group_data(CoolMaterialKey)]
/// struct CoolMaterial {
///     #[uniform(0)]
///     color: LinearRgba,
///     is_shaded: bool,
/// }
///
/// #[derive(Copy, Clone, Hash, Eq, PartialEq)]
/// struct CoolMaterialKey {
///     is_shaded: bool,
/// }
///
/// impl From<&CoolMaterial> for CoolMaterialKey {
///     fn from(material: &CoolMaterial) -> CoolMaterialKey {
///         CoolMaterialKey {
///             is_shaded: material.is_shaded,
///         }
///     }
/// }
/// ```
pub trait AsBindGroup: Sized + 'static {
    /// Data that will be stored alongside the "prepared" bind group.
    type Data: Send + Sync;

    /// label
    fn label() -> Option<&'static str> {
        None
    }

    /// Creates a bind group for `self` matching the layout defined in
    /// [`AsBindGroup::bind_group_layout`].
    fn as_bind_group(
        &self,
        layout: &BindGroupLayout,
        render_device: &RenderDevice,
        render_queue: &RenderQueue,
        images: &RenderAssets<GpuImage>,
        fallback_image: &FallbackImage,
        bind_group_store: &mut RenderBindGroupStore,
    ) -> Result<PreparedBindGroup<Self::Data>, AsBindGroupError> {
        let UnpreparedBindGroup {
            bindings,
            array_uniforms,
            bindless_resources,
            data,
        } = Self::unprepared_bind_group(
            self,
            layout,
            render_device,
            render_queue,
            images,
            fallback_image,
            bind_group_store.bindless_textures_enabled,
        )?;

        let array_size = array_uniforms
            .as_ref()
            .map(|array_uniforms| array_uniforms.count)
            .unwrap_or_default();
        let bind_group_id = bind_group_store.get_or_create_bind_group_id(&bindings, array_size);

        // FIXME: don't make this if there are no bindless resources.
        let (mut buffer_arrays, mut bindless, bindless_index) = match bind_group_store
            .bind_group_id_to_bind_group
            .remove(&bind_group_id)
        {
            None => (
                HashMap::new(),
                bindless_resources
                    .iter()
                    .map(|bindless_resources| RenderBindlessResources {
                        binding_arrays: default(),
                        binding_array_indices: default(),
                        binding_array_index_buffer_binding: bindless_resources
                            .binding_array_index_buffer_binding,
                        binding_array_index_buffer: render_device.create_buffer(
                            &BufferDescriptor {
                                label: Some("binding array index buffer"),
                                size: bindless_resources.resources.len().div_ceil(4) as u64
                                    * mem::size_of::<UVec4>() as u64
                                    * array_uniforms
                                        .as_ref()
                                        .map(|array_uniforms| array_uniforms.count)
                                        .unwrap_or_default()
                                        as u64,
                                usage: BufferUsages::COPY_DST | BufferUsages::STORAGE,
                                mapped_at_creation: false,
                            },
                        ),
                    })
                    .collect(),
                RenderBindlessIndex(0),
            ),
            Some(RenderBindGroup {
                bind_group: _,
                buffer_arrays,
                bindless,
                bindless_count,
            }) => (buffer_arrays, bindless, RenderBindlessIndex(bindless_count)),
        };

        if let Some(array_uniforms) = array_uniforms {
            for (binding_id, data) in array_uniforms.uniforms {
                let aligned_data_size = align_up(data.len() as u32, 16);
                let buffer = buffer_arrays.entry(binding_id).or_insert_with(|| {
                    render_device.create_buffer(&BufferDescriptor {
                        label: Some("buffer array"),
                        size: aligned_data_size as u64 * array_uniforms.count as u64,
                        usage: BufferUsages::COPY_DST | BufferUsages::UNIFORM,
                        mapped_at_creation: false,
                    })
                });
                render_queue.write_buffer(
                    buffer,
                    aligned_data_size as u64 * bindless_index.0 as u64,
                    &data,
                );
            }
        }

        let bind_group = if !bindless.is_empty() {
            let mut binding_array_refs = HashMap::new();
            let mut entries = vec![];

            for (bindless, bindless_resources) in
                bindless.iter_mut().zip(bindless_resources.into_iter())
            {
                // Push new bindless resources.
                for (binding_id, bindless_resource) in bindless_resources.resources {
                    match bindless_resource {
                        UnpreparedBindlessResource::TextureView(Some(texture_view)) => {
                            match *bindless
                                .binding_arrays
                                .entry(binding_id)
                                .or_insert_with(|| BindingArray::Texture(vec![]))
                            {
                                BindingArray::Texture(ref mut textures) => {
                                    bindless.binding_array_indices.push(textures.len() as u32);
                                    textures.push(texture_view);
                                }
                                _ => error!("Expected texture view array"),
                            }
                        }

                        UnpreparedBindlessResource::Sampler(Some(sampler)) => {
                            match *bindless
                                .binding_arrays
                                .entry(binding_id)
                                .or_insert_with(|| BindingArray::Sampler(vec![]))
                            {
                                BindingArray::Sampler(ref mut samplers) => {
                                    bindless.binding_array_indices.push(samplers.len() as u32);
                                    samplers.push(sampler);
                                }
                                _ => error!("Expected sampler array"),
                            }
                        }

                        UnpreparedBindlessResource::TextureView(None)
                        | UnpreparedBindlessResource::Sampler(None) => {
                            bindless.binding_array_indices.push(!0);
                        }
                    }
                }

                while bindless.binding_array_indices.len() % 4 != 0 {
                    bindless.binding_array_indices.push(!0);
                }

                // FIXME: Why are we writing this whole buffer?
                render_queue.write_buffer(
                    &bindless.binding_array_index_buffer,
                    0,
                    bytemuck::cast_slice(&bindless.binding_array_indices[..]),
                );
            }

            for bindless in &bindless {
                // Rebuild binding arrays.
                //
                // FIXME: What a horrid hack. We should fix `wgpu`'s bindless API to not
                // require references on the inner level.
                for (binding_id, binding) in &bindless.binding_arrays {
                    match *binding {
                        BindingArray::Sampler(ref samplers) => {
                            let sampler_refs = BindingRefArray::Sampler(
                                samplers
                                    .iter()
                                    .map(|bevy_sampler| &**bevy_sampler)
                                    .collect(),
                            );
                            binding_array_refs.insert(*binding_id, sampler_refs);
                        }

                        BindingArray::Texture(ref textures) => {
                            let texture_refs = BindingRefArray::TextureView(
                                textures
                                    .iter()
                                    .map(|bevy_texture_view| &**bevy_texture_view)
                                    .collect(),
                            );
                            binding_array_refs.insert(*binding_id, texture_refs);
                        }
                    }
                }
            }

            for (index, binding) in &bindings {
                entries.push(BindGroupEntry {
                    binding: *index,
                    resource: binding.get_binding(
                        Some(&binding_array_refs),
                        &buffer_arrays,
                        fallback_image,
                    ),
                });
            }

            for bindless in &bindless {
                entries.push(BindGroupEntry {
                    binding: bindless.binding_array_index_buffer_binding,
                    resource: bindless.binding_array_index_buffer.as_entire_binding(),
                });
            }

            render_device.create_bind_group(Self::label(), layout, &entries)
        } else {
            let entries = bindings
                .iter()
                .map(|(index, binding)| BindGroupEntry {
                    binding: *index,
                    resource: binding.get_binding(None, &buffer_arrays, fallback_image),
                })
                .collect::<Vec<_>>();
            render_device.create_bind_group(Self::label(), layout, &entries)
        };

        bind_group_store.bind_group_id_to_bind_group.insert(
            bind_group_id,
            RenderBindGroup {
                bind_group,
                buffer_arrays,
                bindless,
                bindless_count: bindless_index.0 + 1,
            },
        );

        Ok(PreparedBindGroup {
            bindings,
            bind_group_id,
            bindless_index,
            data,
        })
    }

    /// Returns a vec of (binding index, `OwnedBindingResource`).  In cases
    /// where `OwnedBindingResource` is not available (as for bindless texture
    /// arrays currently), an implementor may define `as_bind_group` directly.
    /// This may prevent certain features from working correctly.
    fn unprepared_bind_group(
        &self,
        layout: &BindGroupLayout,
        render_device: &RenderDevice,
        render_queue: &RenderQueue,
        images: &RenderAssets<GpuImage>,
        fallback_image: &FallbackImage,
        bindless: bool,
    ) -> Result<UnpreparedBindGroup<Self::Data>, AsBindGroupError>;

    /// Creates the bind group layout matching all bind groups returned by [`AsBindGroup::as_bind_group`]
    fn bind_group_layout(render_device: &RenderDevice, bindless: bool) -> BindGroupLayout
    where
        Self: Sized,
    {
        render_device.create_bind_group_layout(
            Self::label(),
            &Self::bind_group_layout_entries(render_device, bindless),
        )
    }

    /// Returns a vec of bind group layout entries
    fn bind_group_layout_entries(
        render_device: &RenderDevice,
        bindless: bool,
    ) -> Vec<BindGroupLayoutEntry>
    where
        Self: Sized;
}

/// An error that occurs during [`AsBindGroup::as_bind_group`] calls.
#[derive(Debug, Error)]
pub enum AsBindGroupError {
    /// The bind group could not be generated. Try again next frame.
    #[error("The bind group could not be generated")]
    RetryNextUpdate,
}

pub type OwnedBindings = Vec<(u32, OwnedBindingResource)>;

/// A prepared bind group returned as a result of [`AsBindGroup::as_bind_group`].
pub struct PreparedBindGroup<D> {
    pub bindings: OwnedBindings,
    pub bind_group_id: RenderBindGroupId,
    pub bindless_index: RenderBindlessIndex,
    pub data: D,
}

/// a map containing `OwnedBindingResource`s, keyed by the target binding index
pub struct UnpreparedBindGroup<D> {
    pub bindings: Vec<(u32, OwnedBindingResource)>,
    pub array_uniforms: Option<UnpreparedArrayUniforms>,
    pub bindless_resources: Vec<UnpreparedBindlessResources>,
    pub data: D,
}

pub struct UnpreparedArrayUniforms {
    pub count: u32,
    pub uniforms: Vec<(u32, Vec<u8>)>,
}

pub struct UnpreparedBindlessResources {
    pub resources: BTreeMap<u32, UnpreparedBindlessResource>,
    pub binding_array_index_buffer_binding: u32,
}

pub enum UnpreparedBindlessResource {
    TextureView(Option<TextureView>),
    Sampler(Option<Sampler>),
}

/// An owned binding resource of any type (ex: a [`Buffer`], [`TextureView`], etc).
/// This is used by types like [`PreparedBindGroup`] to hold a single list of all
/// render resources used by bindings.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum OwnedBindingResource {
    Buffer(Buffer),
    TextureView(TextureView),
    Sampler(Sampler),
    BufferArray(u32),
    BindlessSampler(u32),
    BindlessTextureView(u32),
}

impl OwnedBindingResource {
    pub fn get_binding<'a>(
        &'a self,
        binding_ref_arrays: Option<&'a HashMap<u32, BindingRefArray<'a>>>,
        subgroup_buffer_bindings: &'a HashMap<u32, Buffer>,
        fallback_image: &'a FallbackImage,
    ) -> BindingResource<'a> {
        match self {
            OwnedBindingResource::Buffer(buffer) => buffer.as_entire_binding(),
            OwnedBindingResource::TextureView(view) => BindingResource::TextureView(view),
            OwnedBindingResource::Sampler(sampler) => BindingResource::Sampler(sampler),
            OwnedBindingResource::BufferArray(binding_id) => subgroup_buffer_bindings
                .get(binding_id)
                .expect("No subgroup buffer binding found")
                .as_entire_binding(),
            OwnedBindingResource::BindlessSampler(binding_id) => {
                match binding_ref_arrays
                    .expect(
                        "Binding ref arrays must be supplied if there are any bindless resources",
                    )
                    .get(binding_id)
                {
                    Some(BindingRefArray::Sampler(samplers)) => {
                        BindingResource::SamplerArray(samplers)
                    }
                    _ => BindingResource::Sampler(&fallback_image.d2.sampler),
                }
            }
            OwnedBindingResource::BindlessTextureView(binding_id) => {
                match binding_ref_arrays
                    .expect(
                        "Binding ref arrays must be supplied if there are any bindless resources",
                    )
                    .get(binding_id)
                {
                    Some(BindingRefArray::TextureView(textures)) => {
                        BindingResource::TextureViewArray(textures)
                    }
                    // TODO: non-2d textures
                    _ => BindingResource::TextureView(&fallback_image.d2.texture_view),
                }
            }
        }
    }
}

#[derive(Resource, Default)]
pub struct RenderBindGroupStore {
    /// We can have multiple bind groups here if the [`RenderBindlessIndex`]
    /// overflows.
    pub bindings_to_bind_group_id: HashMap<OwnedBindings, Vec<RenderBindGroupId>>,
    pub bind_group_id_to_bind_group: HashMap<RenderBindGroupId, RenderBindGroup>,
    pub next_bind_group_id: RenderBindGroupId,
    pub bindless_textures_enabled: bool,
}

#[derive(Clone, Copy, Default, PartialEq, PartialOrd, Eq, Ord, Hash, Deref, DerefMut, Debug)]
#[repr(transparent)]
pub struct RenderBindGroupId(pub NonMaxU32);

#[derive(Clone, Copy, Default, Deref, DerefMut)]
#[repr(transparent)]
pub struct RenderBindlessIndex(pub u32);

impl RenderBindGroupStore {
    fn new(enable_bindless_textures: bool, render_device: &RenderDevice) -> Self {
        let wgpu_features = render_device.features();
        let bindless_textures_enabled =
            enable_bindless_textures && wgpu_features.contains(Features::TEXTURE_BINDING_ARRAY);

        Self {
            bindings_to_bind_group_id: HashMap::new(),
            bind_group_id_to_bind_group: HashMap::new(),
            next_bind_group_id: default(),
            bindless_textures_enabled,
        }
    }

    pub fn get_or_create_bind_group_id(
        &mut self,
        bindings: &[(u32, OwnedBindingResource)],
        array_size: u32,
    ) -> RenderBindGroupId {
        if let Some(bind_group_ids) = self.bindings_to_bind_group_id.get(bindings) {
            for bind_group_id in bind_group_ids {
                if let Some(bind_group) = self.bind_group_id_to_bind_group.get(bind_group_id) {
                    if bind_group.bindless_count < array_size {
                        return *bind_group_id;
                    }
                }
            }
        }

        // Create a new bind group.
        let bind_group_id = self.next_bind_group_id;
        self.next_bind_group_id.0 = (u32::from(self.next_bind_group_id.0) + 1)
            .try_into()
            .unwrap_or_default();
        self.bindings_to_bind_group_id
            .entry(bindings.to_vec())
            .or_insert_with(default)
            .push(bind_group_id);
        bind_group_id
    }
}

pub struct RenderBindGroup {
    pub bind_group: BindGroup,
    pub buffer_arrays: HashMap<u32, Buffer>,
    pub bindless_count: u32,
    pub bindless: Vec<RenderBindlessResources>,
}

pub enum BindingArray {
    Sampler(Vec<Sampler>),
    Texture(Vec<TextureView>),
}

pub struct RenderBindlessResources {
    pub binding_arrays: HashMap<u32, BindingArray>,
    pub binding_array_indices: Vec<u32>,
    pub binding_array_index_buffer: Buffer,
    pub binding_array_index_buffer_binding: u32,
}

/// Converts a value to a [`ShaderType`] for use in a bind group.
/// This is automatically implemented for references that implement [`Into`].
/// Generally normal [`Into`] / [`From`] impls should be preferred, but
/// sometimes additional runtime metadata is required.
/// This exists largely to make some [`AsBindGroup`] use cases easier.
pub trait AsBindGroupShaderType<T: ShaderType> {
    /// Return the `T` [`ShaderType`] for `self`. When used in [`AsBindGroup`]
    /// derives, it is safe to assume that all images in `self` exist.
    fn as_bind_group_shader_type(&self, images: &RenderAssets<GpuImage>) -> T;
}

impl<T, U: ShaderType> AsBindGroupShaderType<U> for T
where
    for<'a> &'a T: Into<U>,
{
    #[inline]
    fn as_bind_group_shader_type(&self, _images: &RenderAssets<GpuImage>) -> U {
        self.into()
    }
}

fn align_up(x: u32, y: u32) -> u32 {
    if x % y == 0 {
        x
    } else {
        x + y - x % y
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::{self as bevy_render, prelude::Image};
    use bevy_asset::Handle;

    #[test]
    fn texture_visibility() {
        #[derive(AsBindGroup)]
        pub struct TextureVisibilityTest {
            #[texture(0, visibility(all))]
            pub all: Handle<Image>,
            #[texture(1, visibility(none))]
            pub none: Handle<Image>,
            #[texture(2, visibility(fragment))]
            pub fragment: Handle<Image>,
            #[texture(3, visibility(vertex))]
            pub vertex: Handle<Image>,
            #[texture(4, visibility(compute))]
            pub compute: Handle<Image>,
            #[texture(5, visibility(vertex, fragment))]
            pub vertex_fragment: Handle<Image>,
            #[texture(6, visibility(vertex, compute))]
            pub vertex_compute: Handle<Image>,
            #[texture(7, visibility(fragment, compute))]
            pub fragment_compute: Handle<Image>,
            #[texture(8, visibility(vertex, fragment, compute))]
            pub vertex_fragment_compute: Handle<Image>,
        }
    }
}
