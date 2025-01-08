//! Decals, textures that can be projected onto surfaces.

use core::{num::NonZero, ops::Deref};

use bevy_app::{App, Plugin};
use bevy_asset::{load_internal_asset, AssetId, Handle};
use bevy_derive::{Deref, DerefMut};
use bevy_ecs::{
    component::{require, Component},
    entity::{Entity, EntityHashMap},
    prelude::ReflectComponent,
    query::With,
    schedule::IntoSystemConfigs as _,
    system::{Query, Res, ResMut, Resource},
};
use bevy_image::Image;
use bevy_math::Mat4;
use bevy_reflect::Reflect;
use bevy_render::{
    extract_component::{ExtractComponent, ExtractComponentPlugin},
    render_asset::RenderAssets,
    render_resource::{
        binding_types, BindGroupLayoutEntryBuilder, Buffer, BufferUsages, RawBufferVec, Sampler,
        SamplerBindingType, Shader, ShaderType, TextureSampleType, TextureView,
    },
    renderer::{RenderAdapter, RenderDevice, RenderQueue},
    sync_world::RenderEntity,
    texture::{FallbackImage, GpuImage},
    view::{self, ViewVisibility, Visibility, VisibilityClass},
    Extract, ExtractSchedule, Render, RenderApp, RenderSet,
};
use bevy_transform::{components::GlobalTransform, prelude::Transform};
use bevy_utils::hashbrown::HashMap;
use bytemuck::{Pod, Zeroable};

use crate::{
    binding_arrays_are_usable, prepare_lights, GlobalClusterableObjectMeta, LightVisibilityClass,
};

pub(crate) const DECAL_SHADER_HANDLE: Handle<Shader> = Handle::weak_from_u128(2881025580737984685);

/// On WebGL and WebGPU, we must disable decals, as otherwise we can overflow
/// the number of texture bindings when deferred rendering is in use.
pub(crate) const DECALS_ARE_USABLE: bool = cfg!(not(target_arch = "wasm32"));

pub(crate) const MAX_VIEW_DECALS: usize = 16;

pub struct DecalPlugin;

/// An object that projects a decal onto surfaces within its bounds.
///
/// Conceptually, a decal projector is a 1×1×1 cube centered on its origin. It
/// projects the given [`Self::image`] onto surfaces in the +Z direction (thus
/// you may find [`Transform::looking_at`] useful).
#[derive(Component, Debug, Clone, Reflect, ExtractComponent)]
#[reflect(Component, Debug)]
#[require(Transform, Visibility, VisibilityClass)]
#[component(on_add = view::add_visibility_class::<LightVisibilityClass>)]
pub struct DecalProjector {
    /// The image that the decal projector projects.
    ///
    /// This must be a 2D image. If it has an alpha channel, it'll be alpha
    /// blended with the underlying surface and/or other decals. All images in
    /// the scene must use the same sampler.
    pub image: Handle<Image>,
}

/// Stores information about all the decals in the scene.
#[derive(Resource, Default)]
pub struct RenderDecals {
    /// Maps an index in the shader binding array to the associated decal image.
    ///
    /// [`Self::texture_to_binding_index`] holds the inverse mapping.
    binding_index_to_textures: Vec<AssetId<Image>>,
    /// Maps a decal image to the shader binding array.
    ///
    /// [`Self::binding_index_to_texture`] holds the inverse mapping.
    texture_to_binding_index: HashMap<AssetId<Image>, u32>,
    /// The information concerning each decal that we provide to the shader.
    decals: Vec<Decal>,
    /// Maps the [`bevy_render::sync_world::RenderEntity`] of each decal to the
    /// index of that decal in the [`Self::decals`] list.
    entity_to_decal_index: EntityHashMap<usize>,
}

impl RenderDecals {
    /// Clears out this [`RenderDecals`] in preparation for a new frame.
    fn clear(&mut self) {
        self.binding_index_to_textures.clear();
        self.texture_to_binding_index.clear();
        self.decals.clear();
        self.entity_to_decal_index.clear();
    }
}

/// The per-view bind group entries pertaining to decals.
pub(crate) struct RenderViewDecalBindGroupEntries<'a> {
    /// The list of decals, corresponding to `mesh_view_bindings::decals` in the
    /// shader.
    pub(crate) decals: &'a Buffer,
    /// The list of textures, corresponding to
    /// `mesh_view_bindings::decal_textures` in the shader.
    pub(crate) texture_views: RenderViewDecalTextureViews<'a>,
    /// The sampler that the shader uses to sample decals, corresponding to
    /// `mesh_view_bindings::decal_sampler` in the shader.
    pub(crate) sampler: &'a Sampler,
}

/// The per-view bind group entry for the texture view.
pub(crate) enum RenderViewDecalTextureViews<'a> {
    /// Multiple textures. This is used when binding arrays are usable on the
    /// current platform.
    BindingArray(Vec<&'a <TextureView as Deref>::Target>),
    /// A single texture. This is used when binding arrays aren't usable on the
    /// current platform.
    SingleBinding(&'a TextureView),
}

/// A render-world resource that holds the buffer of [`Decal`]s ready to upload
/// to the GPU.
#[derive(Resource, Deref, DerefMut)]
pub struct DecalsBuffer(RawBufferVec<Decal>);

impl Default for DecalsBuffer {
    fn default() -> Self {
        DecalsBuffer(RawBufferVec::new(BufferUsages::STORAGE))
    }
}

impl Plugin for DecalPlugin {
    fn build(&self, app: &mut App) {
        load_internal_asset!(app, DECAL_SHADER_HANDLE, "decal.wgsl", Shader::from_wgsl);

        app.add_plugins(ExtractComponentPlugin::<DecalProjector>::default())
            .register_type::<DecalProjector>();

        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };

        render_app
            .init_resource::<DecalsBuffer>()
            .init_resource::<RenderDecals>()
            .add_systems(ExtractSchedule, extract_decals)
            .add_systems(
                Render,
                prepare_decals
                    .in_set(RenderSet::ManageViews)
                    .after(prepare_lights),
            )
            .add_systems(Render, upload_decals.in_set(RenderSet::PrepareResources));
    }
}

/// The GPU data structure that stores information about each decal.
#[derive(Clone, Copy, Default, ShaderType, Pod, Zeroable)]
#[repr(C)]
pub struct Decal {
    /// The inverse of the model matrix.
    ///
    /// The shader uses this in order to back-transform world positions into
    /// model space.
    local_from_world: Mat4,
    /// The index of the decal texture in the binding array.
    image_index: u32,
    /// Padding.
    pad_a: u32,
    /// Padding.
    pad_b: u32,
    /// Padding.
    pad_c: u32,
}

/// Extracts decals from the main world into the render world.
pub fn extract_decals(
    decals: Extract<
        Query<(
            RenderEntity,
            &DecalProjector,
            &GlobalTransform,
            &ViewVisibility,
        )>,
    >,
    mut render_decals: ResMut<RenderDecals>,
) {
    // Clear out the `RenderDecals` in preparation for a new frame.
    render_decals.clear();

    // Loop over each decal.
    for (decal_entity, decal_projector, global_transform, view_visibility) in &decals {
        // If the decal is invisible, skip it.
        if !view_visibility.get() {
            continue;
        }

        // Insert or add the image.
        let image_index = render_decals.get_or_insert_image(&decal_projector.image.id());

        // Record the decal.
        let decal_index = render_decals.decals.len();
        render_decals
            .entity_to_decal_index
            .insert(decal_entity, decal_index);

        render_decals.decals.push(Decal {
            local_from_world: global_transform.affine().inverse().into(),
            image_index,
            pad_a: 0,
            pad_b: 0,
            pad_c: 0,
        });
    }
}

/// Adds all decals in the scene to the [`GlobalClusterableObjectMeta`] table.
fn prepare_decals(
    decals: Query<Entity, With<DecalProjector>>,
    mut global_clusterable_object_meta: ResMut<GlobalClusterableObjectMeta>,
    render_decals: Res<RenderDecals>,
) {
    for decal_entity in &decals {
        if let Some(index) = render_decals.entity_to_decal_index.get(&decal_entity) {
            global_clusterable_object_meta
                .entity_to_index
                .insert(decal_entity, *index);
        }
    }
}

/// Returns the layout for the decal-related bind group entries for a single
/// view.
pub(crate) fn get_bind_group_layout_entries(
    render_device: &RenderDevice,
    render_adapter: &RenderAdapter,
) -> [BindGroupLayoutEntryBuilder; 3] {
    // If binding arrays aren't supported on the current platform, use only a
    // single texture.
    let mut texture_2d_binding =
        binding_types::texture_2d(TextureSampleType::Float { filterable: true });
    if binding_arrays_are_usable(render_device, render_adapter) {
        texture_2d_binding =
            texture_2d_binding.count(NonZero::<u32>::new(MAX_VIEW_DECALS as u32).unwrap());
    };

    [
        // `decals`
        binding_types::storage_buffer_read_only::<Decal>(false),
        // `decal_textures`
        texture_2d_binding,
        // `decal_sampler`
        binding_types::sampler(SamplerBindingType::Filtering),
    ]
}

impl<'a> RenderViewDecalBindGroupEntries<'a> {
    /// Creates and returns the bind group entries for decals for a single view.
    pub(crate) fn get(
        render_decals: &RenderDecals,
        decals_buffer: &'a DecalsBuffer,
        images: &'a RenderAssets<GpuImage>,
        fallback_image: &'a FallbackImage,
        render_device: &RenderDevice,
        render_adapter: &RenderAdapter,
    ) -> Option<RenderViewDecalBindGroupEntries<'a>> {
        // We use the first sampler among all the images. This assumes that all
        // images use the same sampler, which is a documented restriction. If
        // there's no sampler, we just use the one from the fallback image.
        let sampler = match render_decals
            .binding_index_to_textures
            .iter()
            .filter_map(|image_id| images.get(*image_id))
            .next()
        {
            Some(gpu_image) => &gpu_image.sampler,
            None => &fallback_image.d2.sampler,
        };

        // Gather up the decal textures. If binding arrays are usable on the
        // current platform, then we gather up multiple decals. Otherwise, we
        // handle just one.
        let texture_views = if binding_arrays_are_usable(render_device, render_adapter) {
            // Collect all the decals in the scene.
            let mut texture_views = vec![];
            for image_id in &render_decals.binding_index_to_textures {
                match images.get(*image_id) {
                    None => texture_views.push(&*fallback_image.d2.texture_view),
                    Some(gpu_image) => texture_views.push(&*gpu_image.texture_view),
                }
            }

            // Pad out the binding array to its maximum length, which is
            // required on some platforms.
            while texture_views.len() < MAX_VIEW_DECALS {
                texture_views.push(&*fallback_image.d2.texture_view);
            }

            RenderViewDecalTextureViews::BindingArray(texture_views)
        } else {
            // Take the first decal.
            let texture_view = match render_decals
                .binding_index_to_textures
                .iter()
                .filter_map(|image_id| images.get(*image_id))
                .next()
            {
                Some(gpu_image) => &gpu_image.texture_view,
                None => &fallback_image.d2.texture_view,
            };
            RenderViewDecalTextureViews::SingleBinding(texture_view)
        };

        Some(RenderViewDecalBindGroupEntries {
            decals: decals_buffer.buffer()?,
            texture_views,
            sampler,
        })
    }
}

impl RenderDecals {
    /// Returns the index of the given image in the decal texture binding array,
    /// adding it to the list if necessary.
    fn get_or_insert_image(&mut self, image_id: &AssetId<Image>) -> u32 {
        *self
            .texture_to_binding_index
            .entry(*image_id)
            .or_insert_with(|| {
                let index = self.binding_index_to_textures.len() as u32;
                self.binding_index_to_textures.push(*image_id);
                index
            })
    }
}

/// Uploads the list of decals from [`RenderDecals::decals`] to the GPU.
fn upload_decals(
    render_decals: Res<RenderDecals>,
    mut decals_buffer: ResMut<DecalsBuffer>,
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
) {
    decals_buffer.clear();

    for &decal in &render_decals.decals {
        decals_buffer.push(decal);
    }

    // Make sure the buffer is non-empty.
    // Otherwise there won't be a buffer to bind.
    if decals_buffer.is_empty() {
        decals_buffer.push(Decal::default());
    }

    decals_buffer.write_buffer(&render_device, &render_queue);
}
