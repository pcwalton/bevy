//! Environment maps and reflection probes.
//!
//! An *environment map* consists of a pair of diffuse and specular cubemaps
//! that together reflect the static surrounding area of a region in space. When
//! available, the PBR shader uses these to apply diffuse light and calculate
//! specular reflections.
//!
//! Environment maps come in two flavors, depending on what other components the
//! entities they're attached to have:
//!
//! 1. If attached to a view, they represent the objects located a very far
//!    distance from the view, in a similar manner to a skybox. Essentially, these
//!    *view environment maps* represent a higher-quality replacement for
//!    [`crate::AmbientLight`] for outdoor scenes. The indirect light from such
//!    environment maps are added to every point of the scene, including
//!    interior enclosed areas.
//!
//! 2. If attached to a [`LightProbe`], environment maps represent the immediate
//!    surroundings of a specific location in the scene. These types of
//!    environment maps are known as *reflection probes*.
//!    [`ReflectionProbeBundle`] is available as a mechanism to conveniently add
//!    these to a scene.
//!
//! Typically, environment maps are static (i.e. "baked", calculated ahead of
//! time) and so only reflect fixed static geometry. The environment maps must
//! be pre-filtered into a pair of cubemaps, one for the diffuse component and
//! one for the specular component, according to the [split-sum approximation].
//! To pre-filter your environment map, you can use the [glTF IBL Sampler] or
//! its [artist-friendly UI]. The diffuse map uses the Lambertian distribution,
//! while the specular map uses the GGX distribution.
//!
//! The Khronos Group has [several pre-filtered environment maps] available for
//! you to use.
//!
//! Currently, reflection probes (i.e. environment maps attached to light
//! probes) use binding arrays (also known as bindless textures) and
//! consequently aren't supported on WebGL2 or WebGPU. Reflection probes are
//! also unsupported if GLSL is in use, due to `naga` limitations. Environment
//! maps attached to views are, however, supported on all platforms.
//!
//! [split-sum approximation]: https://cdn2.unrealengine.com/Resources/files/2013SiggraphPresentationsNotes-26915738.pdf
//!
//! [glTF IBL Sampler]: https://github.com/KhronosGroup/glTF-IBL-Sampler
//!
//! [artist-friendly UI]: https://github.com/pcwalton/gltf-ibl-sampler-egui
//!
//! [several pre-filtered environment maps]: https://github.com/KhronosGroup/glTF-Sample-Environments

use bevy_asset::{AssetId, Handle};
use bevy_ecs::{
    bundle::Bundle, component::Component, query::QueryItem, system::lifetimeless::Read,
};
use bevy_reflect::Reflect;
use bevy_render::{
    extract_instances::ExtractInstance,
    prelude::SpatialBundle,
    render_asset::RenderAssets,
    render_resource::{
        binding_types, BindGroupLayoutEntryBuilder, Sampler, SamplerBindingType, Shader,
        TextureSampleType, TextureView,
    },
    renderer::RenderDevice,
    texture::{FallbackImage, Image},
};
use bevy_utils::HashMap;
use std::num::NonZeroU32;
use std::ops::Deref;

use crate::{LightProbe, MAX_VIEW_REFLECTION_PROBES};

/// A handle to the environment map helper shader.
pub const ENVIRONMENT_MAP_SHADER_HANDLE: Handle<Shader> =
    Handle::weak_from_u128(154476556247605696);

/// A pair of cubemap textures that represent the surroundings of a specific
/// area in space.
///
/// See [`crate::environment_map`] for detailed information.
#[derive(Clone, Component, Reflect)]
pub struct EnvironmentMapLight {
    /// The blurry image that represents diffuse radiance surrounding a region.
    pub diffuse_map: Handle<Image>,
    /// The typically-sharper, mipmapped image that represents specular radiance
    /// surrounding a region.
    pub specular_map: Handle<Image>,
}

/// Like [`EnvironmentMapLight`], but contains asset IDs instead of handles.
///
/// This is for use in the render app.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct EnvironmentMapIds {
    /// The blurry image that represents diffuse radiance surrounding a region.
    pub(crate) diffuse: AssetId<Image>,
    /// The typically-sharper, mipmapped image that represents specular radiance
    /// surrounding a region.
    pub(crate) specular: AssetId<Image>,
}

/// A bundle that contains everything needed to make an entity a reflection
/// probe.
///
/// A reflection probe is a type of environment map that specifies the light
/// surrounding a region in space. For more information, see
/// [`crate::environment_map`].
#[derive(Bundle)]
pub struct ReflectionProbeBundle {
    /// Contains a transform that specifies the position of this reflection probe in space.
    pub spatial: SpatialBundle,
    /// Marks this environment map as a light probe.
    pub light_probe: LightProbe,
    /// The cubemaps that make up this environment map.
    pub environment_map: EnvironmentMapLight,
}

/// A component, part of the render world, that stores the mapping from
/// environment map ID to texture index in the diffuse and specular binding
/// arrays.
///
/// Cubemap textures belonging to environment maps are collected into binding
/// arrays, and the index of each texture is presented to the shader for runtime
/// lookup.
///
/// This component is attached to each view in the render world, because each
/// view may have a different set of cubemaps that it considers and therefore
/// cubemap indices are per-view.
#[derive(Component, Default)]
pub struct RenderViewEnvironmentMaps {
    /// The list of environment maps presented to the shader, in order.
    binding_index_to_cubemap: Vec<EnvironmentMapIds>,
    /// The reverse of `binding_index_to_cubemap`: a map from the environment
    /// map IDs to the index in `binding_index_to_cubemap`.
    cubemap_to_binding_index: HashMap<EnvironmentMapIds, u32>,
}

/// All the bind group entries necessary for PBR shaders to access the
/// environment maps exposed to a view.
pub(crate) enum RenderViewBindGroupEntries<'a> {
    /// The version used when binding arrays aren't available on the current
    /// platform.
    Single {
        /// The texture view of the view's diffuse cubemap.
        diffuse_texture_view: &'a TextureView,

        /// The texture view of the view's specular cubemap.
        specular_texture_view: &'a TextureView,

        /// The sampler used to sample elements of both `diffuse_texture_views` and
        /// `specular_texture_views`.
        sampler: &'a Sampler,
    },

    /// The version used when binding arrays aren't available on the current
    /// platform.
    Multiple {
        /// A texture view of each diffuse cubemap, in the same order that they are
        /// supplied to the view (i.e. in the same order as
        /// `binding_index_to_cubemap` in [`RenderViewEnvironmentMaps`]).
        ///
        /// This is a vector of `wgpu::TextureView`s. But we don't want to import
        /// `wgpu` in this crate, so we refer to it indirectly like this.
        diffuse_texture_views: Vec<&'a <TextureView as Deref>::Target>,

        /// As above, but for specular cubemaps.
        specular_texture_views: Vec<&'a <TextureView as Deref>::Target>,

        /// The sampler used to sample elements of both `diffuse_texture_views` and
        /// `specular_texture_views`.
        sampler: &'a Sampler,
    },
}

pub(crate) trait RenderDeviceExt {
    fn binding_arrays_are_available(&self) -> bool;
}

impl ExtractInstance for EnvironmentMapIds {
    type Data = Read<EnvironmentMapLight>;

    type Filter = ();

    fn extract(item: QueryItem<'_, Self::Data>) -> Option<Self> {
        Some(EnvironmentMapIds {
            diffuse: item.diffuse_map.id(),
            specular: item.specular_map.id(),
        })
    }
}

impl RenderViewEnvironmentMaps {
    pub(crate) fn new() -> Self {
        Self::default()
    }
}

impl RenderViewEnvironmentMaps {
    /// Whether there are no environment maps associated with the view.
    pub(crate) fn is_empty(&self) -> bool {
        self.binding_index_to_cubemap.is_empty()
    }

    /// Adds a cubemap to the list of bindings, if it wasn't there already, and
    /// returns its index within that list.
    pub(crate) fn get_or_insert_cubemap(&mut self, cubemap_id: &EnvironmentMapIds) -> u32 {
        *self
            .cubemap_to_binding_index
            .entry(*cubemap_id)
            .or_insert_with(|| {
                let index = self.binding_index_to_cubemap.len() as u32;
                self.binding_index_to_cubemap.push(*cubemap_id);
                index
            })
    }
}

/// Returns the bind group layout entries for the environment map diffuse and
/// specular binding arrays respectively, in addition to the sampler.
pub(crate) fn get_bind_group_layout_entries(
    render_device: &RenderDevice,
) -> [BindGroupLayoutEntryBuilder; 3] {
    let mut texture_cube_binding =
        binding_types::texture_cube(TextureSampleType::Float { filterable: true });
    if render_device.binding_arrays_are_available() {
        texture_cube_binding =
            texture_cube_binding.count(NonZeroU32::new(MAX_VIEW_REFLECTION_PROBES as _).unwrap());
    }

    [
        texture_cube_binding,
        texture_cube_binding,
        binding_types::sampler(SamplerBindingType::Filtering),
    ]
}

impl<'a> RenderViewBindGroupEntries<'a> {
    /// Looks up and returns the bindings for the environment map diffuse and
    /// specular binding arrays respectively, as well as the sampler.
    pub(crate) fn get(
        render_view_environment_maps: Option<&RenderViewEnvironmentMaps>,
        images: &'a RenderAssets<Image>,
        fallback_image: &'a FallbackImage,
        render_device: &RenderDevice,
    ) -> RenderViewBindGroupEntries<'a> {
        if render_device.binding_arrays_are_available() {
            let mut diffuse_texture_views = vec![];
            let mut specular_texture_views = vec![];
            let mut sampler = None;

            if let Some(environment_maps) = render_view_environment_maps {
                for &cubemap_id in &environment_maps.binding_index_to_cubemap {
                    add_texture_view(
                        &mut diffuse_texture_views,
                        &mut sampler,
                        cubemap_id.diffuse,
                        images,
                        fallback_image,
                    );
                    add_texture_view(
                        &mut specular_texture_views,
                        &mut sampler,
                        cubemap_id.specular,
                        images,
                        fallback_image,
                    );
                }
            }

            // Pad out the bindings to the size of the binding array using fallback
            // textures. This is necessary on D3D12 and Metal.
            diffuse_texture_views.resize(
                MAX_VIEW_REFLECTION_PROBES,
                &*fallback_image.cube.texture_view,
            );
            specular_texture_views.resize(
                MAX_VIEW_REFLECTION_PROBES,
                &*fallback_image.cube.texture_view,
            );

            return RenderViewBindGroupEntries::Multiple {
                diffuse_texture_views,
                specular_texture_views,
                sampler: sampler.unwrap_or(&fallback_image.cube.sampler),
            };
        }

        if let Some(environment_maps) = render_view_environment_maps {
            if let Some(cubemap) = environment_maps.binding_index_to_cubemap.get(0) {
                if let (Some(diffuse_image), Some(specular_image)) =
                    (images.get(cubemap.diffuse), images.get(cubemap.specular))
                {
                    return RenderViewBindGroupEntries::Single {
                        diffuse_texture_view: &diffuse_image.texture_view,
                        specular_texture_view: &specular_image.texture_view,
                        sampler: &diffuse_image.sampler,
                    };
                }
            }
        }

        RenderViewBindGroupEntries::Single {
            diffuse_texture_view: &fallback_image.cube.texture_view,
            specular_texture_view: &fallback_image.cube.texture_view,
            sampler: &fallback_image.cube.sampler,
        }
    }
}

/// Adds a diffuse or specular texture view to the `texture_views` list, and
/// populates `sampler` if this is the first such view.
fn add_texture_view<'a>(
    texture_views: &mut Vec<&'a <TextureView as Deref>::Target>,
    sampler: &mut Option<&'a Sampler>,
    image_id: AssetId<Image>,
    images: &'a RenderAssets<Image>,
    fallback_image: &'a FallbackImage,
) {
    match images.get(image_id) {
        None => {
            // Use the fallback image if the cubemap isn't loaded yet.
            texture_views.push(&*fallback_image.cube.texture_view);
        }
        Some(image) => {
            // If this is the first texture view, populate `sampler`.
            if sampler.is_none() {
                *sampler = Some(&image.sampler);
            }

            texture_views.push(&*image.texture_view);
        }
    }
}

impl RenderDeviceExt for RenderDevice {
    fn binding_arrays_are_available(&self) -> bool {
        use bevy_render::settings::WgpuFeatures;

        !cfg!(feature = "shader_format_glsl")
            && self
                .features()
                .contains(WgpuFeatures::TEXTURE_BINDING_ARRAY)
    }
}
