use bevy_app::{App, Plugin};
use bevy_asset::{load_internal_asset, Handle};
use bevy_derive::{Deref, DerefMut};
use bevy_ecs::{
    prelude::{Entity, Res, ResMut},
    query::With,
    schedule::IntoSystemConfigs,
    system::{Commands, Query, Resource},
    world::{FromWorld, World},
};
use bevy_math::{uvec4, Mat4, UVec2};
use bevy_render::{
    camera::{
        CameraRenderGraph, ExtractedCamera, NormalizedRenderTarget, ReflectionPlaneKey, Viewport,
    },
    prelude::Camera,
    render_resource::TextureFormat,
    render_resource::{
        AddressMode, Extent3d, FilterMode, Sampler, SamplerDescriptor, Shader, ShaderType, Texture,
        TextureAspect, TextureDescriptor, TextureDimension, TextureUsages, TextureView,
        TextureViewDescriptor, TextureViewDimension, UniformBuffer,
    },
    renderer::{RenderDevice, RenderQueue},
    texture::BevyDefault,
    view::{ColorGrading, ExtractedView, RenderReflectionPlaneTextureViews, ViewTarget},
    Extract, ExtractSchedule, Render, RenderApp, RenderSet,
};
use bevy_transform::prelude::GlobalTransform;
use bevy_utils::{EntityHashMap, EntityHashSet, HashMap};

use crate::ReflectionPlane;

pub const REFLECTION_PLANES_SHADER_HANDLE: Handle<Shader> =
    Handle::weak_from_u128(166489733890667948920569244961095141211);

pub const MAX_REFLECTION_PLANES: usize = 256;

#[derive(Default)]
pub struct ReflectionPlanesPlugin;

/// Each reflection plane/view combination is associated with an entity that
/// represents the virtual camera for that reflection plane relative to that
/// view.
#[derive(Resource, Default, Deref, DerefMut)]
pub struct RenderReflectionPlaneViews(HashMap<ReflectionPlaneKey, Entity>);

#[derive(ShaderType)]
pub struct GpuReflectionPlanes {
    data: [GpuReflectionPlane; MAX_REFLECTION_PLANES],
}

#[derive(ShaderType, Default, Clone, Copy)]
struct GpuReflectionPlane {
    transform: Mat4,
    index: u32,
}

#[derive(Resource)]
pub struct RenderReflectionPlanes {
    pub buffer: UniformBuffer<GpuReflectionPlanes>,
    /// Maps from the camera entity to its texture array and view.
    pub textures: EntityHashMap<Entity, (Texture, TextureView)>,
    pub sampler: Sampler,
}

impl Plugin for ReflectionPlanesPlugin {
    fn build(&self, app: &mut App) {
        load_internal_asset!(
            app,
            REFLECTION_PLANES_SHADER_HANDLE,
            "reflection_planes.wgsl",
            Shader::from_wgsl
        );

        let Ok(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };

        render_app
            .init_resource::<RenderReflectionPlaneViews>()
            .add_systems(ExtractSchedule, extract_reflection_planes)
            .add_systems(
                Render,
                build_reflection_plane_textures.in_set(RenderSet::PrepareResources),
            );
    }

    fn finish(&self, app: &mut App) {
        let Ok(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };

        render_app.init_resource::<RenderReflectionPlanes>();
    }
}

pub fn extract_reflection_planes(
    mut commands: Commands,
    mut render_reflection_plane_views: ResMut<RenderReflectionPlaneViews>,
    cameras: Extract<
        Query<(
            Entity,
            &Camera,
            &CameraRenderGraph,
            &GlobalTransform,
            Option<&ColorGrading>,
        )>,
    >,
    reflection_planes: Extract<Query<Entity, With<ReflectionPlane>>>,
) {
    for reflection_plane_entity in reflection_planes.iter() {
        for (camera_entity, camera, camera_render_graph, camera_transform, color_grading) in
            cameras.iter()
        {
            let key = ReflectionPlaneKey {
                reflection_plane: reflection_plane_entity,
                camera: camera_entity,
            };

            let reflection_plane_view_entity = *render_reflection_plane_views
                .entry(key)
                .or_insert_with(|| commands.spawn(()).id());

            let viewport = Viewport {
                physical_position: UVec2::default(),
                physical_size: camera.physical_viewport_size().unwrap_or_else(|| {
                    camera
                        .physical_target_size()
                        .expect("Couldn't determine the size of the render target")
                }),
                ..Viewport::default()
            };

            commands
                .entity(reflection_plane_view_entity)
                .insert(ExtractedCamera {
                    target: None,
                    viewport: Some(viewport.clone()),
                    physical_viewport_size: camera.physical_viewport_size(),
                    physical_target_size: camera.physical_viewport_size(),
                    render_graph: (*camera_render_graph).clone(),
                    order: camera.order - 1,
                    output_mode: camera.output_mode,
                    msaa_writeback: false,
                    sorted_camera_index_for_target: 0,
                })
                .insert(ExtractedView {
                    projection: camera.projection_matrix(),
                    transform: *camera_transform,
                    view_projection: None,
                    hdr: camera.hdr,
                    viewport: uvec4(0, 0, viewport.physical_size.x, viewport.physical_size.y),
                    color_grading: color_grading.cloned().unwrap_or_default(),
                })
                .insert(key);
        }
    }
}

pub fn build_reflection_plane_textures(
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
    mut reflection_planes: Query<(
        Entity,
        &ReflectionPlaneKey,
        &mut ExtractedCamera,
        &ExtractedView,
    )>,
    mut render_reflection_planes: ResMut<RenderReflectionPlanes>,
    mut render_reflection_plane_texture_views: ResMut<RenderReflectionPlaneTextureViews>,
) {
    // FIXME: Don't recreate this every frame!

    let mut reflection_plane_entities = EntityHashSet::default();
    for (_, reflection_plane_key, _, _) in reflection_planes.iter() {
        reflection_plane_entities.insert(reflection_plane_key.reflection_plane);
    }

    render_reflection_planes.textures.clear();
    render_reflection_plane_texture_views.clear();

    // Recreate textures, and assign each plane an index.
    let mut reflection_plane_indices = EntityHashMap::default();
    for (_, reflection_plane_key, extracted_camera, extracted_view) in reflection_planes.iter() {
        if !render_reflection_planes
            .textures
            .contains_key(&reflection_plane_key.camera)
        {
            let physical_viewport_size = extracted_camera.physical_viewport_size.unwrap();
            let texture = render_device.create_texture(&TextureDescriptor {
                label: Some("reflection_plane_texture"),
                size: Extent3d {
                    width: physical_viewport_size.x,
                    height: physical_viewport_size.y,
                    depth_or_array_layers: reflection_plane_entities.len() as u32,
                },
                // TODO: Generate mips.
                mip_level_count: 1,
                sample_count: 1,
                dimension: TextureDimension::D2,
                format: if extracted_view.hdr {
                    ViewTarget::TEXTURE_FORMAT_HDR
                } else {
                    TextureFormat::bevy_default()
                },
                usage: TextureUsages::TEXTURE_BINDING | TextureUsages::RENDER_ATTACHMENT,
                view_formats: &[],
            });

            let texture_view = texture.create_view(&TextureViewDescriptor {
                label: Some("reflection_plane_texture_view"),
                format: Some(texture.format()),
                dimension: Some(TextureViewDimension::D2Array),
                aspect: TextureAspect::All,
                base_mip_level: 0,
                mip_level_count: Some(1),
                base_array_layer: 0,
                array_layer_count: Some(texture.depth_or_array_layers()),
            });

            render_reflection_planes
                .textures
                .insert(reflection_plane_key.camera, (texture, texture_view));
        }

        if !reflection_plane_indices.contains_key(&reflection_plane_key.reflection_plane) {
            let reflection_plane_index = reflection_plane_indices.len();
            reflection_plane_indices.insert(
                reflection_plane_key.reflection_plane,
                reflection_plane_index as u32,
            );
            render_reflection_planes.buffer.get_mut().data[reflection_plane_index] =
                GpuReflectionPlane {
                    transform: extracted_view.transform.compute_matrix(),
                    index: reflection_plane_index as u32,
                };
        }
    }

    // Recreate views.
    for (reflection_plane_view_entity, reflection_plane_key, mut extracted_camera, _) in
        reflection_planes.iter_mut()
    {
        let &(ref texture, _) = &render_reflection_planes.textures[&reflection_plane_key.camera];
        let texture_view = texture.create_view(&TextureViewDescriptor {
            label: Some("reflection_plane_texture_view"),
            format: Some(texture.format()),
            dimension: Some(TextureViewDimension::D2),
            aspect: TextureAspect::All,
            base_mip_level: 0,
            mip_level_count: Some(1),
            base_array_layer: reflection_plane_indices[&reflection_plane_key.reflection_plane],
            array_layer_count: Some(1),
        });

        render_reflection_plane_texture_views.insert(
            reflection_plane_view_entity,
            (texture_view.clone(), texture.format()),
        );

        extracted_camera.target = Some(NormalizedRenderTarget::ReflectionPlane(
            reflection_plane_view_entity,
        ));
    }

    // Finally, upload the uniform buffer.
    render_reflection_planes.buffer.write_buffer(&render_device, &render_queue);
}

impl FromWorld for RenderReflectionPlanes {
    fn from_world(world: &mut World) -> Self {
        let render_device = world.resource::<RenderDevice>();
        let sampler = render_device.create_sampler(&SamplerDescriptor {
            label: Some("reflection_planes_sampler"),
            address_mode_u: AddressMode::ClampToEdge,
            address_mode_v: AddressMode::ClampToEdge,
            address_mode_w: AddressMode::ClampToEdge,
            mag_filter: FilterMode::Linear,
            min_filter: FilterMode::Linear,
            mipmap_filter: FilterMode::Linear,
            ..SamplerDescriptor::default()
        });
        RenderReflectionPlanes {
            buffer: UniformBuffer::from_world(world),
            sampler,
            textures: EntityHashMap::default(),
        }
    }
}

impl Default for GpuReflectionPlanes {
    fn default() -> Self {
        Self {
            data: [GpuReflectionPlane::default(); MAX_REFLECTION_PLANES],
        }
    }
}
