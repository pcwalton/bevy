//! Volumetric fog and volumetric lighting, also known as light shafts or god
//! rays.
//!
//! This module implements a more physically-accurate, but slower, form of fog
//! than the [`crate::fog`] module does. Notably, this *volumetric fog* allows
//! for light beams from directional lights to shine through, creating what is
//! known as *light shafts* or *god rays*.
//!
//! To add volumetric fog to a scene, add [`VolumetricFogSettings`] to the
//! camera, and add [`VolumetricLight`] to directional lights that you wish to
//! be volumetric. [`VolumetricFogSettings`] feature numerous settings that
//! allow you to define the accuracy of the simulation, as well as the look of
//! the fog. Currently, only interaction with directional lights that have
//! shadow maps is supported. Note that the overhead of the effect scales
//! directly with the number of directional lights in use, so apply
//! [`VolumetricLight`] sparingly for the best results.
//!
//! The overall algorithm, which is implemented as a postprocessing effect, is a
//! combination of the techniques described in [Scratchapixel] and [this blog
//! post]. It uses raymarching in screen space, transformed into shadow map
//! space for sampling and combined with physically-based modeling of absorption
//! and scattering. Bevy employs the widely-used [Henyey-Greenstein phase
//! function] to model asymmetry; this essentially allows light shafts to fade
//! into and out of existence as the user views them.
//!
//! [Scratchapixel]: https://www.scratchapixel.com/lessons/3d-basic-rendering/volume-rendering-for-developers/intro-volume-rendering.html
//!
//! [this blog post]: https://www.alexandre-pestana.com/volumetric-lights/
//!
//! [Henyey-Greenstein phase function]: https://www.pbr-book.org/4ed/Volume_Scattering/Phase_Functions#TheHenyeyndashGreensteinPhaseFunction

use bevy_app::{App, Plugin};
use bevy_asset::{load_internal_asset, Assets, Handle};
use bevy_color::{Color, ColorToComponents};
use bevy_core_pipeline::{
    core_3d::{
        graph::{Core3d, Node3d},
        prepare_core_3d_depth_textures, Camera3d,
    },
    prepass::{DeferredPrepass, DepthPrepass, MotionVectorPrepass, NormalPrepass},
};
use bevy_derive::{Deref, DerefMut};
use bevy_ecs::{
    component::Component,
    entity::Entity,
    query::{Has, QueryItem, With},
    reflect::ReflectComponent,
    schedule::IntoSystemConfigs as _,
    system::{lifetimeless::Read, Commands, Local, Query, Res, ResMut, Resource},
    world::{FromWorld, World},
};
use bevy_math::{
    primitives::{Cuboid, Plane3d},
    Mat4, Vec2, Vec3, Vec3A, Vec4Swizzles as _,
};
use bevy_reflect::Reflect;
use bevy_render::{
    camera::{CameraProjection, Projection},
    mesh::{GpuBufferInfo, GpuMesh, Mesh, MeshVertexBufferLayoutRef, Meshable},
    render_asset::RenderAssets,
    render_graph::{NodeRunError, RenderGraphApp, RenderGraphContext, ViewNode, ViewNodeRunner},
    render_resource::{
        binding_types::{
            sampler, texture_2d, texture_depth_2d, texture_depth_2d_multisampled, uniform_buffer,
        },
        BindGroupEntries, BindGroupLayout, BindGroupLayoutEntries, CachedRenderPipelineId,
        ColorTargetState, ColorWrites, DynamicUniformBuffer, FilterMode, FragmentState,
        MultisampleState, Operations, PipelineCache, PrimitiveState, RenderPassColorAttachment,
        RenderPassDescriptor, RenderPipelineDescriptor, Sampler, SamplerBindingType,
        SamplerDescriptor, Shader, ShaderStages, ShaderType, SpecializedRenderPipeline,
        SpecializedRenderPipelines, TextureFormat, TextureSampleType, TextureUsages, VertexState,
    },
    renderer::{RenderContext, RenderDevice, RenderQueue},
    texture::BevyDefault,
    view::{ExtractedView, Msaa, ViewDepthTexture, ViewTarget, ViewUniformOffset},
    Extract, ExtractSchedule, Render, RenderApp, RenderSet,
};
use bevy_transform::components::GlobalTransform;
use bevy_utils::prelude::default;

use crate::{
    graph::NodePbr, MeshPipelineViewLayoutKey, MeshPipelineViewLayouts, MeshViewBindGroup,
    ViewFogUniformOffset, ViewLightProbesUniformOffset, ViewLightsUniformOffset,
    ViewScreenSpaceReflectionsUniformOffset,
};

/// The volumetric fog shader.
pub const VOLUMETRIC_FOG_HANDLE: Handle<Shader> = Handle::weak_from_u128(17400058287583986650);

pub const PLANE_MESH: Handle<Mesh> = Handle::weak_from_u128(435245126479971076);
pub const CUBE_MESH: Handle<Mesh> = Handle::weak_from_u128(5023959819001661507);

/// A plugin that implements volumetric fog.
pub struct VolumetricFogPlugin;

/// Add this component to a [`DirectionalLight`] with a shadow map
/// (`shadows_enabled: true`) to make volumetric fog interact with it.
///
/// This allows the light to generate light shafts/god rays.
#[derive(Clone, Copy, Component, Default, Debug, Reflect)]
#[reflect(Component)]
pub struct VolumetricLight;

/// When placed on a [`Camera3d`], enables volumetric fog and volumetric
/// lighting, also known as light shafts or god rays.
#[derive(Clone, Copy, Component, Debug, Reflect)]
#[reflect(Component)]
pub struct VolumetricFogSettings {
    /// Color of the ambient light.
    ///
    /// This is separate from Bevy's [`AmbientLight`](crate::light::AmbientLight) because an
    /// [`EnvironmentMapLight`](crate::environment_map::EnvironmentMapLight) is
    /// still considered an ambient light for the purposes of volumetric fog. If you're using a
    /// [`EnvironmentMapLight`](crate::environment_map::EnvironmentMapLight), for best results,
    /// this should be a good approximation of the average color of the environment map.
    ///
    /// Defaults to white.
    pub ambient_color: Color,

    /// The brightness of the ambient light.
    ///
    /// If there's no [`EnvironmentMapLight`](crate::environment_map::EnvironmentMapLight),
    /// set this to 0.
    ///
    /// Defaults to 0.1.
    pub ambient_intensity: f32,

    /// The number of raymarching steps to perform.
    ///
    /// Higher values produce higher-quality results with less banding, but
    /// reduce performance.
    ///
    /// The default value is 64.
    pub step_count: u32,

    /// The maximum distance that Bevy will trace a ray for, in world space.
    ///
    /// You can think of this as the radius of a sphere of fog surrounding the
    /// camera. It has to be capped to a finite value or else there would be an
    /// infinite amount of fog, which would result in completely-opaque areas
    /// where the skybox would be.
    ///
    /// The default value is 25.
    pub max_depth: f32,
}

#[derive(Clone, Component, Debug, Reflect)]
#[reflect(Component)]
pub struct FogVolume {
    /// The color of the fog.
    ///
    /// Note that the fog must be lit by a [`VolumetricLight`] or ambient light
    /// in order for this color to appear.
    ///
    /// Defaults to white.
    pub fog_color: Color,

    /// The absorption coefficient, which measures what fraction of light is
    /// absorbed by the fog at each step.
    ///
    /// Increasing this value makes the fog darker.
    ///
    /// The default value is 0.3.
    pub absorption: f32,

    /// The scattering coefficient, which measures the fraction of light that's
    /// scattered toward, and away from, the viewer.
    ///
    /// The default value is 0.3.
    pub scattering: f32,

    /// The density of fog, which measures how dark the fog is.
    ///
    /// The default value is 0.1.
    pub density: f32,

    /// Measures the fraction of light that's scattered *toward* the camera, as opposed to *away* from the camera.
    ///
    /// Increasing this value makes light shafts become more prominent when the
    /// camera is facing toward their source and less prominent when the camera
    /// is facing away. Essentially, a high value here means the light shafts
    /// will fade into view as the camera focuses on them and fade away when the
    /// camera is pointing away.
    ///
    /// The default value is 0.8.
    pub scattering_asymmetry: f32,

    /// Applies a nonphysical color to the light.
    ///
    /// This can be useful for artistic purposes but is nonphysical.
    ///
    /// The default value is white.
    pub light_tint: Color,

    /// Scales the light by a fixed fraction.
    ///
    /// This can be useful for artistic purposes but is nonphysical.
    ///
    /// The default value is 1.0, which results in no adjustment.
    pub light_intensity: f32,
}

/// The GPU pipeline for the volumetric fog postprocessing effect.
#[derive(Resource)]
pub struct VolumetricFogPipeline {
    /// A reference to the shared set of mesh pipeline view layouts.
    mesh_view_layouts: MeshPipelineViewLayouts,
    /// The view bind group when multisample antialiasing isn't in use.
    volumetric_view_bind_group_layout_no_msaa: BindGroupLayout,
    /// The view bind group when multisample antialiasing is in use.
    volumetric_view_bind_group_layout_msaa: BindGroupLayout,
    /// The sampler that we use to sample the postprocessing input.
    color_sampler: Sampler,
}

#[derive(Component, Deref, DerefMut)]
pub struct ViewVolumetricFogPipeline(pub CachedRenderPipelineId);

/// The node in the render graph, part of the postprocessing stack, that
/// implements volumetric fog.
#[derive(Default)]
pub struct VolumetricFogNode;

/// Identifies a single specialization of the volumetric fog shader.
#[derive(PartialEq, Eq, Hash, Clone)]
pub struct VolumetricFogPipelineKey {
    /// The layout of the view, which is needed for the raymarching.
    mesh_pipeline_view_key: MeshPipelineViewLayoutKey,
    vertex_buffer_layout: MeshVertexBufferLayoutRef,
    /// Whether the view has high dynamic range.
    hdr: bool,
}

/// The same as [`VolumetricFogSettings`] and [`FogVolume`], but formatted for
/// the GPU.
#[derive(ShaderType)]
pub struct VolumetricFogUniform {
    clip_from_local: Mat4,
    fog_color: Vec3,
    light_tint: Vec3,
    ambient_color: Vec3,
    ambient_intensity: f32,
    step_count: u32,
    max_depth: f32,
    absorption: f32,
    scattering: f32,
    density: f32,
    scattering_asymmetry: f32,
    light_intensity: f32,
}

/// Specifies the offset within the [`VolumetricFogUniformBuffer`] of the
/// [`VolumetricFogUniform`] for a specific view.
#[derive(Component, Deref, DerefMut)]
pub struct ViewVolumetricFog(Vec<ViewFogVolume>);

pub struct ViewFogVolume {
    uniform_buffer_offset: u32,
    exterior: bool,
}

/// The GPU buffer that stores the [`VolumetricFogUniform`] data.
#[derive(Resource, Default, Deref, DerefMut)]
pub struct VolumetricFogUniformBuffer(pub DynamicUniformBuffer<VolumetricFogUniform>);

impl Plugin for VolumetricFogPlugin {
    fn build(&self, app: &mut App) {
        load_internal_asset!(
            app,
            VOLUMETRIC_FOG_HANDLE,
            "volumetric_fog.wgsl",
            Shader::from_wgsl
        );

        let mut meshes = app.world_mut().resource_mut::<Assets<Mesh>>();
        meshes.insert(
            &PLANE_MESH,
            Plane3d::new(Vec3::NEG_Z, Vec2::ONE).mesh().into(),
        );
        meshes.insert(&CUBE_MESH, Cuboid::new(1.0, 1.0, 1.0).mesh().into());

        app.register_type::<VolumetricFogSettings>()
            .register_type::<VolumetricLight>();

        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };

        render_app
            .init_resource::<SpecializedRenderPipelines<VolumetricFogPipeline>>()
            .init_resource::<VolumetricFogUniformBuffer>()
            .add_systems(ExtractSchedule, extract_volumetric_fog)
            .add_systems(
                Render,
                (
                    prepare_volumetric_fog_pipelines.in_set(RenderSet::Prepare),
                    prepare_volumetric_fog_uniforms.in_set(RenderSet::Prepare),
                    prepare_view_depth_textures_for_volumetric_fog
                        .in_set(RenderSet::Prepare)
                        .before(prepare_core_3d_depth_textures),
                ),
            );
    }

    fn finish(&self, app: &mut App) {
        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };

        render_app
            .init_resource::<VolumetricFogPipeline>()
            .add_render_graph_node::<ViewNodeRunner<VolumetricFogNode>>(
                Core3d,
                NodePbr::VolumetricFog,
            )
            .add_render_graph_edges(
                Core3d,
                // Volumetric fog is a postprocessing effect. Run it after the
                // main pass but before bloom.
                (Node3d::EndMainPass, NodePbr::VolumetricFog, Node3d::Bloom),
            );
    }
}

impl Default for VolumetricFogSettings {
    fn default() -> Self {
        Self {
            step_count: 64,
            max_depth: 25.0,
            // Matches `AmbientLight` defaults.
            ambient_color: Color::WHITE,
            ambient_intensity: 0.1,
        }
    }
}

impl Default for FogVolume {
    fn default() -> Self {
        Self {
            absorption: 0.3,
            scattering: 0.3,
            density: 0.1,
            scattering_asymmetry: 0.5,
            fog_color: Color::WHITE,
            light_tint: Color::WHITE,
            light_intensity: 1.0,
        }
    }
}

impl FromWorld for VolumetricFogPipeline {
    fn from_world(world: &mut World) -> Self {
        let render_device = world.resource::<RenderDevice>();
        let mesh_view_layouts = world.resource::<MeshPipelineViewLayouts>();

        // Create the bind group layout entries common to both the MSAA and
        // non-MSAA bind group layouts.
        let base_bind_group_layout_entries = &*BindGroupLayoutEntries::sequential(
            ShaderStages::VERTEX_FRAGMENT,
            (
                // `volumetric_fog`
                uniform_buffer::<VolumetricFogUniform>(true),
                // `color_texture`
                texture_2d(TextureSampleType::Float { filterable: true }),
                // `color_sampler`
                sampler(SamplerBindingType::Filtering),
            ),
        );

        // Because `texture_depth_2d` and `texture_depth_2d_multisampled` are
        // different types, we need to make separate bind group layouts for
        // each.

        let mut bind_group_layout_entries_no_msaa = base_bind_group_layout_entries.to_vec();
        bind_group_layout_entries_no_msaa.extend_from_slice(&BindGroupLayoutEntries::with_indices(
            ShaderStages::FRAGMENT,
            ((3, texture_depth_2d()),),
        ));
        let volumetric_view_bind_group_layout_no_msaa = render_device.create_bind_group_layout(
            "volumetric lighting view bind group layout",
            &bind_group_layout_entries_no_msaa,
        );

        let mut bind_group_layout_entries_msaa = base_bind_group_layout_entries.to_vec();
        bind_group_layout_entries_msaa.extend_from_slice(&BindGroupLayoutEntries::with_indices(
            ShaderStages::FRAGMENT,
            ((3, texture_depth_2d_multisampled()),),
        ));
        let volumetric_view_bind_group_layout_msaa = render_device.create_bind_group_layout(
            "volumetric lighting view bind group layout (multisampled)",
            &bind_group_layout_entries_msaa,
        );

        let color_sampler = render_device.create_sampler(&SamplerDescriptor {
            label: Some("volumetric lighting color sampler"),
            mag_filter: FilterMode::Linear,
            min_filter: FilterMode::Linear,
            compare: None,
            ..default()
        });

        VolumetricFogPipeline {
            mesh_view_layouts: mesh_view_layouts.clone(),
            volumetric_view_bind_group_layout_no_msaa,
            volumetric_view_bind_group_layout_msaa,
            color_sampler,
        }
    }
}

/// Extracts [`VolumetricFogSettings`], [`FogVolume`], and [`VolumetricLight`]s
/// from the main world to the render world.
pub fn extract_volumetric_fog(
    mut commands: Commands,
    view_targets: Extract<Query<(Entity, &VolumetricFogSettings)>>,
    fog_volumes: Extract<Query<(Entity, &FogVolume, &GlobalTransform)>>,
    volumetric_lights: Extract<Query<(Entity, &VolumetricLight)>>,
) {
    if volumetric_lights.is_empty() {
        return;
    }

    for (entity, volumetric_fog_settings) in view_targets.iter() {
        commands
            .get_or_spawn(entity)
            .insert(*volumetric_fog_settings);
    }

    for (entity, fog_volume, fog_transform) in fog_volumes.iter() {
        commands
            .get_or_spawn(entity)
            .insert((*fog_volume).clone())
            .insert(*fog_transform);
    }

    for (entity, volumetric_light) in volumetric_lights.iter() {
        commands.get_or_spawn(entity).insert(*volumetric_light);
    }
}

impl ViewNode for VolumetricFogNode {
    type ViewQuery = (
        Read<ViewTarget>,
        Read<ViewDepthTexture>,
        Read<ViewVolumetricFogPipeline>,
        Read<ViewUniformOffset>,
        Read<ViewLightsUniformOffset>,
        Read<ViewFogUniformOffset>,
        Read<ViewLightProbesUniformOffset>,
        Read<ViewVolumetricFog>,
        Read<MeshViewBindGroup>,
        Read<ViewScreenSpaceReflectionsUniformOffset>,
    );

    fn run<'w>(
        &self,
        _: &mut RenderGraphContext,
        render_context: &mut RenderContext<'w>,
        (
            view_target,
            view_depth_texture,
            view_volumetric_lighting_pipeline,
            view_uniform_offset,
            view_lights_offset,
            view_fog_offset,
            view_light_probes_offset,
            view_fog_volumes,
            view_bind_group,
            view_ssr_offset,
        ): QueryItem<'w, Self::ViewQuery>,
        world: &'w World,
    ) -> Result<(), NodeRunError> {
        let pipeline_cache = world.resource::<PipelineCache>();
        let volumetric_lighting_pipeline = world.resource::<VolumetricFogPipeline>();
        let volumetric_lighting_uniform_buffers = world.resource::<VolumetricFogUniformBuffer>();
        let msaa = world.resource::<Msaa>();

        // Fetch the uniform buffer and binding.
        let (Some(pipeline), Some(volumetric_lighting_uniform_buffer_binding)) = (
            pipeline_cache.get_render_pipeline(**view_volumetric_lighting_pipeline),
            volumetric_lighting_uniform_buffers.binding(),
        ) else {
            return Ok(());
        };

        let gpu_meshes = world.resource::<RenderAssets<GpuMesh>>();

        for view_fog_volume in view_fog_volumes.iter() {
            let mesh_handle = if view_fog_volume.exterior {
                CUBE_MESH.clone()
            } else {
                PLANE_MESH.clone()
            };

            let Some(gpu_mesh) = gpu_meshes.get(&mesh_handle) else {
                println!("GPU mesh wasn't in `RenderAssets<GpuMesh>`!");
                return Ok(());
            };

            let postprocess = view_target.post_process_write();

            // Create the bind group for the view.
            //
            // TODO: Cache this.
            let volumetric_view_bind_group_layout = match *msaa {
                Msaa::Off => {
                    &volumetric_lighting_pipeline.volumetric_view_bind_group_layout_no_msaa
                }
                _ => &volumetric_lighting_pipeline.volumetric_view_bind_group_layout_msaa,
            };
            let volumetric_view_bind_group = render_context.render_device().create_bind_group(
                None,
                volumetric_view_bind_group_layout,
                &BindGroupEntries::sequential((
                    volumetric_lighting_uniform_buffer_binding.clone(),
                    postprocess.source,
                    &volumetric_lighting_pipeline.color_sampler,
                    view_depth_texture.view(),
                )),
            );

            let render_pass_descriptor = RenderPassDescriptor {
                label: Some("volumetric lighting pass"),
                color_attachments: &[Some(RenderPassColorAttachment {
                    view: postprocess.destination,
                    resolve_target: None,
                    ops: Operations::default(),
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            };

            let mut render_pass = render_context
                .command_encoder()
                .begin_render_pass(&render_pass_descriptor);

            render_pass.set_vertex_buffer(0, *gpu_mesh.vertex_buffer.slice(..));
            render_pass.set_pipeline(pipeline);
            render_pass.set_bind_group(
                0,
                &view_bind_group.value,
                &[
                    view_uniform_offset.offset,
                    view_lights_offset.offset,
                    view_fog_offset.offset,
                    **view_light_probes_offset,
                    **view_ssr_offset,
                ],
            );
            render_pass.set_bind_group(
                1,
                &volumetric_view_bind_group,
                &[view_fog_volume.uniform_buffer_offset],
            );

            // Draw elements or arrays, as appropriate.
            match &gpu_mesh.buffer_info {
                GpuBufferInfo::Indexed {
                    buffer,
                    index_format,
                    count,
                } => {
                    render_pass.set_index_buffer(*buffer.slice(..), *index_format);
                    render_pass.draw_indexed(0..*count, 0, 0..1);
                }
                GpuBufferInfo::NonIndexed => {
                    render_pass.draw(0..gpu_mesh.vertex_count, 0..1);
                }
            }
        }

        Ok(())
    }
}

impl SpecializedRenderPipeline for VolumetricFogPipeline {
    type Key = VolumetricFogPipelineKey;

    fn specialize(&self, key: Self::Key) -> RenderPipelineDescriptor {
        let mesh_view_layout = self
            .mesh_view_layouts
            .get_view_layout(key.mesh_pipeline_view_key);

        // We always use hardware 2x2 filtering for sampling the shadow map; the
        // more accurate versions with percentage-closer filtering aren't worth
        // the overhead.
        let mut shader_defs = vec!["SHADOW_FILTER_METHOD_HARDWARE_2X2".into()];

        // We need a separate layout for MSAA and non-MSAA.
        let volumetric_view_bind_group_layout = if key
            .mesh_pipeline_view_key
            .contains(MeshPipelineViewLayoutKey::MULTISAMPLED)
        {
            shader_defs.push("MULTISAMPLED".into());
            self.volumetric_view_bind_group_layout_msaa.clone()
        } else {
            self.volumetric_view_bind_group_layout_no_msaa.clone()
        };

        let vertex_format = key
            .vertex_buffer_layout
            .0
            .get_layout(&[Mesh::ATTRIBUTE_POSITION.at_shader_location(0)])
            .expect("Failed to get vertex layout for volumetric fog hull");

        RenderPipelineDescriptor {
            label: Some("volumetric lighting pipeline".into()),
            layout: vec![mesh_view_layout.clone(), volumetric_view_bind_group_layout],
            push_constant_ranges: vec![],
            vertex: VertexState {
                shader: VOLUMETRIC_FOG_HANDLE,
                shader_defs: shader_defs.clone(),
                entry_point: "vertex".into(),
                buffers: vec![vertex_format],
            },
            primitive: PrimitiveState::default(),
            depth_stencil: None,
            multisample: MultisampleState::default(),
            fragment: Some(FragmentState {
                shader: VOLUMETRIC_FOG_HANDLE,
                shader_defs,
                entry_point: "fragment".into(),
                targets: vec![Some(ColorTargetState {
                    format: if key.hdr {
                        ViewTarget::TEXTURE_FORMAT_HDR
                    } else {
                        TextureFormat::bevy_default()
                    },
                    blend: None,
                    write_mask: ColorWrites::ALL,
                })],
            }),
        }
    }
}

/// Specializes volumetric fog pipelines for all views with that effect enabled.
#[allow(clippy::too_many_arguments)]
pub fn prepare_volumetric_fog_pipelines(
    mut commands: Commands,
    pipeline_cache: Res<PipelineCache>,
    mut pipelines: ResMut<SpecializedRenderPipelines<VolumetricFogPipeline>>,
    volumetric_lighting_pipeline: Res<VolumetricFogPipeline>,
    view_targets: Query<
        (
            Entity,
            &ExtractedView,
            Has<NormalPrepass>,
            Has<DepthPrepass>,
            Has<MotionVectorPrepass>,
            Has<DeferredPrepass>,
        ),
        With<VolumetricFogSettings>,
    >,
    msaa: Res<Msaa>,
    meshes: Res<RenderAssets<GpuMesh>>,
) {
    let plane_mesh = meshes.get(&PLANE_MESH).expect("Plane mesh not found!");

    for (entity, view, normal_prepass, depth_prepass, motion_vector_prepass, deferred_prepass) in
        view_targets.iter()
    {
        // Create a mesh pipeline view layout key corresponding to the view.
        let mut mesh_pipeline_view_key = MeshPipelineViewLayoutKey::from(*msaa);
        mesh_pipeline_view_key.set(MeshPipelineViewLayoutKey::NORMAL_PREPASS, normal_prepass);
        mesh_pipeline_view_key.set(MeshPipelineViewLayoutKey::DEPTH_PREPASS, depth_prepass);
        mesh_pipeline_view_key.set(
            MeshPipelineViewLayoutKey::MOTION_VECTOR_PREPASS,
            motion_vector_prepass,
        );
        mesh_pipeline_view_key.set(
            MeshPipelineViewLayoutKey::DEFERRED_PREPASS,
            deferred_prepass,
        );

        // Specialize the pipeline.
        let pipeline_id = pipelines.specialize(
            &pipeline_cache,
            &volumetric_lighting_pipeline,
            VolumetricFogPipelineKey {
                mesh_pipeline_view_key,
                vertex_buffer_layout: plane_mesh.layout.clone(),
                hdr: view.hdr,
            },
        );

        commands
            .entity(entity)
            .insert(ViewVolumetricFogPipeline(pipeline_id));
    }
}

/// A system that converts [`VolumetricFogSettings`] into [`VolumetricFogUniform`]s.
pub fn prepare_volumetric_fog_uniforms(
    mut commands: Commands,
    mut volumetric_lighting_uniform_buffer: ResMut<VolumetricFogUniformBuffer>,
    view_targets: Query<(Entity, &ExtractedView, &VolumetricFogSettings)>,
    fog_volumes: Query<(Entity, &FogVolume, &GlobalTransform)>,
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
    mut local_from_world_matrices: Local<Vec<Mat4>>,
) {
    let Some(mut writer) = volumetric_lighting_uniform_buffer.get_writer(
        view_targets.iter().len(),
        &render_device,
        &render_queue,
    ) else {
        return;
    };

    // Do this up front to avoid O(n^2) matrix inversion.
    local_from_world_matrices.clear();
    for (_, _, fog_transform) in fog_volumes.iter() {
        local_from_world_matrices.push(fog_transform.compute_matrix().inverse());
    }

    for (view_entity, extracted_view, volumetric_fog_settings) in view_targets.iter() {
        let world_from_view = extracted_view.world_from_view.compute_matrix();

        let mut view_fog_volumes = vec![];

        for ((fog_volume_entity, fog_volume, fog_transform), local_from_world) in
            fog_volumes.iter().zip(local_from_world_matrices.iter())
        {
            let local_camera_pos = *local_from_world * world_from_view;

            let interior = Vec3A::from(local_camera_pos.col(3).xyz())
                .abs()
                .cmple(Vec3A::splat(0.5))
                .all();

            let hull_clip_from_local = if interior {
                Mat4::IDENTITY
            } else {
                let world_from_local = fog_transform.compute_matrix();
                let view_from_world = world_from_view.inverse();
                let clip_from_view = extracted_view.clip_from_view;
                clip_from_view * view_from_world * world_from_local
            };

            let uniform_buffer_offset = writer.write(&VolumetricFogUniform {
                clip_from_local: hull_clip_from_local,
                fog_color: fog_volume.fog_color.to_linear().to_vec3(),
                light_tint: fog_volume.light_tint.to_linear().to_vec3(),
                ambient_color: volumetric_fog_settings.ambient_color.to_linear().to_vec3(),
                ambient_intensity: volumetric_fog_settings.ambient_intensity,
                step_count: volumetric_fog_settings.step_count,
                max_depth: volumetric_fog_settings.max_depth,
                absorption: fog_volume.absorption,
                scattering: fog_volume.scattering,
                density: fog_volume.density,
                scattering_asymmetry: fog_volume.scattering_asymmetry,
                light_intensity: fog_volume.light_intensity,
            });

            view_fog_volumes.push(ViewFogVolume {
                uniform_buffer_offset,
                exterior: !interior,
            });
        }

        commands
            .entity(view_entity)
            .insert(ViewVolumetricFog(view_fog_volumes));
    }
}

/// A system that marks all view depth textures as readable in shaders.
///
/// The volumetric lighting pass needs to do this, and it doesn't happen by
/// default.
pub fn prepare_view_depth_textures_for_volumetric_fog(
    mut view_targets: Query<&mut Camera3d>,
    fog_volumes: Query<&VolumetricFogSettings>,
) {
    if fog_volumes.is_empty() {
        return;
    }

    for mut camera in view_targets.iter_mut() {
        camera.depth_texture_usages.0 |= TextureUsages::TEXTURE_BINDING.bits();
    }
}
