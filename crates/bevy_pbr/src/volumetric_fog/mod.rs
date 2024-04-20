//! Volumetric lighting.

use bevy_app::{App, Plugin};
use bevy_asset::{load_internal_asset, Handle};
use bevy_color::Color;
use bevy_core_pipeline::{
    core_3d::{
        graph::{Core3d, Node3d},
        prepare_core_3d_depth_textures, Camera3d,
    },
    fullscreen_vertex_shader::fullscreen_shader_vertex_state,
    prepass::{DeferredPrepass, DepthPrepass, MotionVectorPrepass, NormalPrepass},
};
use bevy_derive::{Deref, DerefMut};
use bevy_ecs::{
    component::Component,
    entity::Entity,
    query::{Has, QueryItem, With},
    schedule::IntoSystemConfigs as _,
    system::{lifetimeless::Read, Commands, Query, Res, ResMut, Resource},
    world::{FromWorld, World},
};
use bevy_math::Vec3;
use bevy_render::{
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
        SpecializedRenderPipelines, TextureFormat, TextureSampleType, TextureUsages,
    },
    renderer::{RenderContext, RenderDevice, RenderQueue},
    texture::BevyDefault,
    view::{ExtractedView, Msaa, ViewDepthTexture, ViewTarget, ViewUniformOffset},
    Extract, ExtractSchedule, Render, RenderApp, RenderSet,
};
use bevy_utils::prelude::default;

use crate::{
    graph::NodePbr, MeshPipelineViewLayoutKey, MeshPipelineViewLayouts, MeshViewBindGroup,
    ViewFogUniformOffset, ViewLightProbesUniformOffset, ViewLightsUniformOffset,
};

pub const VOLUMETRIC_FOG_HANDLE: Handle<Shader> = Handle::weak_from_u128(17400058287583986650);

pub struct VolumetricFogPlugin;

#[derive(Clone, Copy, Component, Debug)]
pub struct VolumetricLight {
    pub distance: f32,
}

#[derive(Clone, Copy, Component, Debug)]
pub struct VolumetricFogSettings {
    pub fog_color: Color,
    /// If you're using a [`crate::EnvironmentMapLight`], then, for best
    /// results, this should be a good approximation of the color of the
    /// environment map.
    pub ambient_color: Color,
    pub ambient_intensity: f32,
    pub step_count: i32,
    pub max_depth: f32,
    pub absorption: f32,
    pub scattering: f32,
    pub density: f32,
    pub scattering_asymmetry: f32,
    pub light_tint: Color,
    pub light_intensity: f32,
}

#[derive(Resource)]
pub struct VolumetricFogPipeline {
    mesh_view_layouts: MeshPipelineViewLayouts,
    volumetric_view_bind_group_layout_no_msaa: BindGroupLayout,
    volumetric_view_bind_group_layout_msaa: BindGroupLayout,
    color_sampler: Sampler,
}

#[derive(Component, Deref, DerefMut)]
pub struct ViewVolumetricFogPipeline(pub CachedRenderPipelineId);

#[derive(Default)]
pub struct VolumetricFogNode;

#[derive(PartialEq, Eq, Hash, Clone, Copy)]
pub struct VolumetricFogPipelineKey {
    mesh_pipeline_view_key: MeshPipelineViewLayoutKey,
    hdr: bool,
}

#[derive(ShaderType)]
pub struct VolumetricFogUniform {
    fog_color: Vec3,
    light_tint: Vec3,
    ambient_color: Vec3,
    ambient_intensity: f32,
    step_count: i32,
    max_depth: f32,
    absorption: f32,
    scattering: f32,
    density: f32,
    scattering_asymmetry: f32,
    light_intensity: f32,
}

#[derive(Component, Deref, DerefMut)]
pub struct ViewVolumetricFogUniformOffset(u32);

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
                (Node3d::EndMainPass, NodePbr::VolumetricFog, Node3d::Bloom),
            );
    }
}

impl FromWorld for VolumetricFogPipeline {
    fn from_world(world: &mut World) -> Self {
        let render_device = world.resource::<RenderDevice>();
        let mesh_view_layouts = world.resource::<MeshPipelineViewLayouts>();

        let base_bind_group_layout_entries = &*BindGroupLayoutEntries::sequential(
            ShaderStages::FRAGMENT,
            (
                uniform_buffer::<VolumetricFogUniform>(true),
                texture_2d(TextureSampleType::Float { filterable: true }),
                sampler(SamplerBindingType::Filtering),
            ),
        );

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

pub fn extract_volumetric_fog(
    mut commands: Commands,
    view_targets: Extract<Query<(Entity, &VolumetricFogSettings)>>,
    volumetric_lights: Extract<Query<(Entity, &VolumetricLight)>>,
) {
    if volumetric_lights.is_empty() {
        return;
    }

    for (view_target, volumetric_fog_settings) in view_targets.iter() {
        commands
            .get_or_spawn(view_target)
            .insert(*volumetric_fog_settings);
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
        Read<ViewVolumetricFogUniformOffset>,
        Read<MeshViewBindGroup>,
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
            view_volumetric_lighting_uniform_buffer_offset,
            view_bind_group,
        ): QueryItem<'w, Self::ViewQuery>,
        world: &'w World,
    ) -> Result<(), NodeRunError> {
        let pipeline_cache = world.resource::<PipelineCache>();
        let volumetric_lighting_pipeline = world.resource::<VolumetricFogPipeline>();
        let volumetric_lighting_uniform_buffer = world.resource::<VolumetricFogUniformBuffer>();
        let msaa = world.resource::<Msaa>();

        let (Some(pipeline), Some(volumetric_lighting_uniform_buffer_binding)) = (
            pipeline_cache.get_render_pipeline(**view_volumetric_lighting_pipeline),
            volumetric_lighting_uniform_buffer.binding(),
        ) else {
            return Ok(());
        };

        let postprocess = view_target.post_process_write();

        // TODO: Cache this?
        let volumetric_view_bind_group_layout = match *msaa {
            Msaa::Off => &volumetric_lighting_pipeline.volumetric_view_bind_group_layout_no_msaa,
            _ => &volumetric_lighting_pipeline.volumetric_view_bind_group_layout_msaa,
        };
        let volumetric_view_bind_group = render_context.render_device().create_bind_group(
            None,
            volumetric_view_bind_group_layout,
            &BindGroupEntries::sequential((
                volumetric_lighting_uniform_buffer_binding,
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

        render_pass.set_pipeline(pipeline);
        render_pass.set_bind_group(
            0,
            &view_bind_group.value,
            &[
                view_uniform_offset.offset,
                view_lights_offset.offset,
                view_fog_offset.offset,
                **view_light_probes_offset,
            ],
        );
        render_pass.set_bind_group(
            1,
            &volumetric_view_bind_group,
            &[**view_volumetric_lighting_uniform_buffer_offset],
        );
        render_pass.draw(0..3, 0..1);

        Ok(())
    }
}

impl SpecializedRenderPipeline for VolumetricFogPipeline {
    type Key = VolumetricFogPipelineKey;

    fn specialize(&self, key: Self::Key) -> RenderPipelineDescriptor {
        let mesh_view_layout = self
            .mesh_view_layouts
            .get_view_layout(key.mesh_pipeline_view_key);

        let mut shader_defs = vec!["SHADOW_FILTER_METHOD_HARDWARE_2X2".into()];

        let volumetric_view_bind_group_layout = if key
            .mesh_pipeline_view_key
            .contains(MeshPipelineViewLayoutKey::MULTISAMPLED)
        {
            shader_defs.push("MULTISAMPLED".into());
            self.volumetric_view_bind_group_layout_msaa.clone()
        } else {
            self.volumetric_view_bind_group_layout_no_msaa.clone()
        };

        RenderPipelineDescriptor {
            label: Some("volumetric lighting pipeline".into()),
            layout: vec![mesh_view_layout.clone(), volumetric_view_bind_group_layout],
            push_constant_ranges: vec![],
            vertex: fullscreen_shader_vertex_state(),
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
) {
    for (entity, view, normal_prepass, depth_prepass, motion_vector_prepass, deferred_prepass) in
        view_targets.iter()
    {
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

        let pipeline_id = pipelines.specialize(
            &pipeline_cache,
            &volumetric_lighting_pipeline,
            VolumetricFogPipelineKey {
                mesh_pipeline_view_key,
                hdr: view.hdr,
            },
        );

        commands
            .entity(entity)
            .insert(ViewVolumetricFogPipeline(pipeline_id));
    }
}

/// A system that converts [`VolumetricFogSettings`]
pub fn prepare_volumetric_fog_uniforms(
    mut commands: Commands,
    mut volumetric_lighting_uniform_buffer: ResMut<VolumetricFogUniformBuffer>,
    view_targets: Query<(Entity, &VolumetricFogSettings)>,
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
) {
    let Some(mut writer) = volumetric_lighting_uniform_buffer.get_writer(
        view_targets.iter().len(),
        &render_device,
        &render_queue,
    ) else {
        return;
    };

    for (entity, volumetric_fog_settings) in view_targets.iter() {
        let offset = writer.write(&VolumetricFogUniform {
            fog_color: Vec3::from_slice(
                &volumetric_fog_settings.fog_color.linear().to_f32_array()[0..3],
            ),
            light_tint: Vec3::from_slice(
                &volumetric_fog_settings.light_tint.linear().to_f32_array()[0..3],
            ),
            ambient_color: Vec3::from_slice(
                &volumetric_fog_settings
                    .ambient_color
                    .linear()
                    .to_f32_array()[0..3],
            ),
            ambient_intensity: volumetric_fog_settings.ambient_intensity,
            step_count: volumetric_fog_settings.step_count,
            max_depth: volumetric_fog_settings.max_depth,
            absorption: volumetric_fog_settings.absorption,
            scattering: volumetric_fog_settings.scattering,
            density: volumetric_fog_settings.density,
            scattering_asymmetry: volumetric_fog_settings.scattering_asymmetry,
            light_intensity: volumetric_fog_settings.light_intensity,
        });

        commands
            .entity(entity)
            .insert(ViewVolumetricFogUniformOffset(offset));
    }
}

/// A system that marks all view depth textures as readable in shaders.
///
/// The volumetric lighting pass needs to do this, and it doesn't happen by
/// default.
pub fn prepare_view_depth_textures_for_volumetric_fog(
    mut view_targets: Query<&mut Camera3d, With<VolumetricFogSettings>>,
) {
    for mut camera in view_targets.iter_mut() {
        camera.depth_texture_usages.0 |= TextureUsages::TEXTURE_BINDING.bits();
    }
}

impl Default for VolumetricFogSettings {
    fn default() -> Self {
        Self {
            step_count: 128,
            max_depth: 25.0,
            absorption: 0.3,
            scattering: 0.3,
            density: 0.1,
            scattering_asymmetry: 0.5,
            fog_color: Color::WHITE,
            // Matches `AmbientLight` defaults.
            ambient_color: Color::WHITE,
            ambient_intensity: 0.1,
            light_tint: Color::WHITE,
            light_intensity: 1.0,
        }
    }
}

impl Default for VolumetricLight {
    fn default() -> Self {
        Self { distance: 10000.0 }
    }
}
