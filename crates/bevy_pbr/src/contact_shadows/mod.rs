//! Contact shadows, also known as screen space shadows.

use bevy_app::{App, Plugin};
use bevy_asset::{load_internal_asset, Handle};
use bevy_core_pipeline::{
    core_3d::{
        graph::{Core3d, Node3d},
        prepare_core_3d_depth_textures, Camera3d,
    },
    fullscreen_vertex_shader,
    prepass::DepthPrepass,
};
use bevy_derive::{Deref, DerefMut};
use bevy_ecs::{
    component::Component,
    entity::Entity,
    query::{QueryItem, With},
    reflect::ReflectComponent,
    schedule::IntoSystemConfigs as _,
    system::{lifetimeless::Read, Commands, Query, Res, ResMut, Resource},
    world::{FromWorld, World},
};
use bevy_reflect::{std_traits::ReflectDefault, Reflect};
use bevy_render::{
    camera::ExtractedCamera,
    extract_component::{ExtractComponent, ExtractComponentPlugin},
    render_graph::{NodeRunError, RenderGraphApp, RenderGraphContext, ViewNode, ViewNodeRunner},
    render_resource::{
        binding_types, BindGroupEntries, BindGroupLayout, BindGroupLayoutEntries, BufferUsages,
        CachedRenderPipelineId, CompareFunction, DepthStencilState, Extent3d, FilterMode,
        FragmentState, LoadOp, MultisampleState, Operations, PipelineCache, RawBufferVec,
        RenderPassDepthStencilAttachment, RenderPassDescriptor, RenderPipelineDescriptor, Sampler,
        SamplerDescriptor, Shader, ShaderStages, ShaderType, SpecializedRenderPipeline,
        SpecializedRenderPipelines, StoreOp, TextureDescriptor, TextureDimension, TextureFormat,
        TextureUsages,
    },
    renderer::{RenderContext, RenderDevice, RenderQueue},
    texture::{CachedTexture, TextureCache},
    view::{Msaa, ViewDepthTexture},
    Render, RenderApp, RenderSet,
};
use bevy_utils::prelude::default;
use bytemuck::{Pod, Zeroable};

use crate::graph::NodePbr;

const RESOLVE_MULTISAMPLE_DEPTH_BUFFER_SHADER_HANDLE: Handle<Shader> =
    Handle::weak_from_u128(15034918822512644580);

pub struct ContactShadowsPlugin;

#[derive(Component, Deref, DerefMut)]
pub struct ViewResolvedMultisampleDepthTexture(CachedTexture);

#[derive(Resource)]
pub struct ContactShadowsSamplers {
    pub linear: Sampler,
    pub nearest: Sampler,
}

#[derive(Resource)]
pub struct ResolveMultisampleDepthBufferPipeline {
    bind_group_layout: BindGroupLayout,
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct ResolveMultisampleDepthBufferPipelineKey;

#[derive(Component, Deref, DerefMut)]
pub struct ViewResolveMultisampleDepthBufferPipeline(CachedRenderPipelineId);

#[derive(Default)]
pub struct ResolveMultisampleDepthBufferNode;

#[derive(Clone, Copy, Debug, Component, Reflect, ExtractComponent)]
#[reflect(Component, Default)]
pub struct ContactShadowsSettings {
    pub max_distance: f32,
    pub thickness: f32,
    pub linear_steps: u32,
    pub linear_march_exponent: f32,
    pub bisection_steps: u32,
    pub use_secant: bool,
}

#[derive(Clone, Copy, Component, Default, ShaderType, Pod, Zeroable)]
#[repr(C)]
pub struct ContactShadowsUniform {
    max_distance: f32,
    thickness: f32,
    linear_steps: u32,
    linear_march_exponent: f32,
    bisection_steps: u32,
    use_secant: u32,
}

#[derive(Resource, Deref, DerefMut)]
pub struct ContactShadowsUniformBuffer(RawBufferVec<ContactShadowsUniform>);

#[derive(Component, Deref, DerefMut)]
pub struct ViewContactShadowsUniformOffset(u32);

impl Plugin for ContactShadowsPlugin {
    fn build(&self, app: &mut App) {
        load_internal_asset!(
            app,
            RESOLVE_MULTISAMPLE_DEPTH_BUFFER_SHADER_HANDLE,
            "resolve_multisample_depth.wgsl",
            Shader::from_wgsl
        );

        app.register_type::<ContactShadowsSettings>()
            .add_plugins(ExtractComponentPlugin::<ContactShadowsSettings>::default());

        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };

        render_app
            .init_resource::<ContactShadowsUniformBuffer>()
            .add_systems(
                Render,
                (
                    prepare_multisample_depth_resolve_pipelines.in_set(RenderSet::Prepare),
                    prepare_view_depth_textures_for_multisample_resolution
                        .in_set(RenderSet::Prepare)
                        .before(prepare_core_3d_depth_textures),
                    prepare_view_contact_shadows_textures.in_set(RenderSet::PrepareResources),
                    prepare_contact_shadows_uniforms.in_set(RenderSet::PrepareResources),
                ),
            )
            .add_render_graph_node::<ViewNodeRunner<ResolveMultisampleDepthBufferNode>>(
                Core3d,
                NodePbr::ContactShadowsInit,
            );
    }

    fn finish(&self, app: &mut App) {
        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };

        render_app
            .init_resource::<ContactShadowsSamplers>()
            .init_resource::<ResolveMultisampleDepthBufferPipeline>()
            .init_resource::<SpecializedRenderPipelines<ResolveMultisampleDepthBufferPipeline>>()
            .add_render_graph_edges(
                Core3d,
                (
                    Node3d::EndPrepasses,
                    NodePbr::ContactShadowsInit,
                    Node3d::StartMainPass,
                ),
            );
    }
}

impl FromWorld for ContactShadowsSamplers {
    fn from_world(world: &mut World) -> Self {
        let render_device = world.resource::<RenderDevice>();
        ContactShadowsSamplers {
            linear: render_device.create_sampler(&SamplerDescriptor {
                label: Some("contact shadows linear sampler"),
                mag_filter: FilterMode::Linear,
                min_filter: FilterMode::Linear,
                mipmap_filter: FilterMode::Linear,
                ..default()
            }),
            nearest: render_device.create_sampler(&SamplerDescriptor {
                label: Some("contact shadows nearest-neighbor sampler"),
                mag_filter: FilterMode::Nearest,
                min_filter: FilterMode::Nearest,
                mipmap_filter: FilterMode::Nearest,
                ..default()
            }),
        }
    }
}

impl FromWorld for ResolveMultisampleDepthBufferPipeline {
    fn from_world(world: &mut World) -> Self {
        let render_device = world.resource::<RenderDevice>();

        let bind_group_layout = render_device.create_bind_group_layout(
            "resolve multisample depth buffer bind group layout",
            &BindGroupLayoutEntries::single(
                ShaderStages::FRAGMENT,
                binding_types::texture_depth_2d_multisampled(),
            ),
        );

        Self { bind_group_layout }
    }
}

pub fn prepare_view_contact_shadows_textures(
    mut commands: Commands,
    render_device: ResMut<RenderDevice>,
    mut texture_cache: ResMut<TextureCache>,
    views: Query<(Entity, &ExtractedCamera), With<DepthPrepass>>,
) {
    for (view_entity, camera) in views.iter() {
        let Some(physical_target_size) = camera.physical_target_size else {
            continue;
        };

        let resolved_multisample_depth_texture = texture_cache.get(
            &render_device,
            TextureDescriptor {
                label: Some("resolved multisample depth texture"),
                size: Extent3d {
                    width: physical_target_size.x,
                    height: physical_target_size.y,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: TextureDimension::D2,
                format: TextureFormat::Depth32Float,
                usage: TextureUsages::RENDER_ATTACHMENT | TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            },
        );

        commands
            .entity(view_entity)
            .insert(ViewResolvedMultisampleDepthTexture(
                resolved_multisample_depth_texture,
            ));
    }
}

impl ViewNode for ResolveMultisampleDepthBufferNode {
    type ViewQuery = (
        Option<Read<ViewResolveMultisampleDepthBufferPipeline>>,
        Option<Read<ViewResolvedMultisampleDepthTexture>>,
        Read<ViewDepthTexture>,
    );

    fn run<'w>(
        &self,
        _: &mut RenderGraphContext,
        render_context: &mut RenderContext<'w>,
        (
            maybe_view_resolve_multisample_depth_buffer_pipeline,
            maybe_view_resolved_multisample_depth_textures,
            view_depth_texture,
        ): QueryItem<'w, Self::ViewQuery>,
        world: &'w World,
    ) -> Result<(), NodeRunError> {
        let (
            Some(view_resolve_multisample_depth_buffer_pipeline),
            Some(view_resolved_multisample_depth_texture),
        ) = (
            maybe_view_resolve_multisample_depth_buffer_pipeline,
            maybe_view_resolved_multisample_depth_textures,
        )
        else {
            println!("*** missing components");
            return Ok(());
        };

        let pipeline_cache = world.resource::<PipelineCache>();
        let Some(resolve_multisample_depth_buffer_render_pipeline) =
            pipeline_cache.get_render_pipeline(**view_resolve_multisample_depth_buffer_pipeline)
        else {
            println!("*** no render pipeline");
            return Ok(());
        };

        let resolve_multisample_depth_buffer_pipeline =
            world.resource::<ResolveMultisampleDepthBufferPipeline>();
        let resolve_multisample_depth_buffer_bind_group =
            render_context.render_device().create_bind_group(
                "resolve multisample depth buffer",
                &resolve_multisample_depth_buffer_pipeline.bind_group_layout,
                &BindGroupEntries::single(view_depth_texture.view()),
            );

        let mut render_pass = render_context.begin_tracked_render_pass(RenderPassDescriptor {
            label: Some("resolve multisample depth buffer pass"),
            color_attachments: &[],
            depth_stencil_attachment: Some(RenderPassDepthStencilAttachment {
                view: &view_resolved_multisample_depth_texture.default_view,
                depth_ops: Some(Operations {
                    load: LoadOp::Clear(0.0),
                    store: StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            timestamp_writes: None,
            occlusion_query_set: None,
        });

        render_pass.set_render_pipeline(resolve_multisample_depth_buffer_render_pipeline);
        render_pass.set_bind_group(0, &resolve_multisample_depth_buffer_bind_group, &[]);
        render_pass.draw(0..3, 0..1);

        Ok(())
    }
}

impl Default for ContactShadowsSettings {
    fn default() -> Self {
        Self {
            max_distance: 0.3,
            thickness: 0.5,
            linear_steps: 4,
            linear_march_exponent: 1.0,
            bisection_steps: 0,
            use_secant: false,
        }
    }
}

impl Default for ContactShadowsUniformBuffer {
    fn default() -> Self {
        ContactShadowsUniformBuffer(RawBufferVec::new(BufferUsages::UNIFORM))
    }
}

pub fn prepare_contact_shadows_uniforms(
    mut commands: Commands,
    mut contact_shadows_uniform_buffer: ResMut<ContactShadowsUniformBuffer>,
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
    views: Query<(Entity, Option<&ContactShadowsSettings>)>,
) {
    contact_shadows_uniform_buffer.clear();

    for (view_entity, maybe_contact_shadows_settings) in views.iter() {
        let uniform_buffer_offset = match maybe_contact_shadows_settings {
            None => 0,
            Some(contact_shadows_settings) => {
                contact_shadows_uniform_buffer.push((*contact_shadows_settings).into())
            }
        };
        commands
            .entity(view_entity)
            .insert(ViewContactShadowsUniformOffset(
                uniform_buffer_offset as u32,
            ));
    }

    // We need to make sure that the buffer contains *something*; otherwise, no
    // buffer will be uploaded.
    if contact_shadows_uniform_buffer.is_empty() {
        contact_shadows_uniform_buffer.push(ContactShadowsUniform::default());
    }

    contact_shadows_uniform_buffer.write_buffer(&render_device, &render_queue);
}

impl From<ContactShadowsSettings> for ContactShadowsUniform {
    fn from(contact_shadows_settings: ContactShadowsSettings) -> Self {
        ContactShadowsUniform {
            max_distance: contact_shadows_settings.max_distance,
            thickness: contact_shadows_settings.thickness,
            linear_steps: contact_shadows_settings.linear_steps,
            linear_march_exponent: contact_shadows_settings.linear_march_exponent,
            bisection_steps: contact_shadows_settings.bisection_steps,
            use_secant: contact_shadows_settings.use_secant as _,
        }
    }
}

pub fn prepare_multisample_depth_resolve_pipelines(
    mut commands: Commands,
    pipeline_cache: Res<PipelineCache>,
    mut pipelines: ResMut<SpecializedRenderPipelines<ResolveMultisampleDepthBufferPipeline>>,
    multisample_depth_buffer_resolve_pipeline: Res<ResolveMultisampleDepthBufferPipeline>,
    msaa: Res<Msaa>,
    views: Query<Entity, With<ContactShadowsSettings>>,
) {
    if *msaa == Msaa::Off {
        return;
    }

    for entity in views.iter() {
        let pipeline_id = pipelines.specialize(
            &pipeline_cache,
            &multisample_depth_buffer_resolve_pipeline,
            ResolveMultisampleDepthBufferPipelineKey,
        );

        commands
            .entity(entity)
            .insert(ViewResolveMultisampleDepthBufferPipeline(pipeline_id));
    }
}

impl SpecializedRenderPipeline for ResolveMultisampleDepthBufferPipeline {
    type Key = ResolveMultisampleDepthBufferPipelineKey;

    fn specialize(&self, _: Self::Key) -> RenderPipelineDescriptor {
        RenderPipelineDescriptor {
            label: Some("resolve multisample depth buffer pipeline".into()),
            layout: vec![self.bind_group_layout.clone()],
            push_constant_ranges: vec![],
            vertex: fullscreen_vertex_shader::fullscreen_shader_vertex_state(),
            primitive: default(),
            depth_stencil: Some(DepthStencilState {
                format: TextureFormat::Depth32Float,
                depth_write_enabled: true,
                depth_compare: CompareFunction::Always,
                stencil: default(),
                bias: default(),
            }),
            multisample: MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            fragment: Some(FragmentState {
                shader: RESOLVE_MULTISAMPLE_DEPTH_BUFFER_SHADER_HANDLE,
                shader_defs: vec![],
                entry_point: "fragment".into(),
                targets: vec![],
            }),
        }
    }
}

/// A system that marks all view depth textures with contact shadows enabled as
/// readable in shaders.
pub fn prepare_view_depth_textures_for_multisample_resolution(
    mut view_targets: Query<&mut Camera3d, With<ContactShadowsSettings>>,
) {
    for mut camera in view_targets.iter_mut() {
        camera.depth_texture_usages.0 |= TextureUsages::TEXTURE_BINDING.bits();
    }
}
