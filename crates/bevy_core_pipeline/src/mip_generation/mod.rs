//! Downsampling of textures to produce mipmap levels.

use core::array;

use bevy_app::{App, Plugin};
use bevy_asset::{load_internal_asset, Handle};
use bevy_derive::{Deref, DerefMut};
use bevy_ecs::{
    component::Component,
    entity::Entity,
    query::{QueryItem, With},
    schedule::IntoSystemConfigs as _,
    system::{lifetimeless::Read, Commands, Query, Res, ResMut, Resource},
    world::{FromWorld, World},
};
use bevy_math::{uvec2, UVec2, UVec4};
use bevy_render::{
    render_graph::{NodeRunError, RenderGraphApp, RenderGraphContext, ViewNode, ViewNodeRunner},
    render_resource::{
        binding_types::{sampler, storage_buffer_read_only_sized, texture_storage_2d},
        BindGroup, BindGroupEntries, BindGroupLayout, BindGroupLayoutEntries,
        CachedComputePipelineId, ComputePassDescriptor, ComputePipeline, ComputePipelineDescriptor,
        Extent3d, IntoBinding, PipelineCache, PushConstantRange, Sampler, SamplerBindingType,
        SamplerDescriptor, Shader, ShaderStages, StorageTextureAccess, TextureAspect,
        TextureDescriptor, TextureDimension, TextureFormat, TextureUsages, TextureView,
        TextureViewDescriptor, TextureViewDimension,
    },
    renderer::{RenderContext, RenderDevice},
    texture::TextureCache,
    view::{ExtractedView, ViewDepthTexture},
    Render, RenderApp, RenderSet,
};

use crate::{
    core_3d::graph::{Core3d, Node3d},
    occlusion_culling::OcclusionCulling,
};

pub const DOWNSAMPLE_DEPTH_SHADER_HANDLE: Handle<Shader> =
    Handle::weak_from_u128(3876351454330663524);

pub const DEPTH_PYRAMID_MIP_COUNT: usize = 12;

pub struct MipGenerationPlugin;

impl Plugin for MipGenerationPlugin {
    fn build(&self, app: &mut App) {
        load_internal_asset!(
            app,
            DOWNSAMPLE_DEPTH_SHADER_HANDLE,
            "downsample_depth.wgsl",
            Shader::from_wgsl
        );

        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };

        render_app
            .add_render_graph_node::<ViewNodeRunner<DownsampleDepthNode>>(
                Core3d,
                Node3d::DownsampleDepth,
            )
            .add_render_graph_edges(
                Core3d,
                (
                    Node3d::EarlyPrepass,
                    Node3d::DownsampleDepth,
                    Node3d::DeferredPrepass,
                ),
            )
            .add_systems(
                Render,
                (
                    prepare_view_depth_pyramids,
                    prepare_downsample_depth_view_bind_groups,
                )
                    .chain()
                    .in_set(RenderSet::PrepareResources),
            )
            .add_systems(
                Render,
                prepare_downsample_depth_pipelines.in_set(RenderSet::PrepareResources),
            );
    }

    fn finish(&self, app: &mut App) {
        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };
        render_app
            .init_resource::<DownsampleDepthBindGroupLayout>()
            .init_resource::<DownsampleDepthPipelines>()
            .init_resource::<DepthPyramidDummyTexture>();
    }
}

#[derive(Default)]
pub struct DownsampleDepthNode;

impl ViewNode for DownsampleDepthNode {
    type ViewQuery = (
        Read<ViewDepthPyramid>,
        Read<ViewDownsampleDepthBindGroup>,
        Read<ViewDownsampleDepthPipelines>,
        Read<ViewDepthTexture>,
    );

    fn run<'w>(
        &self,
        graph: &mut RenderGraphContext,
        render_context: &mut RenderContext<'w>,
        (
            view_depth_pyramid,
            view_downsample_depth_bind_group,
            view_downsample_depth_pipelines,
            view_depth_texture,
        ): QueryItem<'w, Self::ViewQuery>,
        world: &'w World,
    ) -> Result<(), NodeRunError> {
        let view_size = uvec2(
            view_depth_texture.texture.width(),
            view_depth_texture.texture.height(),
        );
        view_depth_pyramid.downsample_depth(
            render_context,
            view_size,
            view_downsample_depth_bind_group,
            &view_downsample_depth_pipelines.first,
            &view_downsample_depth_pipelines.second,
        );
        Ok(())
    }
}

#[derive(Resource)]
pub struct DownsampleDepthBindGroupLayout {
    bind_group_layout: BindGroupLayout,
    sampler: Sampler,
}

impl FromWorld for DownsampleDepthBindGroupLayout {
    fn from_world(world: &mut World) -> Self {
        let render_device = world.resource::<RenderDevice>();

        Self {
            bind_group_layout: render_device.create_bind_group_layout(
                "downsample depth bind group layout",
                &BindGroupLayoutEntries::sequential(
                    ShaderStages::COMPUTE,
                    (
                        // TODO: this is probably wrong, it's specialized to meshlets
                        storage_buffer_read_only_sized(false, None),
                        texture_storage_2d(
                            TextureFormat::R32Float,
                            StorageTextureAccess::WriteOnly,
                        ),
                        texture_storage_2d(
                            TextureFormat::R32Float,
                            StorageTextureAccess::WriteOnly,
                        ),
                        texture_storage_2d(
                            TextureFormat::R32Float,
                            StorageTextureAccess::WriteOnly,
                        ),
                        texture_storage_2d(
                            TextureFormat::R32Float,
                            StorageTextureAccess::WriteOnly,
                        ),
                        texture_storage_2d(
                            TextureFormat::R32Float,
                            StorageTextureAccess::WriteOnly,
                        ),
                        texture_storage_2d(
                            TextureFormat::R32Float,
                            StorageTextureAccess::ReadWrite,
                        ),
                        texture_storage_2d(
                            TextureFormat::R32Float,
                            StorageTextureAccess::WriteOnly,
                        ),
                        texture_storage_2d(
                            TextureFormat::R32Float,
                            StorageTextureAccess::WriteOnly,
                        ),
                        texture_storage_2d(
                            TextureFormat::R32Float,
                            StorageTextureAccess::WriteOnly,
                        ),
                        texture_storage_2d(
                            TextureFormat::R32Float,
                            StorageTextureAccess::WriteOnly,
                        ),
                        texture_storage_2d(
                            TextureFormat::R32Float,
                            StorageTextureAccess::WriteOnly,
                        ),
                        texture_storage_2d(
                            TextureFormat::R32Float,
                            StorageTextureAccess::WriteOnly,
                        ),
                        sampler(SamplerBindingType::NonFiltering),
                    ),
                ),
            ),
            sampler: render_device.create_sampler(&SamplerDescriptor {
                label: Some("depth pyramid sampler"),
                ..SamplerDescriptor::default()
            }),
        }
    }
}

#[derive(Resource)]
pub struct DownsampleDepthPipelines {
    first: CachedComputePipelineId,
    second: CachedComputePipelineId,
}

impl FromWorld for DownsampleDepthPipelines {
    fn from_world(world: &mut World) -> Self {
        let downsample_depth_bind_group_layout = world
            .resource::<DownsampleDepthBindGroupLayout>()
            .bind_group_layout
            .clone();
        let pipeline_cache = world.resource_mut::<PipelineCache>();

        Self {
            first: pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
                label: Some("downsample depth first pipeline".into()),
                layout: vec![downsample_depth_bind_group_layout.clone()],
                push_constant_ranges: vec![PushConstantRange {
                    stages: ShaderStages::COMPUTE,
                    range: 0..8,
                }],
                shader: DOWNSAMPLE_DEPTH_SHADER_HANDLE,
                shader_defs: vec![],
                entry_point: "downsample_depth_first".into(),
                zero_initialize_workgroup_memory: false,
            }),
            second: pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
                label: Some("downsample depth second pipeline".into()),
                layout: vec![downsample_depth_bind_group_layout.clone()],
                push_constant_ranges: vec![PushConstantRange {
                    stages: ShaderStages::COMPUTE,
                    range: 0..8,
                }],
                shader: DOWNSAMPLE_DEPTH_SHADER_HANDLE,
                shader_defs: vec![],
                entry_point: "downsample_depth_second".into(),
                zero_initialize_workgroup_memory: false,
            }),
        }
    }
}

#[derive(Resource, Deref, DerefMut)]
pub struct DepthPyramidDummyTexture(TextureView);

impl FromWorld for DepthPyramidDummyTexture {
    fn from_world(world: &mut World) -> Self {
        let render_device = world.resource::<RenderDevice>();

        DepthPyramidDummyTexture(create_depth_pyramid_dummy_texture(
            render_device,
            "depth pyramid dummy texture",
            "depth pyramid dummy texture view",
        ))
    }
}

pub fn create_depth_pyramid_dummy_texture(
    render_device: &RenderDevice,
    texture_label: &'static str,
    texture_view_label: &'static str,
) -> TextureView {
    render_device
        .create_texture(&TextureDescriptor {
            label: Some(texture_label),
            size: Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::R32Float,
            usage: TextureUsages::STORAGE_BINDING,
            view_formats: &[],
        })
        .create_view(&TextureViewDescriptor {
            label: Some(texture_view_label),
            format: Some(TextureFormat::R32Float),
            dimension: Some(TextureViewDimension::D2),
            aspect: TextureAspect::All,
            base_mip_level: 0,
            mip_level_count: Some(1),
            base_array_layer: 0,
            array_layer_count: Some(1),
        })
}

#[derive(Component)]
pub struct ViewDepthPyramid {
    pub all_mips: TextureView,
    pub mips: [TextureView; DEPTH_PYRAMID_MIP_COUNT],
    pub mip_count: u32,
}

impl ViewDepthPyramid {
    pub fn new(
        render_device: &RenderDevice,
        texture_cache: &mut TextureCache,
        depth_pyramid_dummy_texture: &TextureView,
        viewport: UVec4,
        texture_label: &'static str,
        texture_view_label: &'static str,
    ) -> ViewDepthPyramid {
        let depth_pyramid_size = Extent3d {
            width: viewport.z.div_ceil(2),
            height: viewport.w.div_ceil(2),
            depth_or_array_layers: 1,
        };
        let depth_pyramid_mip_count = depth_pyramid_size.max_mips(TextureDimension::D2);
        let depth_pyramid = texture_cache.get(
            render_device,
            TextureDescriptor {
                label: Some(texture_label),
                size: depth_pyramid_size,
                mip_level_count: depth_pyramid_mip_count,
                sample_count: 1,
                dimension: TextureDimension::D2,
                format: TextureFormat::R32Float,
                usage: TextureUsages::STORAGE_BINDING | TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            },
        );
        let depth_pyramid_mips = array::from_fn(|i| {
            if (i as u32) < depth_pyramid_mip_count {
                depth_pyramid.texture.create_view(&TextureViewDescriptor {
                    label: Some(texture_view_label),
                    format: Some(TextureFormat::R32Float),
                    dimension: Some(TextureViewDimension::D2),
                    aspect: TextureAspect::All,
                    base_mip_level: i as u32,
                    mip_level_count: Some(1),
                    base_array_layer: 0,
                    array_layer_count: Some(1),
                })
            } else {
                (*depth_pyramid_dummy_texture).clone()
            }
        });
        let depth_pyramid_all_mips = depth_pyramid.default_view.clone();

        Self {
            all_mips: depth_pyramid_all_mips,
            mips: depth_pyramid_mips,
            mip_count: depth_pyramid_mip_count,
        }
    }

    pub fn create_bind_group<'a, R>(
        &'a self,
        render_device: &RenderDevice,
        label: &'static str,
        bind_group_layout: &BindGroupLayout,
        source_image: R,
        sampler: &'a Sampler,
    ) -> BindGroup
    where
        R: IntoBinding<'a>,
    {
        render_device.create_bind_group(
            label,
            bind_group_layout,
            &BindGroupEntries::sequential((
                source_image,
                &self.mips[0],
                &self.mips[1],
                &self.mips[2],
                &self.mips[3],
                &self.mips[4],
                &self.mips[5],
                &self.mips[6],
                &self.mips[7],
                &self.mips[8],
                &self.mips[9],
                &self.mips[10],
                &self.mips[11],
                sampler,
            )),
        )
    }

    pub fn downsample_depth(
        &self,
        render_context: &mut RenderContext,
        view_size: UVec2,
        downsample_depth_bind_group: &BindGroup,
        downsample_depth_first_pipeline: &ComputePipeline,
        downsample_depth_second_pipeline: &ComputePipeline,
    ) {
        let command_encoder = render_context.command_encoder();
        let mut downsample_pass = command_encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some("downsample depth"),
            timestamp_writes: None,
        });
        downsample_pass.set_pipeline(downsample_depth_first_pipeline);
        downsample_pass.set_push_constants(0, bytemuck::cast_slice(&[self.mip_count, view_size.x]));
        downsample_pass.set_bind_group(0, downsample_depth_bind_group, &[]);
        downsample_pass.dispatch_workgroups(view_size.x.div_ceil(64), view_size.y.div_ceil(64), 1);

        if self.mip_count >= 7 {
            downsample_pass.set_pipeline(downsample_depth_second_pipeline);
            downsample_pass.dispatch_workgroups(1, 1, 1);
        }
    }
}

fn prepare_view_depth_pyramids(
    mut commands: Commands,
    render_device: Res<RenderDevice>,
    mut texture_cache: ResMut<TextureCache>,
    depth_pyramid_dummy_texture: Res<DepthPyramidDummyTexture>,
    views: Query<(Entity, &ExtractedView), With<OcclusionCulling>>,
) {
    for (view_entity, view) in &views {
        commands.entity(view_entity).insert(ViewDepthPyramid::new(
            &render_device,
            &mut texture_cache,
            &depth_pyramid_dummy_texture,
            view.viewport,
            "view depth pyramid texture",
            "view depth pyramid texture view",
        ));
    }
}

#[derive(Component, Deref, DerefMut)]
pub struct ViewDownsampleDepthBindGroup(BindGroup);

fn prepare_downsample_depth_view_bind_groups(
    mut commands: Commands,
    render_device: Res<RenderDevice>,
    downsample_depth_bind_group_layout: Res<DownsampleDepthBindGroupLayout>,
    view_depth_textures: Query<(Entity, &ViewDepthPyramid, &ViewDepthTexture)>,
) {
    for (view_entity, view_depth_pyramid, view_depth_texture) in &view_depth_textures {
        commands
            .entity(view_entity)
            .insert(ViewDownsampleDepthBindGroup(
                view_depth_pyramid.create_bind_group(
                    &render_device,
                    "downsample depth bind group",
                    &downsample_depth_bind_group_layout.bind_group_layout,
                    view_depth_texture.view(),
                    &downsample_depth_bind_group_layout.sampler,
                ),
            ));
    }
}

#[derive(Component)]
pub struct ViewDownsampleDepthPipelines {
    first: ComputePipeline,
    second: ComputePipeline,
}

fn prepare_downsample_depth_pipelines(
    mut commands: Commands,
    views: Query<Entity, With<OcclusionCulling>>,
    downsample_depth_pipelines: Res<DownsampleDepthPipelines>,
    pipeline_cache: Res<PipelineCache>,
) {
    for view_entity in &views {
        let (Some(first), Some(second)) = (
            pipeline_cache.get_compute_pipeline(downsample_depth_pipelines.first),
            pipeline_cache.get_compute_pipeline(downsample_depth_pipelines.second),
        ) else {
            continue;
        };
        commands
            .entity(view_entity)
            .insert(ViewDownsampleDepthPipelines {
                first: (*first).clone(),
                second: (*second).clone(),
            });
    }
}
