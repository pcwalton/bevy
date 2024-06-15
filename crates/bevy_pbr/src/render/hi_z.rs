//! Hierarchical Z buffer generation.

use std::array;

use bevy_app::{App, Plugin};
use bevy_asset::{load_internal_asset, Handle};
use bevy_core_pipeline::core_3d::graph::{Core3d, Node3d};
use bevy_derive::{Deref, DerefMut};
use bevy_ecs::{
    component::Component,
    entity::Entity,
    query::{AnyOf, QueryItem},
    schedule::IntoSystemConfigs as _,
    system::{lifetimeless::Read, Commands, Query, Res, ResMut, Resource},
    world::{FromWorld, World},
};
use bevy_math::{UVec2, Vec4Swizzles as _};
use bevy_render::{
    render_graph::{NodeRunError, RenderGraphApp, RenderGraphContext, ViewNode, ViewNodeRunner},
    render_resource::{
        binding_types::{sampler, texture_depth_2d, texture_storage_2d},
        BindGroup, BindGroupEntries, BindGroupLayout, BindGroupLayoutEntries,
        CachedComputePipelineId, ComputePassDescriptor, ComputePipeline, ComputePipelineDescriptor,
        Extent3d, PipelineCache, PushConstantRange, Sampler, SamplerBindingType, SamplerDescriptor,
        Shader, ShaderStages, StorageTextureAccess, TextureAspect, TextureDescriptor,
        TextureDimension, TextureFormat, TextureUsages, TextureView, TextureViewDescriptor,
        TextureViewDimension,
    },
    renderer::{RenderContext, RenderDevice},
    texture::TextureCache,
    view::{ExtractedView, ViewDepthTexture},
    Render, RenderApp, RenderSet,
};
use bevy_utils::{prelude::default, EntityHashMap};

use crate::{graph::NodePbr, ShadowView};

pub const DOWNSAMPLE_DEPTH_SHADER_HANDLE: Handle<Shader> =
    Handle::weak_from_u128(18069568929807627187);

pub struct HiZPlugin;

#[derive(Resource)]
pub struct HiZPipelineIds {
    downsample_depth_first: CachedComputePipelineId,
    downsample_depth_second: CachedComputePipelineId,
}

pub struct HiZPipelines<'w> {
    downsample_depth_first: &'w ComputePipeline,
    downsample_depth_second: &'w ComputePipeline,
}

#[derive(Resource)]
pub struct HiZBindGroupLayouts {
    downsample_depth: BindGroupLayout,
    depth_pyramid_sampler: Sampler,
    depth_pyramid_dummy_texture: TextureView,
}

pub struct HiZViewResources {
    pub depth_pyramid_all_mips: TextureView,
    depth_pyramid_mips: [TextureView; 12],
    pub depth_pyramid_mip_count: u32,
    pub previous_depth_pyramid: TextureView,
    pub view_size: UVec2,
}

#[derive(Component)]
pub struct HiZBindGroups {
    pub downsample_depth: BindGroup,
}

#[derive(Default, Resource, Deref, DerefMut)]
pub struct RenderHiZViewResources(EntityHashMap<Entity, HiZViewResources>);

#[derive(Default)]
pub struct EarlyDownsampleDepthBufferNode;

#[derive(Default)]
pub struct LateDownsampleDepthBufferNode;

impl Plugin for HiZPlugin {
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
            .init_resource::<RenderHiZViewResources>()
            .add_systems(
                Render,
                prepare_hi_z_resources.in_set(RenderSet::PrepareResources),
            )
            .add_systems(
                Render,
                prepare_hi_z_bind_groups.in_set(RenderSet::PrepareBindGroups),
            )
            .add_render_graph_node::<ViewNodeRunner<EarlyDownsampleDepthBufferNode>>(
                Core3d,
                NodePbr::EarlyDownsampleDepthBuffer,
            )
            .add_render_graph_node::<ViewNodeRunner<LateDownsampleDepthBufferNode>>(
                Core3d,
                NodePbr::LateDownsampleDepthBuffer,
            )
            .add_render_graph_edges(
                Core3d,
                (
                    Node3d::EarlyPrepass,
                    NodePbr::EarlyDownsampleDepthBuffer,
                    NodePbr::LateGpuPreprocess,
                ),
            )
            .add_render_graph_edges(
                Core3d,
                (
                    Node3d::LatePrepass,
                    NodePbr::LateDownsampleDepthBuffer,
                    Node3d::EndMainPass,
                ),
            );
    }

    fn finish(&self, app: &mut App) {
        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };

        render_app
            .init_resource::<HiZBindGroupLayouts>()
            .init_resource::<HiZPipelineIds>();
    }
}

impl FromWorld for HiZBindGroupLayouts {
    fn from_world(world: &mut World) -> Self {
        let render_device = world.resource::<RenderDevice>();

        Self {
            downsample_depth: render_device.create_bind_group_layout(
                "downsample_depth_bind_group_layout",
                &BindGroupLayoutEntries::sequential(ShaderStages::COMPUTE, {
                    let write_only_r32float = || {
                        texture_storage_2d(TextureFormat::R32Float, StorageTextureAccess::WriteOnly)
                    };
                    (
                        texture_depth_2d(),
                        write_only_r32float(),
                        write_only_r32float(),
                        write_only_r32float(),
                        write_only_r32float(),
                        write_only_r32float(),
                        texture_storage_2d(
                            TextureFormat::R32Float,
                            StorageTextureAccess::ReadWrite,
                        ),
                        write_only_r32float(),
                        write_only_r32float(),
                        write_only_r32float(),
                        write_only_r32float(),
                        write_only_r32float(),
                        write_only_r32float(),
                        sampler(SamplerBindingType::NonFiltering),
                    )
                }),
            ),

            depth_pyramid_dummy_texture: render_device
                .create_texture(&TextureDescriptor {
                    label: Some("depth_pyramid_dummy_texture"),
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
                    label: Some("depth_pyramid_dummy_texture_view"),
                    format: Some(TextureFormat::R32Float),
                    dimension: Some(TextureViewDimension::D2),
                    aspect: TextureAspect::All,
                    base_mip_level: 0,
                    mip_level_count: Some(1),
                    base_array_layer: 0,
                    array_layer_count: Some(1),
                }),

            depth_pyramid_sampler: render_device.create_sampler(&SamplerDescriptor {
                label: Some("depth_pyramid_sampler"),
                ..default()
            }),
        }
    }
}

impl FromWorld for HiZPipelineIds {
    fn from_world(world: &mut World) -> Self {
        let hi_z_bind_group_layouts = world.resource::<HiZBindGroupLayouts>();
        let downsample_depth_layout = hi_z_bind_group_layouts.downsample_depth.clone();
        let pipeline_cache = world.resource_mut::<PipelineCache>();

        Self {
            downsample_depth_first: pipeline_cache.queue_compute_pipeline(
                ComputePipelineDescriptor {
                    label: Some("downsample_depth_first_pipeline".into()),
                    layout: vec![downsample_depth_layout.clone()],
                    push_constant_ranges: vec![PushConstantRange {
                        stages: ShaderStages::COMPUTE,
                        range: 0..4,
                    }],
                    shader: DOWNSAMPLE_DEPTH_SHADER_HANDLE,
                    shader_defs: vec![],
                    entry_point: "downsample_depth_first".into(),
                },
            ),
            downsample_depth_second: pipeline_cache.queue_compute_pipeline(
                ComputePipelineDescriptor {
                    label: Some("downsample_depth_second_pipeline".into()),
                    layout: vec![downsample_depth_layout],
                    push_constant_ranges: vec![PushConstantRange {
                        stages: ShaderStages::COMPUTE,
                        range: 0..4,
                    }],
                    shader: DOWNSAMPLE_DEPTH_SHADER_HANDLE,
                    shader_defs: vec![],
                    entry_point: "downsample_depth_second".into(),
                },
            ),
        }
    }
}

impl HiZPipelineIds {
    pub fn get<'w>(&self, pipeline_cache: &'w PipelineCache) -> Option<HiZPipelines<'w>> {
        Some(HiZPipelines {
            downsample_depth_first: pipeline_cache
                .get_compute_pipeline(self.downsample_depth_first)?,
            downsample_depth_second: pipeline_cache
                .get_compute_pipeline(self.downsample_depth_second)?,
        })
    }
}

impl ViewNode for EarlyDownsampleDepthBufferNode {
    type ViewQuery = (Entity, Read<HiZBindGroups>);

    fn run<'w>(
        &self,
        _: &mut RenderGraphContext,
        render_context: &mut RenderContext<'w>,
        (view_entity, hi_z_bind_groups): QueryItem<'w, Self::ViewQuery>,
        world: &'w World,
    ) -> Result<(), NodeRunError> {
        let (Some(pipeline_cache), Some(hi_z_pipeline_ids), Some(render_hi_z_view_resources)) = (
            world.get_resource::<PipelineCache>(),
            world.get_resource::<HiZPipelineIds>(),
            world.get_resource::<RenderHiZViewResources>(),
        ) else {
            return Ok(());
        };
        let (Some(hi_z_pipelines), Some(hi_z_view_resources)) = (
            hi_z_pipeline_ids.get(pipeline_cache),
            render_hi_z_view_resources.get(&view_entity),
        ) else {
            return Ok(());
        };

        downsample_depth(
            "early depth downsampling",
            render_context,
            &hi_z_bind_groups.downsample_depth,
            hi_z_pipelines.downsample_depth_first,
            hi_z_pipelines.downsample_depth_second,
            hi_z_view_resources.depth_pyramid_mip_count,
            hi_z_view_resources.view_size,
        );

        Ok(())
    }
}

impl ViewNode for LateDownsampleDepthBufferNode {
    type ViewQuery = (Entity, Read<HiZBindGroups>);

    fn run<'w>(
        &self,
        _: &mut RenderGraphContext,
        render_context: &mut RenderContext<'w>,
        (view_entity, hi_z_bind_groups): QueryItem<'w, Self::ViewQuery>,
        world: &'w World,
    ) -> Result<(), NodeRunError> {
        let (Some(pipeline_cache), Some(hi_z_pipeline_ids), Some(render_hi_z_view_resources)) = (
            world.get_resource::<PipelineCache>(),
            world.get_resource::<HiZPipelineIds>(),
            world.get_resource::<RenderHiZViewResources>(),
        ) else {
            return Ok(());
        };
        let (Some(hi_z_pipelines), Some(hi_z_view_resources)) = (
            hi_z_pipeline_ids.get(pipeline_cache),
            render_hi_z_view_resources.get(&view_entity),
        ) else {
            return Ok(());
        };

        downsample_depth(
            "late depth downsampling",
            render_context,
            &hi_z_bind_groups.downsample_depth,
            hi_z_pipelines.downsample_depth_first,
            hi_z_pipelines.downsample_depth_second,
            hi_z_view_resources.depth_pyramid_mip_count,
            hi_z_view_resources.view_size,
        );

        Ok(())
    }
}

pub fn prepare_hi_z_resources(
    render_device: Res<RenderDevice>,
    mut texture_cache: ResMut<TextureCache>,
    hi_z_bind_group_layouts: Res<HiZBindGroupLayouts>,
    mut render_hi_z_view_resources: ResMut<RenderHiZViewResources>,
    views: Query<(Entity, &ExtractedView)>,
) {
    for (view_entity, view) in &views {
        let depth_pyramid_size = Extent3d {
            width: view.viewport.z.div_ceil(2),
            height: view.viewport.w.div_ceil(2),
            depth_or_array_layers: 1,
        };
        let depth_pyramid_mip_count = depth_pyramid_size.max_mips(TextureDimension::D2);
        let depth_pyramid = texture_cache.get(
            &render_device,
            TextureDescriptor {
                label: Some("depth_pyramid"),
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
                    label: Some("depth_pyramid_texture_view"),
                    format: Some(TextureFormat::R32Float),
                    dimension: Some(TextureViewDimension::D2),
                    aspect: TextureAspect::All,
                    base_mip_level: i as u32,
                    mip_level_count: Some(1),
                    base_array_layer: 0,
                    array_layer_count: Some(1),
                })
            } else {
                hi_z_bind_group_layouts.depth_pyramid_dummy_texture.clone()
            }
        });

        let depth_pyramid_all_mips = depth_pyramid.texture.create_view(&TextureViewDescriptor {
            label: Some("depth pyramid texture view 2"),
            format: Some(TextureFormat::R32Float),
            dimension: Some(TextureViewDimension::D2),
            aspect: TextureAspect::All,
            base_mip_level: 0,
            mip_level_count: None,
            base_array_layer: 0,
            array_layer_count: None,
        });

        // Fetch the previous depth pyramid.
        let previous_depth_pyramid = match render_hi_z_view_resources.get(&view_entity) {
            Some(previous_hi_z_view_resources) => {
                previous_hi_z_view_resources.depth_pyramid_all_mips.clone()
            }
            None => depth_pyramid_all_mips.clone(),
        };

        // TODO: Garbage collect these?
        render_hi_z_view_resources.insert(
            view_entity,
            HiZViewResources {
                depth_pyramid_all_mips,
                depth_pyramid_mips,
                depth_pyramid_mip_count,
                previous_depth_pyramid,
                view_size: view.viewport.zw(),
            },
        );
    }
}

pub fn prepare_hi_z_bind_groups(
    mut commands: Commands,
    render_device: Res<RenderDevice>,
    hi_z_bind_group_layouts: Res<HiZBindGroupLayouts>,
    render_hi_z_view_resources: Res<RenderHiZViewResources>,
    views: Query<(Entity, AnyOf<(&ViewDepthTexture, &ShadowView)>)>,
) {
    for (view_entity, view_depth) in &views {
        let Some(view_resources) = render_hi_z_view_resources.get(&view_entity) else {
            continue;
        };

        let view_depth_texture = match view_depth {
            (Some(view_depth), None) => view_depth.view(),
            (None, Some(shadow_view)) => &shadow_view.depth_attachment.view,
            _ => unreachable!(),
        };

        let downsample_depth_bind_group = render_device.create_bind_group(
            "downsample_depth_bind_group",
            &hi_z_bind_group_layouts.downsample_depth,
            &BindGroupEntries::sequential((
                view_depth_texture,
                &view_resources.depth_pyramid_mips[0],
                &view_resources.depth_pyramid_mips[1],
                &view_resources.depth_pyramid_mips[2],
                &view_resources.depth_pyramid_mips[3],
                &view_resources.depth_pyramid_mips[4],
                &view_resources.depth_pyramid_mips[5],
                &view_resources.depth_pyramid_mips[6],
                &view_resources.depth_pyramid_mips[7],
                &view_resources.depth_pyramid_mips[8],
                &view_resources.depth_pyramid_mips[9],
                &view_resources.depth_pyramid_mips[10],
                &view_resources.depth_pyramid_mips[11],
                &hi_z_bind_group_layouts.depth_pyramid_sampler,
            )),
        );

        commands.entity(view_entity).insert(HiZBindGroups {
            downsample_depth: downsample_depth_bind_group,
        });
    }
}

pub(crate) fn downsample_depth(
    label: &str,
    render_context: &mut RenderContext,
    downsample_depth_bind_group: &BindGroup,
    downsample_depth_first_pipeline: &ComputePipeline,
    downsample_depth_second_pipeline: &ComputePipeline,
    depth_pyramid_mip_count: u32,
    view_size: UVec2,
) {
    let command_encoder = render_context.command_encoder();
    let mut downsample_pass = command_encoder.begin_compute_pass(&ComputePassDescriptor {
        label: Some(label),
        timestamp_writes: None,
    });
    downsample_pass.set_pipeline(downsample_depth_first_pipeline);
    downsample_pass.set_push_constants(0, &depth_pyramid_mip_count.to_le_bytes());
    downsample_pass.set_bind_group(0, downsample_depth_bind_group, &[]);
    downsample_pass.dispatch_workgroups(view_size.x.div_ceil(64), view_size.y.div_ceil(64), 1);

    if depth_pyramid_mip_count >= 7 {
        downsample_pass.set_pipeline(downsample_depth_second_pipeline);
        downsample_pass.dispatch_workgroups(1, 1, 1);
    }
}
