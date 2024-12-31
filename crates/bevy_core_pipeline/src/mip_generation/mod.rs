//! Downsampling of textures to produce mipmap levels.

use std::array;

use bevy_app::{App, Plugin};
use bevy_asset::{load_internal_asset, Handle};
use bevy_derive::{Deref, DerefMut};
use bevy_ecs::{
    entity::Entity,
    query::{QueryItem, With},
    system::{lifetimeless::Read, Commands, Query, Res, Resource},
    world::{FromWorld, World},
};
use bevy_math::{IVec4, UVec4};
use bevy_render::{
    render_graph::{NodeRunError, RenderGraphApp, RenderGraphContext, ViewNode, ViewNodeRunner},
    render_resource::{
        binding_types::{sampler, storage_buffer_read_only_sized, texture_storage_2d},
        BindGroupEntries, BindGroupLayout, BindGroupLayoutEntries, CachedComputePipelineId,
        ComputePipelineDescriptor, Extent3d, PipelineCache, PushConstantRange, SamplerBindingType,
        Shader, ShaderStages, StorageTextureAccess, TextureAspect, TextureDescriptor,
        TextureDimension, TextureFormat, TextureUsages, TextureView, TextureViewDescriptor,
        TextureViewDimension,
    },
    renderer::{RenderContext, RenderDevice},
    texture::TextureCache,
    view::{ExtractedView, ViewDepthTexture},
    RenderApp,
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
            );
    }

    fn finish(&self, app: &mut App) {
        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };
        render_app
            .init_resource::<DownsampleDepthBindGroupLayout>()
            .init_resource::<DownsampleDepthPipelines>();
    }
}

#[derive(Default)]
pub struct DownsampleDepthNode;

impl ViewNode for DownsampleDepthNode {
    type ViewQuery = Read<OcclusionCulling>;

    fn run<'w>(
        &self,
        graph: &mut RenderGraphContext,
        render_context: &mut RenderContext<'w>,
        view_query: QueryItem<'w, Self::ViewQuery>,
        world: &'w World,
    ) -> Result<(), NodeRunError> {
        // TODO
        Ok(())
    }
}

#[derive(Resource, Deref, DerefMut)]
pub struct DownsampleDepthBindGroupLayout(BindGroupLayout);

impl FromWorld for DownsampleDepthBindGroupLayout {
    fn from_world(world: &mut World) -> Self {
        let render_device = world.resource::<RenderDevice>();

        Self(render_device.create_bind_group_layout(
            "downsample depth bind group layout",
            &BindGroupLayoutEntries::sequential(
                ShaderStages::COMPUTE,
                (
                    // TODO: this is probably wrong, it's specialized to meshlets
                    storage_buffer_read_only_sized(false, None),
                    texture_storage_2d(TextureFormat::R32Float, StorageTextureAccess::WriteOnly),
                    texture_storage_2d(TextureFormat::R32Float, StorageTextureAccess::WriteOnly),
                    texture_storage_2d(TextureFormat::R32Float, StorageTextureAccess::WriteOnly),
                    texture_storage_2d(TextureFormat::R32Float, StorageTextureAccess::WriteOnly),
                    texture_storage_2d(TextureFormat::R32Float, StorageTextureAccess::WriteOnly),
                    texture_storage_2d(TextureFormat::R32Float, StorageTextureAccess::ReadWrite),
                    texture_storage_2d(TextureFormat::R32Float, StorageTextureAccess::WriteOnly),
                    texture_storage_2d(TextureFormat::R32Float, StorageTextureAccess::WriteOnly),
                    texture_storage_2d(TextureFormat::R32Float, StorageTextureAccess::WriteOnly),
                    texture_storage_2d(TextureFormat::R32Float, StorageTextureAccess::WriteOnly),
                    texture_storage_2d(TextureFormat::R32Float, StorageTextureAccess::WriteOnly),
                    texture_storage_2d(TextureFormat::R32Float, StorageTextureAccess::WriteOnly),
                    sampler(SamplerBindingType::NonFiltering),
                ),
            ),
        ))
    }
}

#[derive(Resource)]
pub struct DownsampleDepthPipelines {
    first: CachedComputePipelineId,
    second: CachedComputePipelineId,
}

impl FromWorld for DownsampleDepthPipelines {
    fn from_world(world: &mut World) -> Self {
        let downsample_depth_layout =
            (**world.resource::<DownsampleDepthBindGroupLayout>()).clone();
        let pipeline_cache = world.resource_mut::<PipelineCache>();

        Self {
            first: pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
                label: Some("downsample depth first pipeline".into()),
                layout: vec![downsample_depth_layout.clone()],
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
                layout: vec![downsample_depth_layout.clone()],
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

pub struct ViewOcclusionDepthPyramid {
    pub depth_pyramid: ViewDepthPyramid,
    pub depth_pyramid_dummy_texture: TextureView,
}

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

fn prepare_view_depth_pyramids(
    mut commands: Commands,
    views: Query<(Entity, &ExtractedView), With<OcclusionCulling>>,
) {
    for (view_entity, view) in &views {}
}

fn prepare_downsample_depth_view_bind_groups(
    render_device: Res<RenderDevice>,
    downsample_depth_bind_group_layout: Res<DownsampleDepthBindGroupLayout>,
    view_depth_textures: Query<&ViewDepthTexture>,
) {
    for view_depth_texture in &view_depth_textures {
        /*let downsample_depth = render_device.create_bind_group(
            "downsample depth bind group",
            &**downsample_depth_bind_group_layout,
            &BindGroupEntries::sequential((
                depth_buffer,
                &
            )),
        );*/
    }
}
