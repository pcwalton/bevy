//! The second step of bloom: the upsampling pipeline.

use std::num::NonZero;

use bevy_core_pipeline::FullscreenShader;

use crate::bloom::{GpuBlendImmediates, MAX_BLOOM_MIP_COUNT};

use super::{
    downsampling_pipeline::BloomUniforms, Bloom, BloomCompositeMode, BLOOM_TEXTURE_FORMAT,
};
use bevy_asset::{load_embedded_asset, AssetServer, Handle};
use bevy_ecs::{
    prelude::{Component, Entity},
    resource::Resource,
    system::{Commands, Query, Res, ResMut},
};
use bevy_render::{
    render_resource::{
        binding_types::{sampler, texture_2d, texture_storage_2d, uniform_buffer},
        *,
    },
    renderer::RenderDevice,
    view::ExtractedView,
};
use bevy_shader::Shader;
use bevy_utils::default;

#[derive(Component)]
pub struct BloomComputeUpsamplingPipelineIds {
    pub id_main: CachedComputePipelineId,
}

#[derive(Component)]
pub struct BloomRasterUpsamplingPipelineIds {
    pub id_main: CachedRenderPipelineId,
    pub id_final: CachedRenderPipelineId,
}

#[derive(Resource)]
pub struct BloomComputeUpsamplingPipeline {
    pub input_bind_group_layout: BindGroupLayoutDescriptor,
    pub output_bind_group_layout: BindGroupLayoutDescriptor,
    /// The fragment shader asset handle.
    pub compute_shader: Handle<Shader>,
}

#[derive(Resource)]
pub struct BloomRasterUpsamplingPipeline {
    pub bind_group_layout: BindGroupLayoutDescriptor,
    /// The asset handle for the fullscreen vertex shader.
    pub fullscreen_shader: FullscreenShader,
    /// The fragment shader asset handle.
    pub fragment_shader: Handle<Shader>,
}

#[derive(PartialEq, Eq, Hash, Clone)]
pub struct BloomUpsamplingPipelineKeys {
    composite_mode: BloomCompositeMode,
    target_format: TextureFormat,
}

pub fn init_bloom_upscaling_pipeline(
    mut commands: Commands,
    render_device: Res<RenderDevice>,
    fullscreen_shader: Res<FullscreenShader>,
    asset_server: Res<AssetServer>,
) {
    if super::bloom_can_be_done_in_compute(&render_device) {
        init_bloom_compute_upsampling_pipeline(&mut commands, &asset_server);
    }
    init_bloom_raster_upsampling_pipeline(&mut commands, &fullscreen_shader, &asset_server);
}

fn init_bloom_compute_upsampling_pipeline(commands: &mut Commands, asset_server: &AssetServer) {
    let input_bind_group_layout = BindGroupLayoutDescriptor::new(
        "bloom_compute_upsampling_input_bind_group_layout",
        &BindGroupLayoutEntries::single(
            ShaderStages::COMPUTE,
            // BloomUniforms
            uniform_buffer::<BloomUniforms>(true),
        ),
    );
    let output_bind_group_layout = BindGroupLayoutDescriptor::new(
        "bloom_compute_upsampling_output_bind_group_layout",
        &BindGroupLayoutEntries::single(
            ShaderStages::COMPUTE,
            // Output texture binding
            texture_storage_2d(BLOOM_TEXTURE_FORMAT, StorageTextureAccess::ReadWrite)
                .count(NonZero::new(MAX_BLOOM_MIP_COUNT).unwrap()),
        ),
    );

    commands.insert_resource(BloomComputeUpsamplingPipeline {
        input_bind_group_layout,
        output_bind_group_layout,
        compute_shader: load_embedded_asset!(asset_server, "bloom.wgsl"),
    });
}

fn init_bloom_raster_upsampling_pipeline(
    commands: &mut Commands,
    fullscreen_shader: &FullscreenShader,
    asset_server: &AssetServer,
) {
    let bind_group_layout = BindGroupLayoutDescriptor::new(
        "bloom_raster_upsampling_bind_group_layout",
        &BindGroupLayoutEntries::sequential(
            ShaderStages::FRAGMENT,
            (
                // BloomUniforms
                uniform_buffer::<BloomUniforms>(true),
                // Sampler
                sampler(SamplerBindingType::Filtering),
                // Input texture
                texture_2d(TextureSampleType::Float { filterable: true }),
            ),
        ),
    );

    commands.insert_resource(BloomRasterUpsamplingPipeline {
        bind_group_layout,
        fullscreen_shader: fullscreen_shader.clone(),
        fragment_shader: load_embedded_asset!(asset_server, "bloom.wgsl"),
    });
}

impl SpecializedComputePipeline for BloomComputeUpsamplingPipeline {
    type Key = BloomUpsamplingPipelineKeys;

    fn specialize(&self, _: Self::Key) -> ComputePipelineDescriptor {
        ComputePipelineDescriptor {
            label: Some("bloom_compute_upsampling_pipeline".into()),
            layout: vec![
                self.input_bind_group_layout.clone(),
                self.output_bind_group_layout.clone(),
            ],
            shader: self.compute_shader.clone(),
            entry_point: Some("upsample".into()),
            shader_defs: vec!["BLOOM_COMPUTE".into()],
            immediate_size: size_of::<GpuBlendImmediates>() as u32,
            ..default()
        }
    }
}

impl SpecializedRenderPipeline for BloomRasterUpsamplingPipeline {
    type Key = BloomUpsamplingPipelineKeys;

    fn specialize(&self, key: Self::Key) -> RenderPipelineDescriptor {
        let color_blend = match key.composite_mode {
            BloomCompositeMode::EnergyConserving => {
                // At the time of developing this we decided to blend our
                // blur pyramid levels using native WGPU render pass blend
                // constants. They are set in the bloom node's run function.
                // This seemed like a good approach at the time which allowed
                // us to perform complex calculations for blend levels on the CPU,
                // however, we missed the fact that this prevented us from using
                // textures to customize bloom appearance on individual parts
                // of the screen and create effects such as lens dirt or
                // screen blur behind certain UI elements.
                //
                // TODO: Use alpha instead of blend constants and move
                // compute_blend_factor to the shader. The shader
                // will likely need to know current mip number or
                // mip "angle" (original texture is 0deg, max mip is 90deg)
                // so make sure you give it that as a uniform.
                // That does have to be provided per each pass unlike other
                // uniforms that are set once.
                BlendComponent {
                    src_factor: BlendFactor::Constant,
                    dst_factor: BlendFactor::OneMinusConstant,
                    operation: BlendOperation::Add,
                }
            }
            BloomCompositeMode::Additive => BlendComponent {
                src_factor: BlendFactor::Constant,
                dst_factor: BlendFactor::One,
                operation: BlendOperation::Add,
            },
        };

        RenderPipelineDescriptor {
            label: Some("bloom_raster_upsampling_pipeline".into()),
            layout: vec![self.bind_group_layout.clone()],
            vertex: self.fullscreen_shader.to_vertex_state(),
            fragment: Some(FragmentState {
                shader: self.fragment_shader.clone(),
                entry_point: Some("upsample".into()),
                targets: vec![Some(ColorTargetState {
                    format: key.target_format,
                    blend: Some(BlendState {
                        color: color_blend,
                        alpha: BlendComponent {
                            src_factor: BlendFactor::Zero,
                            dst_factor: BlendFactor::One,
                            operation: BlendOperation::Add,
                        },
                    }),
                    write_mask: ColorWrites::ALL,
                })],
                ..default()
            }),
            ..default()
        }
    }
}

pub fn prepare_upsampling_pipeline(
    mut commands: Commands,
    pipeline_cache: Res<PipelineCache>,
    mut maybe_compute_pipelines: Option<
        ResMut<SpecializedComputePipelines<BloomComputeUpsamplingPipeline>>,
    >,
    maybe_compute_pipeline: Option<Res<BloomComputeUpsamplingPipeline>>,
    mut maybe_raster_pipelines: Option<
        ResMut<SpecializedRenderPipelines<BloomRasterUpsamplingPipeline>>,
    >,
    maybe_raster_pipeline: Option<Res<BloomRasterUpsamplingPipeline>>,
    views: Query<(&ExtractedView, Entity, &Bloom)>,
) {
    for (view, entity, bloom) in &views {
        if let (&mut Some(ref mut raster_pipelines), Some(raster_pipeline)) =
            (&mut maybe_raster_pipelines, &maybe_raster_pipeline)
        {
            let raster_pipeline_id = raster_pipelines.specialize(
                &pipeline_cache,
                raster_pipeline,
                BloomUpsamplingPipelineKeys {
                    composite_mode: bloom.composite_mode,
                    target_format: BLOOM_TEXTURE_FORMAT,
                },
            );

            let raster_pipeline_final_id = raster_pipelines.specialize(
                &pipeline_cache,
                raster_pipeline,
                BloomUpsamplingPipelineKeys {
                    composite_mode: bloom.composite_mode,
                    target_format: view.target_format,
                },
            );

            commands
                .entity(entity)
                .insert(BloomRasterUpsamplingPipelineIds {
                    id_main: raster_pipeline_id,
                    id_final: raster_pipeline_final_id,
                });
        }

        if let (&mut Some(ref mut compute_pipelines), Some(compute_pipeline)) =
            (&mut maybe_compute_pipelines, &maybe_compute_pipeline)
        {
            let compute_pipeline_id = compute_pipelines.specialize(
                &pipeline_cache,
                compute_pipeline,
                BloomUpsamplingPipelineKeys {
                    composite_mode: bloom.composite_mode,
                    target_format: BLOOM_TEXTURE_FORMAT,
                },
            );

            commands
                .entity(entity)
                .insert(BloomComputeUpsamplingPipelineIds {
                    id_main: compute_pipeline_id,
                });
        }
    }
}
