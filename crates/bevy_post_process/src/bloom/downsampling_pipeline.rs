//! The first step of bloom: the downsampling pipeline.

use std::num::NonZero;

use bevy_core_pipeline::FullscreenShader;

use crate::bloom::{GpuBlendImmediates, MAX_BLOOM_MIP_COUNT};

use super::{Bloom, BLOOM_TEXTURE_FORMAT};
use bevy_asset::{load_embedded_asset, AssetServer, Handle};
use bevy_ecs::{
    prelude::{Component, Entity},
    resource::Resource,
    system::{Commands, Query, Res, ResMut},
};
use bevy_math::{Vec2, Vec4};
use bevy_render::{
    render_resource::{
        binding_types::{sampler, texture_2d, texture_storage_2d, uniform_buffer},
        *,
    },
    renderer::RenderDevice,
};
use bevy_shader::{Shader, ShaderDefVal};
use bevy_utils::default;

#[derive(Component)]
pub struct BloomComputeDownsamplingPipelineIds {
    pub main: CachedComputePipelineId,
}

#[derive(Resource)]
pub struct BloomComputeDownsamplingPipeline {
    /// Layout with a texture, a sampler, and uniforms
    pub input_bind_group_layout: BindGroupLayoutDescriptor,
    pub output_bind_group_layout: BindGroupLayoutDescriptor,
    pub sampler: Sampler,
    /// The compute shader asset handle.
    pub compute_shader: Handle<Shader>,
}

#[derive(Component)]
pub struct BloomRasterDownsamplingPipelineIds {
    pub main: CachedRenderPipelineId,
    pub first: CachedRenderPipelineId,
}

#[derive(Resource)]
pub struct BloomRasterDownsamplingPipeline {
    /// Layout with a texture, a sampler, and uniforms
    pub bind_group_layout: BindGroupLayoutDescriptor,
    pub sampler: Sampler,
    /// The asset handle for the fullscreen vertex shader.
    pub fullscreen_shader: FullscreenShader,
    /// The fragment shader asset handle.
    pub fragment_shader: Handle<Shader>,
}

#[derive(PartialEq, Eq, Hash, Clone)]
pub struct BloomDownsamplingPipelineKeys {
    prefilter: bool,
    first_downsample: bool,
    uniform_scale: bool,
}

/// The uniform struct extracted from [`Bloom`] attached to a Camera.
/// Will be available for use in the Bloom shader.
#[derive(Component, ShaderType, Clone)]
pub struct BloomUniforms {
    // Precomputed values used when thresholding, see https://catlikecoding.com/unity/tutorials/advanced-rendering/bloom/#3.4
    pub threshold_precomputations: Vec4,
    pub viewport: Vec4,
    pub scale: Vec2,
    pub aspect: f32,
}

pub fn init_bloom_downsampling_pipeline(
    mut commands: Commands,
    render_device: Res<RenderDevice>,
    fullscreen_shader: Res<FullscreenShader>,
    asset_server: Res<AssetServer>,
) {
    if super::bloom_can_be_done_in_compute(&render_device) {
        init_bloom_compute_downsampling_pipeline(&mut commands, &render_device, &asset_server);
    }
    init_bloom_raster_downsampling_pipeline(
        &mut commands,
        &render_device,
        &fullscreen_shader,
        &asset_server,
    );
}

fn init_bloom_compute_downsampling_pipeline(
    commands: &mut Commands,
    render_device: &RenderDevice,
    asset_server: &AssetServer,
) {
    // Input bind group layout
    let input_bind_group_layout = BindGroupLayoutDescriptor::new(
        "bloom_compute_downsampling_input_bind_group_layout_with_settings",
        &BindGroupLayoutEntries::single(
            ShaderStages::COMPUTE,
            // Downsampling settings binding
            uniform_buffer::<BloomUniforms>(true),
        ),
    );
    // Output bind group layout
    let output_bind_group_layout = BindGroupLayoutDescriptor::new(
        "bloom_compute_downsampling_output_bind_group_layout_with_settings",
        &BindGroupLayoutEntries::single(
            ShaderStages::COMPUTE,
            // Output texture binding
            texture_storage_2d(BLOOM_TEXTURE_FORMAT, StorageTextureAccess::ReadWrite)
                .count(NonZero::new(MAX_BLOOM_MIP_COUNT).unwrap()),
        ),
    );

    // Sampler
    let sampler = render_device.create_sampler(&SamplerDescriptor {
        min_filter: FilterMode::Linear,
        mag_filter: FilterMode::Linear,
        address_mode_u: AddressMode::ClampToEdge,
        address_mode_v: AddressMode::ClampToEdge,
        ..Default::default()
    });

    commands.insert_resource(BloomComputeDownsamplingPipeline {
        input_bind_group_layout,
        output_bind_group_layout,
        sampler,
        compute_shader: load_embedded_asset!(asset_server, "bloom.wgsl"),
    });
}

fn init_bloom_raster_downsampling_pipeline(
    commands: &mut Commands,
    render_device: &RenderDevice,
    fullscreen_shader: &FullscreenShader,
    asset_server: &AssetServer,
) {
    // Bind group layout
    let bind_group_layout = BindGroupLayoutDescriptor::new(
        "bloom_raster_downsampling_bind_group_layout_with_settings",
        &BindGroupLayoutEntries::sequential(
            ShaderStages::FRAGMENT,
            (
                // Downsampling settings binding
                uniform_buffer::<BloomUniforms>(true),
                // Sampler binding
                sampler(SamplerBindingType::Filtering),
                // Input texture binding
                texture_2d(TextureSampleType::Float { filterable: true }),
            ),
        ),
    );

    // Sampler
    let sampler = render_device.create_sampler(&SamplerDescriptor {
        min_filter: FilterMode::Linear,
        mag_filter: FilterMode::Linear,
        address_mode_u: AddressMode::ClampToEdge,
        address_mode_v: AddressMode::ClampToEdge,
        ..Default::default()
    });

    commands.insert_resource(BloomRasterDownsamplingPipeline {
        bind_group_layout,
        sampler,
        fullscreen_shader: fullscreen_shader.clone(),
        fragment_shader: load_embedded_asset!(asset_server, "bloom.wgsl"),
    });
}

impl BloomDownsamplingPipelineKeys {
    fn get_shader_defs(&self) -> Vec<ShaderDefVal> {
        let mut shader_defs = vec![];

        if self.first_downsample {
            shader_defs.push("FIRST_DOWNSAMPLE".into());
        }

        if self.prefilter {
            shader_defs.push("USE_THRESHOLD".into());
        }

        if self.uniform_scale {
            shader_defs.push("UNIFORM_SCALE".into());
        }

        shader_defs
    }
}

impl SpecializedComputePipeline for BloomComputeDownsamplingPipeline {
    type Key = BloomDownsamplingPipelineKeys;

    fn specialize(&self, key: Self::Key) -> ComputePipelineDescriptor {
        let layout = vec![
            self.input_bind_group_layout.clone(),
            self.output_bind_group_layout.clone(),
        ];

        let entry_point = if key.first_downsample {
            "downsample_first".into()
        } else {
            "downsample".into()
        };

        let mut shader_defs = key.get_shader_defs();
        shader_defs.push("BLOOM_COMPUTE".into());

        ComputePipelineDescriptor {
            label: Some(
                if key.first_downsample {
                    "bloom_compute_downsampling_pipeline_first"
                } else {
                    "bloom_compute_downsampling_pipeline"
                }
                .into(),
            ),
            layout,
            shader: self.compute_shader.clone(),
            shader_defs,
            entry_point: Some(entry_point),
            immediate_size: size_of::<GpuBlendImmediates>() as u32,
            ..default()
        }
    }
}

impl SpecializedRenderPipeline for BloomRasterDownsamplingPipeline {
    type Key = BloomDownsamplingPipelineKeys;

    fn specialize(&self, key: Self::Key) -> RenderPipelineDescriptor {
        let layout = vec![self.bind_group_layout.clone()];

        let entry_point = if key.first_downsample {
            "downsample_first".into()
        } else {
            "downsample".into()
        };

        RenderPipelineDescriptor {
            label: Some(
                if key.first_downsample {
                    "bloom_raster_downsampling_pipeline_first"
                } else {
                    "bloom_raster_downsampling_pipeline"
                }
                .into(),
            ),
            layout,
            vertex: self.fullscreen_shader.to_vertex_state(),
            fragment: Some(FragmentState {
                shader: self.fragment_shader.clone(),
                shader_defs: key.get_shader_defs(),
                entry_point: Some(entry_point),
                targets: vec![Some(ColorTargetState {
                    format: BLOOM_TEXTURE_FORMAT,
                    blend: None,
                    write_mask: ColorWrites::ALL,
                })],
                constants: vec![],
            }),
            ..default()
        }
    }
}

pub fn prepare_downsampling_pipeline(
    mut commands: Commands,
    pipeline_cache: Res<PipelineCache>,
    mut maybe_compute_pipelines: Option<
        ResMut<SpecializedComputePipelines<BloomComputeDownsamplingPipeline>>,
    >,
    maybe_compute_pipeline: Option<Res<BloomComputeDownsamplingPipeline>>,
    mut maybe_raster_pipelines: Option<
        ResMut<SpecializedRenderPipelines<BloomRasterDownsamplingPipeline>>,
    >,
    maybe_raster_pipeline: Option<Res<BloomRasterDownsamplingPipeline>>,
    views: Query<(Entity, &Bloom)>,
) {
    for (entity, bloom) in &views {
        let prefilter = bloom.prefilter.threshold > 0.0;

        let main_downsample_pipeline_keys = BloomDownsamplingPipelineKeys {
            prefilter,
            first_downsample: false,
            uniform_scale: bloom.scale == Vec2::ONE,
        };
        let first_downsample_pipeline_keys = BloomDownsamplingPipelineKeys {
            prefilter,
            first_downsample: true,
            uniform_scale: bloom.scale == Vec2::ONE,
        };

        if let (&mut Some(ref mut raster_pipelines), Some(raster_pipeline)) =
            (&mut maybe_raster_pipelines, &maybe_raster_pipeline)
        {
            let raster_pipeline_main_id = raster_pipelines.specialize(
                &pipeline_cache,
                raster_pipeline,
                main_downsample_pipeline_keys.clone(),
            );

            let raster_pipeline_first_id = raster_pipelines.specialize(
                &pipeline_cache,
                raster_pipeline,
                first_downsample_pipeline_keys.clone(),
            );

            commands
                .entity(entity)
                .insert(BloomRasterDownsamplingPipelineIds {
                    first: raster_pipeline_first_id,
                    main: raster_pipeline_main_id,
                });
        }

        if let (&mut Some(ref mut compute_pipelines), Some(compute_pipeline)) =
            (&mut maybe_compute_pipelines, &maybe_compute_pipeline)
        {
            let compute_pipeline_main_id = compute_pipelines.specialize(
                &pipeline_cache,
                compute_pipeline,
                main_downsample_pipeline_keys,
            );

            commands
                .entity(entity)
                .insert(BloomComputeDownsamplingPipelineIds {
                    main: compute_pipeline_main_id,
                });
        }
    }
}
