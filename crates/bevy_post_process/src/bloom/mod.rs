//! Simulates the effect of extremely bright light scattering inside an optical
//! lens.
//!
//! This effectively creates a halo around bright objects.

mod downsampling_pipeline;
mod settings;
mod upsampling_pipeline;

use bevy_image::ToExtents;
use bytemuck::{Pod, Zeroable};
pub use settings::{Bloom, BloomCompositeMode, BloomPrefilter};

use crate::bloom::{
    downsampling_pipeline::{
        init_bloom_downsampling_pipeline, BloomComputeDownsamplingPipeline,
        BloomComputeDownsamplingPipelineIds,
    },
    upsampling_pipeline::{
        init_bloom_upscaling_pipeline, BloomComputeUpsamplingPipeline,
        BloomComputeUpsamplingPipelineIds, BloomRasterUpsamplingPipeline,
        BloomRasterUpsamplingPipelineIds,
    },
};
use bevy_app::{App, Plugin};
use bevy_asset::embedded_asset;
use bevy_color::{Gray, LinearRgba};
use bevy_core_pipeline::{
    schedule::{Core2d, Core2dSystems, Core3d, Core3dSystems},
    tonemapping::tonemapping,
};
use bevy_ecs::prelude::*;
use bevy_math::{ops, uvec2, UVec2};
use bevy_render::{
    camera::ExtractedCamera,
    diagnostic::RecordDiagnostics,
    extract_component::{
        ComponentUniforms, DynamicUniformIndex, ExtractComponentPlugin, UniformComponentPlugin,
    },
    render_resource::*,
    renderer::{RenderContext, RenderDevice, ViewQuery},
    texture::{CachedTexture, TextureCache},
    view::ViewTarget,
    GpuResourceAppExt, Render, RenderApp, RenderStartup, RenderSystems,
};
use downsampling_pipeline::{
    prepare_downsampling_pipeline, BloomRasterDownsamplingPipeline,
    BloomRasterDownsamplingPipelineIds, BloomUniforms,
};
use upsampling_pipeline::prepare_upsampling_pipeline;

const BLOOM_TEXTURE_FORMAT: TextureFormat = TextureFormat::Rg11b10Ufloat;
const MAX_BLOOM_MIP_COUNT: u32 = 16;

#[derive(Default)]
pub struct BloomPlugin;

impl Plugin for BloomPlugin {
    fn build(&self, app: &mut App) {
        embedded_asset!(app, "bloom.wgsl");

        app.add_plugins((
            ExtractComponentPlugin::<Bloom>::default(),
            UniformComponentPlugin::<BloomUniforms>::default(),
        ));

        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };
        render_app
            .init_gpu_resource::<SpecializedRenderPipelines<BloomRasterDownsamplingPipeline>>()
            .init_gpu_resource::<SpecializedRenderPipelines<BloomRasterUpsamplingPipeline>>()
            .add_systems(
                RenderStartup,
                (
                    init_bloom_downsampling_pipeline,
                    init_bloom_upscaling_pipeline,
                ),
            )
            .add_systems(
                Render,
                (
                    prepare_downsampling_pipeline.in_set(RenderSystems::Prepare),
                    prepare_upsampling_pipeline.in_set(RenderSystems::Prepare),
                    prepare_bloom_textures.in_set(RenderSystems::PrepareResources),
                ),
            )
            .add_systems(
                Core3d,
                bloom.before(tonemapping).in_set(Core3dSystems::PostProcess),
            )
            .add_systems(
                Core2d,
                bloom.before(tonemapping).in_set(Core2dSystems::PostProcess),
            );
    }

    fn finish(&self, app: &mut App) {
        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };

        let compute_supported = render_app
            .world()
            .get_resource::<RenderDevice>()
            .is_some_and(bloom_can_be_done_in_compute);
        if compute_supported {
            render_app
                .init_gpu_resource::<SpecializedComputePipelines<BloomComputeDownsamplingPipeline>>(
                )
                .init_gpu_resource::<SpecializedComputePipelines<BloomComputeUpsamplingPipeline>>()
                .add_systems(
                    Render,
                    prepare_bloom_compute_bind_groups.in_set(RenderSystems::PrepareBindGroups),
                );
        } else {
            render_app.add_systems(
                Render,
                prepare_bloom_raster_bind_groups.in_set(RenderSystems::PrepareBindGroups),
            );
        }
    }
}

pub fn bloom(
    view: ViewQuery<(
        &ExtractedCamera,
        &ViewTarget,
        &BloomTexture,
        AnyOf<(&BloomComputeBindGroups, &BloomRasterBindGroups)>,
        &DynamicUniformIndex<BloomUniforms>,
        &Bloom,
        Option<&BloomComputeUpsamplingPipelineIds>,
        &BloomRasterUpsamplingPipelineIds,
        Option<&BloomComputeDownsamplingPipelineIds>,
        &BloomRasterDownsamplingPipelineIds,
    )>,
    raster_downsampling_pipeline_res: Res<BloomRasterDownsamplingPipeline>,
    pipeline_cache: Res<PipelineCache>,
    uniforms: Res<ComponentUniforms<BloomUniforms>>,
    mut ctx: RenderContext,
) {
    let (
        camera,
        view_target,
        bloom_texture,
        (maybe_compute_bind_groups, maybe_raster_bind_groups),
        uniform_index,
        bloom_settings,
        maybe_compute_upsampling_pipeline_ids,
        raster_upsampling_pipeline_ids,
        maybe_compute_downsampling_pipeline_ids,
        raster_downsampling_pipeline_ids,
    ) = view.into_inner();

    if bloom_settings.intensity == 0.0 || !camera.hdr {
        return;
    }

    let Some(uniforms_binding) = uniforms.binding() else {
        return;
    };

    // Raster

    if let (
        Some(raster_bind_groups),
        Some(raster_downsampling_first_pipeline),
        Some(raster_downsampling_pipeline),
        Some(raster_upsampling_pipeline),
        Some(raster_upsampling_final_pipeline),
    ) = (
        maybe_raster_bind_groups,
        pipeline_cache.get_render_pipeline(raster_downsampling_pipeline_ids.first),
        pipeline_cache.get_render_pipeline(raster_downsampling_pipeline_ids.main),
        pipeline_cache.get_render_pipeline(raster_upsampling_pipeline_ids.id_main),
        pipeline_cache.get_render_pipeline(raster_upsampling_pipeline_ids.id_final),
    ) {
        let view_texture = view_target.main_texture_view();
        let view_texture_unsampled = view_target.get_unsampled_color_attachment();

        // Create the first downsampling bind group (reads from main texture)
        let downsampling_first_bind_group = ctx.render_device().create_bind_group(
            "bloom_downsampling_first_bind_group",
            &pipeline_cache
                .get_bind_group_layout(&raster_downsampling_pipeline_res.bind_group_layout),
            &BindGroupEntries::sequential((
                uniforms_binding.clone(),
                &raster_bind_groups.sampler,
                view_texture,
            )),
        );

        let diagnostics = ctx.diagnostic_recorder();
        let diagnostics = diagnostics.as_deref();
        let time_span = diagnostics.time_span(ctx.command_encoder(), "bloom");

        let command_encoder = ctx.command_encoder();
        command_encoder.push_debug_group("bloom");

        // First downsample pass
        {
            let view = &bloom_texture.view(0);
            let mut downsampling_first_pass =
                command_encoder.begin_render_pass(&RenderPassDescriptor {
                    label: Some("bloom_raster_downsampling_first_pass"),
                    color_attachments: &[Some(RenderPassColorAttachment {
                        view,
                        depth_slice: None,
                        resolve_target: None,
                        ops: Operations::default(),
                    })],
                    depth_stencil_attachment: None,
                    timestamp_writes: None,
                    occlusion_query_set: None,
                    multiview_mask: None,
                });
            downsampling_first_pass.set_pipeline(raster_downsampling_first_pipeline);
            downsampling_first_pass.set_bind_group(
                0,
                &downsampling_first_bind_group,
                &[uniform_index.index()],
            );
            downsampling_first_pass.draw(0..3, 0..1);
        }

        // Other downsample passes
        for mip in 1..bloom_texture.mip_count {
            let view = &bloom_texture.view(mip);
            let mut downsampling_pass = command_encoder.begin_render_pass(&RenderPassDescriptor {
                label: Some("bloom_raster_downsampling_pass"),
                color_attachments: &[Some(RenderPassColorAttachment {
                    view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: Operations::default(),
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            downsampling_pass.set_pipeline(raster_downsampling_pipeline);
            downsampling_pass.set_bind_group(
                0,
                &raster_bind_groups.downsampling_bind_groups[mip as usize - 1],
                &[uniform_index.index()],
            );
            downsampling_pass.draw(0..3, 0..1);
        }

        // Upsample passes except the final one
        for mip in (1..bloom_texture.mip_count).rev() {
            let view = &bloom_texture.view(mip - 1);
            let mut upsampling_pass = command_encoder.begin_render_pass(&RenderPassDescriptor {
                label: Some("bloom_raster_upsampling_pass"),
                color_attachments: &[Some(RenderPassColorAttachment {
                    view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: Operations {
                        load: LoadOp::Load,
                        store: StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            upsampling_pass.set_pipeline(raster_upsampling_pipeline);
            upsampling_pass.set_bind_group(
                0,
                &raster_bind_groups.upsampling_bind_groups
                    [(bloom_texture.mip_count - mip - 1) as usize],
                &[uniform_index.index()],
            );
            let blend = compute_blend_factor(
                bloom_settings,
                mip as f32,
                (bloom_texture.mip_count - 1) as f32,
            );
            upsampling_pass.set_blend_constant(LinearRgba::gray(blend).into());
            upsampling_pass.draw(0..3, 0..1);
        }

        // Final upsample pass
        {
            let mut upsampling_final_pass =
                command_encoder.begin_render_pass(&RenderPassDescriptor {
                    label: Some("bloom_raster_upsampling_final_pass"),
                    color_attachments: &[Some(view_texture_unsampled)],
                    depth_stencil_attachment: None,
                    timestamp_writes: None,
                    occlusion_query_set: None,
                    multiview_mask: None,
                });
            upsampling_final_pass.set_pipeline(raster_upsampling_final_pipeline);
            upsampling_final_pass.set_bind_group(
                0,
                &raster_bind_groups.upsampling_bind_groups[(bloom_texture.mip_count - 1) as usize],
                &[uniform_index.index()],
            );
            if let Some(viewport) = camera.viewport.as_ref() {
                upsampling_final_pass.set_viewport(
                    viewport.physical_position.x as f32,
                    viewport.physical_position.y as f32,
                    viewport.physical_size.x as f32,
                    viewport.physical_size.y as f32,
                    viewport.depth.start,
                    viewport.depth.end,
                );
            }
            let blend =
                compute_blend_factor(bloom_settings, 0.0, (bloom_texture.mip_count - 1) as f32);
            upsampling_final_pass.set_blend_constant(LinearRgba::gray(blend).into());
            upsampling_final_pass.draw(0..3, 0..1);
        }

        command_encoder.pop_debug_group();
        time_span.end(ctx.command_encoder());
    }

    // Compute

    if let (
        Some(compute_bind_groups),
        Some(compute_upsampling_pipeline_ids),
        Some(compute_downsampling_pipeline_ids),
    ) = (
        maybe_compute_bind_groups,
        maybe_compute_upsampling_pipeline_ids,
        maybe_compute_downsampling_pipeline_ids,
    ) && let (
        Some(raster_downsampling_first_pipeline),
        Some(compute_downsampling_pipeline),
        Some(compute_upsampling_main_pipeline),
        Some(raster_upsampling_final_pipeline),
    ) = (
        pipeline_cache.get_render_pipeline(raster_downsampling_pipeline_ids.first),
        pipeline_cache.get_compute_pipeline(compute_downsampling_pipeline_ids.main),
        pipeline_cache.get_compute_pipeline(compute_upsampling_pipeline_ids.id_main),
        pipeline_cache.get_render_pipeline(raster_upsampling_pipeline_ids.id_final),
    ) {
        let view_texture = view_target.main_texture_view();
        let view_texture_unsampled = view_target.get_unsampled_color_attachment();

        // Create the first downsampling bind group (reads from main texture)
        let downsampling_first_bind_group = ctx.render_device().create_bind_group(
            "bloom_downsampling_first_bind_group",
            &pipeline_cache
                .get_bind_group_layout(&raster_downsampling_pipeline_res.bind_group_layout),
            &BindGroupEntries::sequential((
                uniforms_binding.clone(),
                &compute_bind_groups.sampler,
                view_texture,
            )),
        );

        let diagnostics = ctx.diagnostic_recorder();
        let diagnostics = diagnostics.as_deref();
        let time_span = diagnostics.time_span(ctx.command_encoder(), "bloom");

        let command_encoder = ctx.command_encoder();
        command_encoder.push_debug_group("bloom");

        // First downsample pass
        {
            let view = &bloom_texture.view(0);
            let mut downsampling_first_pass =
                command_encoder.begin_render_pass(&RenderPassDescriptor {
                    label: Some("bloom_raster_downsampling_first_pass"),
                    color_attachments: &[Some(RenderPassColorAttachment {
                        view,
                        depth_slice: None,
                        resolve_target: None,
                        ops: Operations::default(),
                    })],
                    depth_stencil_attachment: None,
                    timestamp_writes: None,
                    occlusion_query_set: None,
                    multiview_mask: None,
                });
            downsampling_first_pass.set_pipeline(raster_downsampling_first_pipeline);
            downsampling_first_pass.set_bind_group(
                0,
                &downsampling_first_bind_group,
                &[uniform_index.index()],
            );
            downsampling_first_pass.draw(0..3, 0..1);
        }

        {
            let mut pass = command_encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("bloom_compute_pass"),
                timestamp_writes: None,
            });

            // Other downsample passes
            pass.set_pipeline(compute_downsampling_pipeline);
            pass.set_bind_group(
                0,
                &compute_bind_groups
                    .compute_main_downsampling_bind_groups
                    .input,
                &[uniform_index.index()],
            );
            pass.set_bind_group(
                1,
                &compute_bind_groups
                    .compute_main_downsampling_bind_groups
                    .output,
                &[],
            );
            for output_mip in 1..bloom_texture.mip_count {
                let input_mip = output_mip - 1;
                pass.set_immediates(
                    0,
                    bytemuck::cast_slice(&[GpuBlendImmediates {
                        input_mip,
                        output_mip,
                        src_blend_factor: 1.0,
                        dest_blend_factor: 0.0,
                    }]),
                );
                let workgroup_counts = get_workgroup_counts(bloom_texture, output_mip);
                pass.dispatch_workgroups(workgroup_counts.x, workgroup_counts.y, 1);
            }

            // Upsample passes except the final one
            pass.set_pipeline(compute_upsampling_main_pipeline);
            pass.set_bind_group(
                0,
                &compute_bind_groups
                    .compute_main_upsampling_bind_groups
                    .input,
                &[uniform_index.index()],
            );
            pass.set_bind_group(
                1,
                &compute_bind_groups
                    .compute_main_upsampling_bind_groups
                    .output,
                &[],
            );
            for input_mip in (1..bloom_texture.mip_count).rev() {
                let output_mip = input_mip - 1;
                let blend = compute_blend_factor(
                    bloom_settings,
                    input_mip as f32,
                    (bloom_texture.mip_count - 1) as f32,
                );
                pass.set_immediates(
                    0,
                    bytemuck::cast_slice(&[GpuBlendImmediates {
                        input_mip,
                        output_mip,
                        src_blend_factor: blend,
                        dest_blend_factor: match bloom_settings.composite_mode {
                            BloomCompositeMode::EnergyConserving => 1.0 - blend,
                            BloomCompositeMode::Additive => 1.0,
                        },
                    }]),
                );
                let workgroup_counts = get_workgroup_counts(bloom_texture, output_mip);
                pass.dispatch_workgroups(workgroup_counts.x, workgroup_counts.y, 1);
            }
        }

        // Final upsample pass
        {
            let mut upsampling_final_pass =
                command_encoder.begin_render_pass(&RenderPassDescriptor {
                    label: Some("bloom_raster_upsampling_final_pass"),
                    color_attachments: &[Some(view_texture_unsampled)],
                    depth_stencil_attachment: None,
                    timestamp_writes: None,
                    occlusion_query_set: None,
                    multiview_mask: None,
                });
            upsampling_final_pass.set_pipeline(raster_upsampling_final_pipeline);
            upsampling_final_pass.set_bind_group(
                0,
                &compute_bind_groups.raster_final_upsampling_bind_group,
                &[uniform_index.index()],
            );
            if let Some(viewport) = camera.viewport.as_ref() {
                upsampling_final_pass.set_viewport(
                    viewport.physical_position.x as f32,
                    viewport.physical_position.y as f32,
                    viewport.physical_size.x as f32,
                    viewport.physical_size.y as f32,
                    viewport.depth.start,
                    viewport.depth.end,
                );
            }
            let blend =
                compute_blend_factor(bloom_settings, 0.0, (bloom_texture.mip_count - 1) as f32);
            upsampling_final_pass.set_blend_constant(LinearRgba::gray(blend).into());
            upsampling_final_pass.draw(0..3, 0..1);
        }

        command_encoder.pop_debug_group();
        time_span.end(ctx.command_encoder());
    }
}

#[derive(Component)]
pub struct BloomTexture {
    // First mip is half the screen resolution, successive mips are half the previous
    #[cfg(any(
        not(feature = "webgl"),
        not(target_arch = "wasm32"),
        feature = "webgpu"
    ))]
    texture: CachedTexture,
    // WebGL does not support binding specific mip levels for sampling, fallback to separate textures instead
    #[cfg(all(feature = "webgl", target_arch = "wasm32", not(feature = "webgpu")))]
    texture: Vec<CachedTexture>,
    mip_count: u32,
}

impl BloomTexture {
    #[cfg(any(
        not(feature = "webgl"),
        not(target_arch = "wasm32"),
        feature = "webgpu"
    ))]
    fn view(&self, base_mip_level: u32) -> TextureView {
        self.texture.texture.create_view(&TextureViewDescriptor {
            base_mip_level,
            mip_level_count: Some(1u32),
            ..Default::default()
        })
    }
    #[cfg(all(feature = "webgl", target_arch = "wasm32", not(feature = "webgpu")))]
    fn view(&self, base_mip_level: u32) -> TextureView {
        self.texture[base_mip_level as usize]
            .texture
            .create_view(&TextureViewDescriptor {
                base_mip_level: 0,
                mip_level_count: Some(1u32),
                ..Default::default()
            })
    }
}

fn prepare_bloom_textures(
    mut commands: Commands,
    mut texture_cache: ResMut<TextureCache>,
    render_device: Res<RenderDevice>,
    views: Query<(Entity, &ExtractedCamera, &Bloom)>,
) {
    for (entity, camera, bloom) in &views {
        if let Some(viewport) = camera.physical_viewport_size {
            // How many times we can halve the resolution minus one so we don't go unnecessarily low
            let mip_count = bloom.max_mip_dimension.ilog2().max(2) - 1;
            let mip_height_ratio = if viewport.y != 0 {
                bloom.max_mip_dimension as f32 / viewport.y as f32
            } else {
                0.
            };

            let mut usage = TextureUsages::RENDER_ATTACHMENT | TextureUsages::TEXTURE_BINDING;
            if bloom_can_be_done_in_compute(&render_device) {
                usage |= TextureUsages::STORAGE_BINDING;
            }

            let texture_descriptor = TextureDescriptor {
                label: Some("bloom_texture"),
                size: (viewport.as_vec2() * mip_height_ratio)
                    .round()
                    .as_uvec2()
                    .max(UVec2::ONE)
                    .to_extents(),
                mip_level_count: mip_count,
                sample_count: 1,
                dimension: TextureDimension::D2,
                format: BLOOM_TEXTURE_FORMAT,
                usage,
                view_formats: &[],
            };

            #[cfg(any(
                not(feature = "webgl"),
                not(target_arch = "wasm32"),
                feature = "webgpu"
            ))]
            let texture = texture_cache.get(&render_device, texture_descriptor);
            #[cfg(all(feature = "webgl", target_arch = "wasm32", not(feature = "webgpu")))]
            let texture: Vec<CachedTexture> = (0..mip_count)
                .map(|mip| {
                    texture_cache.get(
                        &render_device,
                        TextureDescriptor {
                            size: Extent3d {
                                width: (texture_descriptor.size.width >> mip).max(1),
                                height: (texture_descriptor.size.height >> mip).max(1),
                                depth_or_array_layers: 1,
                            },
                            mip_level_count: 1,
                            ..texture_descriptor.clone()
                        },
                    )
                })
                .collect();

            commands
                .entity(entity)
                .insert(BloomTexture { texture, mip_count });
        }
    }
}

#[derive(Component)]
pub struct BloomComputeBindGroups {
    cache_key: (TextureId, BufferId),
    compute_main_downsampling_bind_groups: BloomLevelComputeBindGroups,
    compute_main_upsampling_bind_groups: BloomLevelComputeBindGroups,
    raster_final_upsampling_bind_group: BindGroup,
    sampler: Sampler,
}

struct BloomLevelComputeBindGroups {
    input: BindGroup,
    output: BindGroup,
}

#[derive(Component)]
pub struct BloomRasterBindGroups {
    #[cfg(any(
        not(feature = "webgl"),
        not(target_arch = "wasm32"),
        feature = "webgpu"
    ))]
    cache_key: (TextureId, BufferId),
    #[cfg(all(feature = "webgl", target_arch = "wasm32", not(feature = "webgpu")))]
    cache_key: (Vec<TextureId>, BufferId),
    downsampling_bind_groups: Box<[BindGroup]>,
    upsampling_bind_groups: Box<[BindGroup]>,
    sampler: Sampler,
}

fn prepare_bloom_raster_bind_groups(
    mut commands: Commands,
    render_device: Res<RenderDevice>,
    downsampling_pipeline: Res<BloomRasterDownsamplingPipeline>,
    upsampling_pipeline: Res<BloomRasterUpsamplingPipeline>,
    views: Query<(Entity, &BloomTexture, Option<&BloomRasterBindGroups>)>,
    uniforms: Res<ComponentUniforms<BloomUniforms>>,
    pipeline_cache: Res<PipelineCache>,
) {
    let sampler = &downsampling_pipeline.sampler;

    for (entity, bloom_texture, bloom_bind_groups) in &views {
        #[cfg(any(
            not(feature = "webgl"),
            not(target_arch = "wasm32"),
            feature = "webgpu"
        ))]
        let cache_key = (
            bloom_texture.texture.texture.id(),
            uniforms.buffer().unwrap().id(),
        );
        #[cfg(all(feature = "webgl", target_arch = "wasm32", not(feature = "webgpu")))]
        let cache_key = (
            bloom_texture
                .texture
                .iter()
                .map(|tex| tex.texture.id())
                .collect(),
            uniforms.buffer().unwrap().id(),
        );

        if let Some(b) = bloom_bind_groups
            && b.cache_key == cache_key
        {
            continue;
        }

        let bind_group_count = bloom_texture.mip_count as usize - 1;

        let mut downsampling_bind_groups = Vec::with_capacity(bind_group_count);
        for mip in 1..bloom_texture.mip_count {
            downsampling_bind_groups.push(render_device.create_bind_group(
                "bloom_downsampling_raster_bind_group",
                &pipeline_cache.get_bind_group_layout(&downsampling_pipeline.bind_group_layout),
                &BindGroupEntries::sequential((
                    uniforms.binding().unwrap(),
                    sampler,
                    &bloom_texture.view(mip - 1),
                )),
            ));
        }

        let mut upsampling_bind_groups = Vec::with_capacity(bind_group_count);
        for mip in (0..bloom_texture.mip_count).rev() {
            upsampling_bind_groups.push(render_device.create_bind_group(
                "bloom_upsampling_raster_bind_group",
                &pipeline_cache.get_bind_group_layout(&upsampling_pipeline.bind_group_layout),
                &BindGroupEntries::sequential((
                    uniforms.binding().unwrap(),
                    sampler,
                    &bloom_texture.view(mip),
                )),
            ));
        }

        commands.entity(entity).insert(BloomRasterBindGroups {
            cache_key,
            downsampling_bind_groups: downsampling_bind_groups.into_boxed_slice(),
            upsampling_bind_groups: upsampling_bind_groups.into_boxed_slice(),
            sampler: sampler.clone(),
        });
    }
}

fn prepare_bloom_compute_bind_groups(
    mut commands: Commands,
    render_device: Res<RenderDevice>,
    downsampling_pipeline: Res<BloomComputeDownsamplingPipeline>,
    compute_upsampling_pipeline: Res<BloomComputeUpsamplingPipeline>,
    raster_upsampling_pipeline: Res<BloomRasterUpsamplingPipeline>,
    views: Query<(Entity, &BloomTexture, Option<&BloomComputeBindGroups>)>,
    uniforms: Res<ComponentUniforms<BloomUniforms>>,
    pipeline_cache: Res<PipelineCache>,
) {
    let sampler = &downsampling_pipeline.sampler;

    for (entity, bloom_texture, bloom_bind_groups) in &views {
        let cache_key = (
            bloom_texture.texture.texture.id(),
            uniforms.buffer().unwrap().id(),
        );

        if let Some(b) = bloom_bind_groups
            && b.cache_key == cache_key
        {
            continue;
        }

        let mip_views: Vec<_> = (0..bloom_texture.mip_count)
            .map(|mip_level| bloom_texture.view(mip_level))
            .collect();
        let mip_views: Vec<_> = mip_views.iter().map(|mip_view| &**mip_view).collect();

        let compute_main_downsampling_bind_groups = BloomLevelComputeBindGroups {
            input: render_device.create_bind_group(
                "bloom_main_downsampling_compute_input_bind_group",
                &pipeline_cache
                    .get_bind_group_layout(&downsampling_pipeline.input_bind_group_layout),
                &BindGroupEntries::single(uniforms.binding().unwrap()),
            ),
            output: render_device.create_bind_group(
                "bloom_main_downsampling_compute_output_bind_group",
                &pipeline_cache
                    .get_bind_group_layout(&downsampling_pipeline.output_bind_group_layout),
                &BindGroupEntries::single(&mip_views[..]),
            ),
        };

        let compute_main_upsampling_bind_groups = BloomLevelComputeBindGroups {
            input: render_device.create_bind_group(
                "bloom_main_upsampling_compute_input_bind_group",
                &pipeline_cache
                    .get_bind_group_layout(&compute_upsampling_pipeline.input_bind_group_layout),
                &BindGroupEntries::single(uniforms.binding().unwrap()),
            ),
            output: render_device.create_bind_group(
                "bloom_main_upsampling_compute_output_bind_group",
                &pipeline_cache
                    .get_bind_group_layout(&compute_upsampling_pipeline.output_bind_group_layout),
                &BindGroupEntries::single(&mip_views[..]),
            ),
        };

        let raster_final_upsampling_bind_group = render_device.create_bind_group(
            "bloom_final_upsampling_raster_bind_group",
            &pipeline_cache.get_bind_group_layout(&raster_upsampling_pipeline.bind_group_layout),
            &BindGroupEntries::sequential((
                uniforms.binding().unwrap(),
                sampler,
                &bloom_texture.view(0),
            )),
        );

        commands.entity(entity).insert(BloomComputeBindGroups {
            cache_key,
            compute_main_downsampling_bind_groups,
            compute_main_upsampling_bind_groups,
            raster_final_upsampling_bind_group,
            sampler: sampler.clone(),
        });
    }
}

/// Calculates blend intensities of blur pyramid levels
/// during the upsampling + compositing stage.
///
/// The function assumes all pyramid levels are upsampled and
/// blended into higher frequency ones using this function to
/// calculate blend levels every time. The final (highest frequency)
/// pyramid level in not blended into anything therefore this function
/// is not applied to it. As a result, the *mip* parameter of 0 indicates
/// the second-highest frequency pyramid level (in our case that is the
/// 0th mip of the bloom texture with the original image being the
/// actual highest frequency level).
///
/// Parameters:
/// * `mip` - the index of the lower frequency pyramid level (0 - `max_mip`, where 0 indicates highest frequency mip but not the highest frequency image).
/// * `max_mip` - the index of the lowest frequency pyramid level.
///
/// This function can be visually previewed for all values of *mip* (normalized) with tweakable
/// [`Bloom`] parameters on [Desmos graphing calculator](https://www.desmos.com/calculator/ncc8xbhzzl).
fn compute_blend_factor(bloom: &Bloom, mip: f32, max_mip: f32) -> f32 {
    let mut lf_boost =
        (1.0 - ops::powf(
            1.0 - (mip / max_mip),
            1.0 / (1.0 - bloom.low_frequency_boost_curvature),
        )) * bloom.low_frequency_boost;
    let high_pass_lq = 1.0
        - (((mip / max_mip) - bloom.high_pass_frequency) / bloom.high_pass_frequency)
            .clamp(0.0, 1.0);
    lf_boost *= match bloom.composite_mode {
        BloomCompositeMode::EnergyConserving => 1.0 - bloom.intensity,
        BloomCompositeMode::Additive => 1.0,
    };

    (bloom.intensity + lf_boost) * high_pass_lq
}

fn bloom_can_be_done_in_compute(render_device: &RenderDevice) -> bool {
    let limits = render_device.limits();
    limits.max_compute_workgroup_size_x >= 16 && limits.max_compute_workgroup_size_y >= 16
}

fn get_workgroup_counts(bloom_texture: &BloomTexture, mip_level: u32) -> UVec2 {
    (uvec2(
        bloom_texture.texture.texture.width() >> mip_level,
        bloom_texture.texture.texture.height() >> mip_level,
    )
    .max(UVec2::splat(1))
        + 15)
        / 16
}

#[derive(Clone, Copy, Pod, Zeroable)]
#[repr(C)]
struct GpuBlendImmediates {
    input_mip: u32,
    output_mip: u32,
    src_blend_factor: f32,
    dest_blend_factor: f32,
}
