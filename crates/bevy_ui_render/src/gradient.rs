use core::{
    f32::consts::{FRAC_PI_2, TAU},
    hash::Hash,
};

use super::shader_flags::BORDER_ALL;
use crate::*;
use bevy_asset::*;
use bevy_color::{ColorToComponents, Hsla, Hsva, LinearRgba, Okhsla, Oklaba, Oklcha, Srgba};
use bevy_ecs::system::*;
use bevy_math::{
    ops::{cos, sin},
    vec2, vec4, FloatOrd, Rect, Vec2, Vec4,
};
use bevy_math::{Affine2, Vec2Swizzles};
use bevy_mesh::VertexBufferLayout;
use bevy_render::{
    render_phase::*,
    render_resource::{
        binding_types::{storage_buffer_read_only_sized, uniform_buffer},
        *,
    },
    view::*,
    Extract, ExtractSchedule, Render, RenderSystems,
};
use bevy_render::{GpuResourceAppExt, RenderStartup};
use bevy_shader::Shader;
use bevy_sprite::BorderRect;
use bevy_text::{EmSize, RemSize};
use bevy_ui::{
    BackgroundGradient, BorderGradient, ColorStop, ComputedStackIndex, ComputedUiRenderTargetInfo,
    ConicGradient, Gradient, InterpolationColorSpace, LinearGradient, RadialGradient,
    ResolvedBorderRadius, Val,
};
use bevy_utils::default;
use bitflags::bitflags;
use bytemuck::{Pod, Zeroable};

pub struct GradientPlugin;

impl Plugin for GradientPlugin {
    fn build(&self, app: &mut App) {
        embedded_asset!(app, "gradient.wesl");

        if let Some(render_app) = app.get_sub_app_mut(RenderApp) {
            render_app
                .add_render_command::<TransparentUi, DrawGradientFns>()
                .init_resource::<ExtractedGradients>()
                .init_gpu_resource::<UiMeta<ExtractedGradient>>()
                .init_gpu_resource::<SpecializedRenderPipelines<GradientPipeline>>()
                .add_systems(RenderStartup, init_gradient_pipeline)
                .add_systems(
                    ExtractSchedule,
                    (
                        extract_gradients
                            .in_set(RenderUiSystems::ExtractGradient)
                            .after(extract_uinode_background_colors),
                        wipe_phase_items_if_camera_component_changed::<
                            ExtractedGradient,
                            UiAntiAlias,
                        >
                            .in_set(
                                RenderUiSystems::ExtractWipePhaseItemsIfCameraComponentsChanged,
                            ),
                    ),
                )
                .add_systems(
                    Render,
                    (
                        queue_ui_items::<ExtractedGradient>.in_set(RenderSystems::Queue),
                        prepare_uinodes::<ExtractedGradient>
                            .in_set(RenderSystems::PrepareBindGroups),
                    ),
                );
        }
    }
}

#[derive(Resource)]
pub struct GradientPipeline {
    pub view_layout: BindGroupLayoutDescriptor,
    pub instances_layout: BindGroupLayoutDescriptor,
    pub shader: Handle<Shader>,
}

pub fn init_gradient_pipeline(mut commands: Commands, asset_server: Res<AssetServer>) {
    let view_layout = BindGroupLayoutDescriptor::new(
        "ui_gradient_view_layout",
        &BindGroupLayoutEntries::single(
            ShaderStages::VERTEX_FRAGMENT,
            uniform_buffer::<ViewUniform>(true),
        ),
    );

    let instances_layout = BindGroupLayoutDescriptor::new(
        "ui_gradient_instances_layout",
        &BindGroupLayoutEntries::single(
            ShaderStages::VERTEX,
            storage_buffer_read_only_sized(false, None),
        ),
    );

    commands.insert_resource(GradientPipeline {
        view_layout,
        instances_layout,
        shader: load_embedded_asset!(asset_server.as_ref(), "gradient.wesl"),
    });
}

pub fn compute_gradient_line_length(angle: f32, size: Vec2) -> f32 {
    let center = 0.5 * size;
    let v = Vec2::new(sin(angle), -cos(angle));

    let (pos_corner, neg_corner) = if v.x >= 0.0 && v.y <= 0.0 {
        (size.with_y(0.), size.with_x(0.))
    } else if v.x >= 0.0 && v.y > 0.0 {
        (size, Vec2::ZERO)
    } else if v.x < 0.0 && v.y <= 0.0 {
        (Vec2::ZERO, size)
    } else {
        (size.with_x(0.), size.with_y(0.))
    };

    let t_pos = (pos_corner - center).dot(v);
    let t_neg = (neg_corner - center).dot(v);

    (t_pos - t_neg).abs()
}

#[derive(Clone, Copy, Hash, PartialEq, Eq)]
pub struct UiGradientPipelineKey {
    color_space: InterpolationColorSpace,
    pub target_format: TextureFormat,
    flags: UiGradientPipelineKeyFlags,
}

bitflags! {
    #[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
    struct UiGradientPipelineKeyFlags: u8 {
        const ANTI_ALIAS = 1 << 0;
        const RETAINED_INSTANCES = 1 << 1;
    }
}

impl SpecializedRenderPipeline for GradientPipeline {
    type Key = UiGradientPipelineKey;

    fn specialize(&self, key: Self::Key) -> RenderPipelineDescriptor {
        let vertex_layout = VertexBufferLayout::from_vertex_formats(
            VertexStepMode::Vertex,
            vec![
                // position
                VertexFormat::Float32x2,
            ],
        );
        let instance_layout = VertexBufferLayout::from_vertex_formats(
            VertexStepMode::Instance,
            if key
                .flags
                .contains(UiGradientPipelineKeyFlags::RETAINED_INSTANCES)
            {
                vec![
                    // instance index
                    VertexFormat::Uint32,
                ]
            } else {
                vec![
                    // transform
                    VertexFormat::Float32x4,
                    // border radius x values (top left, top right, bottom right, bottom left)
                    VertexFormat::Float32x4,
                    // border radius y values (top left, top right, bottom right, bottom left)
                    VertexFormat::Float32x4,
                    // border
                    VertexFormat::Float32x4,
                    // start color
                    VertexFormat::Float32x4,
                    // end color
                    VertexFormat::Float32x4,
                    // transform translation
                    VertexFormat::Float32x2,
                    // size
                    VertexFormat::Float32x2,
                    // start_point
                    VertexFormat::Float32x2,
                    // dir
                    VertexFormat::Float32x2,
                    // start_len
                    VertexFormat::Float32,
                    // end_len
                    VertexFormat::Float32,
                    // hint
                    VertexFormat::Float32,
                    // flags
                    VertexFormat::Uint32,
                ]
            },
        )
        .offset_locations_by(1);
        let color_space = match key.color_space {
            InterpolationColorSpace::Oklaba => "IN_OKLAB",
            InterpolationColorSpace::Oklcha => "IN_OKLCH",
            InterpolationColorSpace::OklchaLong => "IN_OKLCH_LONG",
            InterpolationColorSpace::Okhsla => "IN_OKHSL",
            InterpolationColorSpace::OkhslaLong => "IN_OKHSL_LONG",
            InterpolationColorSpace::Srgba => "IN_SRGB",
            InterpolationColorSpace::LinearRgba => "IN_LINEAR_RGB",
            InterpolationColorSpace::Hsla => "IN_HSL",
            InterpolationColorSpace::HslaLong => "IN_HSL_LONG",
            InterpolationColorSpace::Hsva => "IN_HSV",
            InterpolationColorSpace::HsvaLong => "IN_HSV_LONG",
        };

        let mut shader_defs = vec![color_space.into()];
        if key.flags.contains(UiGradientPipelineKeyFlags::ANTI_ALIAS) {
            shader_defs.push("ANTI_ALIAS".into());
        }
        if key
            .flags
            .contains(UiGradientPipelineKeyFlags::RETAINED_INSTANCES)
        {
            shader_defs.push("RETAINED_INSTANCES".into());
        }

        let mut layout = vec![self.view_layout.clone()];
        if key
            .flags
            .contains(UiGradientPipelineKeyFlags::RETAINED_INSTANCES)
        {
            layout.push(self.instances_layout.clone());
        }

        RenderPipelineDescriptor {
            vertex: VertexState {
                shader: self.shader.clone(),
                shader_defs: shader_defs.clone(),
                buffers: vec![vertex_layout, instance_layout],
                ..default()
            },
            fragment: Some(FragmentState {
                shader: self.shader.clone(),
                shader_defs,
                targets: vec![Some(ColorTargetState {
                    format: key.target_format,
                    blend: Some(BlendState::ALPHA_BLENDING),
                    write_mask: ColorWrites::ALL,
                })],
                ..default()
            }),
            layout,
            label: Some("ui_gradient_pipeline".into()),
            ..default()
        }
    }
}

pub enum ResolvedGradient {
    Linear { angle: f32 },
    Conic { center: Vec2, start: f32 },
    Radial { center: Vec2, size: Vec2 },
}

pub struct ExtractedGradient {
    pub stack_index: u32,
    pub transform: Affine2,
    pub rect: Rect,
    pub clip: Option<CalculatedClip>,
    pub stops: Vec<(LinearRgba, f32, f32)>,
    pub node_type: NodeType,
    /// Border radius of the UI node.
    /// Ordering: top left, top right, bottom right, bottom left.
    pub border_radius: ResolvedBorderRadius,
    /// Border thickness of the UI node.
    /// Ordering: left, top, right, bottom.
    pub border: BorderRect,
    pub resolved_gradient: ResolvedGradient,
    pub color_space: InterpolationColorSpace,
    pub rendered_stop_indices: Vec<u32>,
}

impl UiRenderObject for ExtractedGradient {
    type DrawFunctions = DrawGradientFns;
    type ViewQueryData = Option<&'static UiAntiAlias>;
    type SpecializedRenderPipeline = GradientPipeline;
    type ViewPipelineKeyBuilder = UiGradientViewPipelineKeyBuilder;
    type PipelineKeySystemParam = SRes<UiMeta<ExtractedGradient>>;
    type InstanceData = UiGradientInstanceData;
    type TexturedGpuAsset = GpuImage;

    fn get_sort_key(&self) -> FloatOrd {
        FloatOrd(
            self.stack_index as f32
                + match self.node_type {
                    NodeType::Rect | NodeType::Inverted => stack_z_offsets::GRADIENT,
                    NodeType::Border(_) => stack_z_offsets::BORDER_GRADIENT,
                },
        )
    }

    fn create_view_pipeline_key_builder<'w, 's>(
        anti_alias: Option<&UiAntiAlias>,
    ) -> Self::ViewPipelineKeyBuilder {
        UiGradientViewPipelineKeyBuilder {
            anti_alias: anti_alias.cloned(),
        }
    }

    fn create_pipeline_key(
        &self,
        cached_camera_view: &CachedCameraView<Self::ViewPipelineKeyBuilder>,
        ui_meta: &mut SystemParamItem<Self::PipelineKeySystemParam>,
    ) -> Option<UiGradientPipelineKey> {
        let mut flags = UiGradientPipelineKeyFlags::empty();
        if matches!(
            cached_camera_view.pipeline_key_builder.anti_alias,
            None | Some(UiAntiAlias::On)
        ) {
            flags.insert(UiGradientPipelineKeyFlags::ANTI_ALIAS);
        }
        if matches!(ui_meta.instances, UiInstances::Retained { .. }) {
            flags.insert(UiGradientPipelineKeyFlags::RETAINED_INSTANCES);
        }

        Some(UiGradientPipelineKey {
            flags,
            color_space: self.color_space,
            target_format: cached_camera_view.extracted_view.target_format,
        })
    }

    fn bind_group_layouts(
        pipeline: &Self::SpecializedRenderPipeline,
    ) -> UiRenderObjectBindGroupLayouts<'_> {
        UiRenderObjectBindGroupLayouts {
            view: &pipeline.view_layout,
            instances: &pipeline.instances_layout,
        }
    }

    fn clip(&self) -> Option<&CalculatedClip> {
        self.clip.as_ref()
    }

    fn quad_count(&self) -> usize {
        self.rendered_stop_indices.len()
    }

    fn populate_quad(
        &self,
        out_quad: &mut UiQuad<Self::InstanceData>,
        quad_index: usize,
        _: &RenderAssets<Self::TexturedGpuAsset>,
        _: &AssetId<<Self::TexturedGpuAsset as RenderAsset>::SourceAsset>,
    ) {
        let uinode_rect = self.rect;

        let rect_size = uinode_rect.size();

        // Specify the corners of the node
        let corner_points = QUAD_VERTEX_POSITIONS.map(|pos| pos * rect_size);
        out_quad.positions = corner_points.map(|pos| self.transform.transform_point2(pos));

        let mut flags = if let NodeType::Border(borders) = self.node_type {
            borders
        } else {
            0
        };

        let (g_start, g_dir, g_flags) = match self.resolved_gradient {
            ResolvedGradient::Linear { angle } => {
                let corner_index = (angle - FRAC_PI_2).rem_euclid(TAU) / FRAC_PI_2;
                (
                    corner_points[corner_index as usize],
                    // CSS angles increase in a clockwise direction
                    vec2(sin(angle), -cos(angle)),
                    0,
                )
            }
            ResolvedGradient::Conic { center, start } => {
                (center, vec2(start, 0.), shader_flags::CONIC)
            }
            ResolvedGradient::Radial { center, size } => (
                center,
                Vec2::splat(if size.y != 0. { size.x / size.y } else { 1. }),
                shader_flags::RADIAL,
            ),
        };

        flags |= g_flags;

        let stop_index = self.rendered_stop_indices[quad_index] as usize;
        let mut start_stop = self.stops[stop_index];
        let end_stop = self.stops[stop_index + 1];
        if start_stop.1 == end_stop.1 && stop_index == self.stops.len() - 2 && 0 < quad_index {
            start_stop.0 = LinearRgba::NONE;
        }
        let start_color = convert_color_to_space(start_stop.0, self.color_space);
        let end_color = convert_color_to_space(end_stop.0, self.color_space);
        let mut stop_flags = flags;
        if 0. < start_stop.1 && (stop_index == 0 || quad_index == 0) {
            stop_flags |= shader_flags::FILL_START;
        }
        if stop_index == self.stops.len() - 2 {
            stop_flags |= shader_flags::FILL_END;
        }

        out_quad.instance_data = UiGradientInstanceData {
            world_from_local: pack_transform(self.transform, rect_size),
            flags: stop_flags,
            radius: self.border_radius.into(),
            border: vec4(
                self.border.min_inset.x,
                self.border.min_inset.y,
                self.border.max_inset.x,
                self.border.max_inset.y,
            ),
            size: rect_size.xy(),
            g_start,
            g_dir,
            start_color: start_color.into(),
            start_len: start_stop.1,
            end_len: end_stop.1,
            end_color: end_color.into(),
            hint: start_stop.2,
            translation: self.transform.translation,
        };
    }
}

/// A render-world resource that stores all gradients in the scene.
pub type ExtractedGradients = UiRenderObjects<ExtractedGradient>;

// Interpolate implicit stops (where position is `f32::NAN`)
// If the first and last stops are implicit set them to the `min` and `max` values
// so that we always have explicit start and end points to interpolate between.
fn interpolate_color_stops(stops: &mut [(LinearRgba, f32, f32)], min: f32, max: f32) {
    if stops[0].1.is_nan() {
        stops[0].1 = min;
    }
    if stops.last().unwrap().1.is_nan() {
        stops.last_mut().unwrap().1 = max;
    }

    let mut i = 1;

    while i < stops.len() - 1 {
        let point = stops[i].1;
        if point.is_nan() {
            let start = i;
            let mut end = i + 1;
            while end < stops.len() - 1 && stops[end].1.is_nan() {
                end += 1;
            }
            let start_point = stops[start - 1].1;
            let end_point = stops[end].1;
            let steps = end - start;
            let step = (end_point - start_point) / (steps + 1) as f32;
            for j in 0..steps {
                stops[i + j].1 = start_point + step * (j + 1) as f32;
            }
            i = end;
        }
        i += 1;
    }
}

fn compute_color_stops(
    stops: &[ColorStop],
    scale_factor: f32,
    length: f32,
    target_size: Vec2,
    scratch: &mut Vec<(LinearRgba, f32, f32)>,
    em_size: EmSize,
    rem_size: RemSize,
) -> Vec<(LinearRgba, f32, f32)> {
    let mut extracted_color_stops = vec![];

    // resolve the physical distances of explicit stops and sort them
    scratch.extend(stops.iter().filter_map(|stop| {
        stop.point
            .resolve(scale_factor, length, target_size, em_size, rem_size)
            .ok()
            .map(|physical_point| (stop.color.to_linear(), physical_point, stop.hint))
    }));
    scratch.sort_by_key(|(_, point, _)| FloatOrd(*point));

    let min = scratch
        .first()
        .map(|(_, min, _)| *min)
        .unwrap_or(0.)
        .min(0.);

    // get the position of the last explicit stop and use the full length of the gradient if no explicit stops
    let max = scratch
        .last()
        .map(|(_, max, _)| *max)
        .unwrap_or(length)
        .max(length);

    let mut sorted_stops_drain = scratch.drain(..);

    // Fill the extracted color stops buffer
    extracted_color_stops.extend(stops.iter().map(|stop| {
        if stop.point == Val::Auto {
            (stop.color.to_linear(), f32::NAN, stop.hint)
        } else {
            sorted_stops_drain.next().unwrap()
        }
    }));

    interpolate_color_stops(&mut extracted_color_stops, min, max);

    extracted_color_stops
}

pub fn extract_gradients(
    mut commands: Commands,
    mut extracted_gradients: ResMut<ExtractedGradients>,
    gradients_query: Extract<
        Query<
            (
                Entity,
                &ComputedNode,
                &ComputedStackIndex,
                &ComputedUiTargetCamera,
                &ComputedUiRenderTargetInfo,
                &UiGlobalTransform,
                &InheritedVisibility,
                Option<&CalculatedClip>,
                AnyOf<(&BackgroundGradient, &BorderGradient)>,
            ),
            Or<(
                Changed<ComputedNode>,
                Changed<ComputedStackIndex>,
                Changed<ComputedUiTargetCamera>,
                Changed<ComputedUiRenderTargetInfo>,
                Changed<UiGlobalTransform>,
                Changed<InheritedVisibility>,
                Changed<CalculatedClip>,
                Changed<BackgroundGradient>,
                Changed<BorderGradient>,
            )>,
        >,
    >,
    unfilitered_gradients_query: Extract<
        Query<(
            Entity,
            &ComputedNode,
            &ComputedStackIndex,
            &ComputedUiTargetCamera,
            &ComputedUiRenderTargetInfo,
            &UiGlobalTransform,
            &InheritedVisibility,
            Option<&CalculatedClip>,
            AnyOf<(&BackgroundGradient, &BorderGradient)>,
        )>,
    >,
    (
        mut removed_computed_node_query,
        mut removed_computed_stack_index_query,
        mut removed_computed_ui_target_camera_query,
        mut removed_computed_ui_render_target_info_query,
        mut removed_ui_global_transform_query,
        mut removed_inherited_visibility_query,
        mut removed_calculated_clip_query,
        mut removed_background_gradient_query,
        mut removed_border_gradient_query,
    ): (
        Extract<RemovedComponents<ComputedNode>>,
        Extract<RemovedComponents<ComputedStackIndex>>,
        Extract<RemovedComponents<ComputedUiTargetCamera>>,
        Extract<RemovedComponents<ComputedUiRenderTargetInfo>>,
        Extract<RemovedComponents<UiGlobalTransform>>,
        Extract<RemovedComponents<InheritedVisibility>>,
        Extract<RemovedComponents<CalculatedClip>>,
        Extract<RemovedComponents<BackgroundGradient>>,
        Extract<RemovedComponents<BorderGradient>>,
    ),
    camera_map: Extract<UiCameraMap>,
    mut nodes_processed_this_frame: Local<MainEntityHashSet>,
) {
    nodes_processed_this_frame.clear();
    extracted_gradients.changed.clear();
    let mut camera_mapper = camera_map.get_mapper();
    let mut sorted_stops = vec![];

    for (
        entity,
        uinode,
        stack_index,
        camera,
        target,
        transform,
        inherited_visibility,
        clip,
        (gradient, gradient_border),
    ) in gradients_query.iter().chain(
        removed_calculated_clip_query
            .read()
            .filter_map(|entity| unfilitered_gradients_query.get(entity).ok()),
    ) {
        let main_entity = MainEntity::from(entity);

        // If there were any previous gradients for this entity, despawn them
        // and record them as changed so the render phase entry can be removed.
        if let Some((prev_camera_entity, mut gradients)) =
            extracted_gradients.objects.remove(&main_entity)
        {
            let changed = extracted_gradients.changed.entry(main_entity).or_default();
            for (render_entity, _) in gradients.drain(..) {
                commands.entity(render_entity).despawn();
                changed.push(ChangedUiObject {
                    render_entity,
                    camera_entity: prev_camera_entity,
                });
            }
        }

        // Skip invisible images
        if !inherited_visibility.get() {
            continue;
        }

        let Some(extracted_camera_entity) = camera_mapper.map(camera) else {
            continue;
        };
        if let Some((camera_entity, _)) = extracted_gradients.objects.get_mut(&main_entity) {
            *camera_entity = extracted_camera_entity;
        }

        for (gradients, node_type) in [
            (gradient.map(|g| &g.0), NodeType::Rect),
            (gradient_border.map(|g| &g.0), NodeType::Border(BORDER_ALL)),
        ]
        .iter()
        .filter_map(|(g, n)| g.map(|g| (g, *n)))
        {
            for gradient in gradients.iter() {
                if gradient.is_empty() {
                    continue;
                }

                nodes_processed_this_frame.insert(main_entity);

                if let Some(color) = gradient.get_single() {
                    // With a single color stop there's no gradient, fill the node with the color
                    let length = compute_gradient_line_length(0.0, uinode.size);
                    let extracted_stops = compute_color_stops(
                        &[
                            ColorStop::new(color, Val::Percent(0.0)),
                            ColorStop::new(color, Val::Percent(100.0)),
                        ],
                        target.scale_factor(),
                        length,
                        target.physical_size().as_vec2(),
                        &mut sorted_stops,
                        uinode.em_size,
                        uinode.rem_size,
                    );
                    let rendered_stop_indices = calculate_rendered_stop_indices(&extracted_stops);
                    extracted_gradients.add(
                        &mut commands,
                        main_entity,
                        extracted_camera_entity,
                        ExtractedGradient {
                            stack_index: stack_index.0,
                            transform: transform.into(),
                            stops: extracted_stops,
                            rect: Rect {
                                min: Vec2::ZERO,
                                max: uinode.size,
                            },
                            clip: clip.cloned(),
                            node_type,
                            border_radius: uinode.border_radius,
                            border: uinode.border,
                            resolved_gradient: ResolvedGradient::Linear { angle: 0.0 },
                            color_space: gradient.get_color_space(),
                            rendered_stop_indices,
                        },
                    );
                    continue;
                }
                match gradient {
                    Gradient::Linear(LinearGradient {
                        color_space,
                        angle,
                        stops,
                    }) => {
                        let length = compute_gradient_line_length(*angle, uinode.size);

                        let extracted_stops = compute_color_stops(
                            stops,
                            target.scale_factor(),
                            length,
                            target.physical_size().as_vec2(),
                            &mut sorted_stops,
                            uinode.em_size,
                            uinode.rem_size,
                        );

                        let rendered_stop_indices =
                            calculate_rendered_stop_indices(&extracted_stops);

                        extracted_gradients.add(
                            &mut commands,
                            main_entity,
                            extracted_camera_entity,
                            ExtractedGradient {
                                stack_index: stack_index.0,
                                transform: transform.into(),
                                stops: extracted_stops,
                                rect: Rect {
                                    min: Vec2::ZERO,
                                    max: uinode.size,
                                },
                                clip: clip.cloned(),
                                node_type,
                                border_radius: uinode.border_radius,
                                border: uinode.border,
                                resolved_gradient: ResolvedGradient::Linear { angle: *angle },
                                color_space: *color_space,
                                rendered_stop_indices,
                            },
                        );
                    }
                    Gradient::Radial(RadialGradient {
                        color_space,
                        position: center,
                        shape,
                        stops,
                    }) => {
                        let c = center.resolve(
                            target.scale_factor(),
                            uinode.size,
                            target.physical_size().as_vec2(),
                            uinode.em_size,
                            uinode.rem_size,
                        );

                        let size = shape.resolve(
                            c,
                            target.scale_factor(),
                            uinode.size,
                            target.physical_size().as_vec2(),
                            uinode.em_size,
                            uinode.rem_size,
                        );

                        let length = size.x;

                        let computed_stops = compute_color_stops(
                            stops,
                            target.scale_factor(),
                            length,
                            target.physical_size().as_vec2(),
                            &mut sorted_stops,
                            uinode.em_size,
                            uinode.rem_size,
                        );

                        let rendered_stop_indices =
                            calculate_rendered_stop_indices(&computed_stops);

                        extracted_gradients.add(
                            &mut commands,
                            main_entity,
                            extracted_camera_entity,
                            ExtractedGradient {
                                stack_index: stack_index.0,
                                transform: transform.into(),
                                stops: computed_stops,
                                rect: Rect {
                                    min: Vec2::ZERO,
                                    max: uinode.size,
                                },
                                clip: clip.cloned(),
                                node_type,
                                border_radius: uinode.border_radius,
                                border: uinode.border,
                                resolved_gradient: ResolvedGradient::Radial { center: c, size },
                                color_space: *color_space,
                                rendered_stop_indices,
                            },
                        );
                    }
                    Gradient::Conic(ConicGradient {
                        color_space,
                        start,
                        position: center,
                        stops,
                    }) => {
                        let g_start = center.resolve(
                            target.scale_factor(),
                            uinode.size,
                            target.physical_size().as_vec2(),
                            uinode.em_size,
                            uinode.rem_size,
                        );

                        // sort the explicit stops
                        sorted_stops.extend(stops.iter().filter_map(|stop| {
                            stop.angle.map(|angle| {
                                (stop.color.to_linear(), angle.clamp(0., TAU), stop.hint)
                            })
                        }));
                        sorted_stops.sort_by_key(|(_, angle, _)| FloatOrd(*angle));
                        let mut sorted_stops_drain = sorted_stops.drain(..);

                        // fill the extracted stops buffer
                        let mut extracted_color_stops: Vec<_> = stops
                            .iter()
                            .map(|stop| {
                                if stop.angle.is_none() {
                                    (stop.color.to_linear(), f32::NAN, stop.hint)
                                } else {
                                    sorted_stops_drain.next().unwrap()
                                }
                            })
                            .collect();

                        interpolate_color_stops(&mut extracted_color_stops, 0., TAU);

                        let rendered_stop_indices =
                            calculate_rendered_stop_indices(&extracted_color_stops);

                        extracted_gradients.add(
                            &mut commands,
                            main_entity,
                            extracted_camera_entity,
                            ExtractedGradient {
                                stack_index: stack_index.0,
                                transform: transform.into(),
                                stops: extracted_color_stops,
                                rect: Rect {
                                    min: Vec2::ZERO,
                                    max: uinode.size,
                                },
                                clip: clip.cloned(),
                                node_type,
                                border_radius: uinode.border_radius,
                                border: uinode.border,
                                resolved_gradient: ResolvedGradient::Conic {
                                    start: *start,
                                    center: g_start,
                                },
                                color_space: *color_space,
                                rendered_stop_indices,
                            },
                        );
                    }
                }
            }
        }
    }

    // Only remove the render-world data if we didn't handle the node above.
    // It's possible that a relevant component was removed and added in the same
    // frame.
    for main_entity in removed_computed_node_query
        .read()
        .chain(removed_computed_stack_index_query.read())
        .chain(removed_computed_ui_target_camera_query.read())
        .chain(removed_computed_ui_render_target_info_query.read())
        .chain(removed_ui_global_transform_query.read())
        .chain(removed_inherited_visibility_query.read())
        .chain(removed_background_gradient_query.read())
        .chain(removed_border_gradient_query.read())
    {
        let main_entity = MainEntity::from(main_entity);
        if nodes_processed_this_frame.contains(&main_entity) {
            continue;
        }
        let Some((prev_camera_entity, mut extracted_nodes)) =
            extracted_gradients.objects.remove(&main_entity)
        else {
            continue;
        };
        let changed = extracted_gradients.changed.entry(main_entity).or_default();
        for (render_entity, _) in extracted_nodes.drain(..) {
            commands.entity(render_entity).despawn();
            changed.push(ChangedUiObject {
                render_entity,
                camera_entity: prev_camera_entity,
            });
        }
    }
}

/// Information from a view needed to construct a pipeline key for gradients.
pub struct UiGradientViewPipelineKeyBuilder {
    /// Whether anti-aliasing is requested for UI nodes in this view.
    anti_alias: Option<UiAntiAlias>,
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct UiGradientVertex {
    position: [f32; 3],
    uv: [f32; 2],
    flags: u32,
    radius: [[f32; 4]; 2],
    border: [f32; 4],
    size: [f32; 2],
    point: [f32; 2],
    g_start: [f32; 2],
    g_dir: [f32; 2],
    start_color: [f32; 4],
    start_len: f32,
    end_len: f32,
    end_color: [f32; 4],
    hint: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Default, Pod, Zeroable)]
pub struct UiGradientInstanceData {
    world_from_local: Vec4,
    radius: [Vec4; 2],
    border: Vec4,
    start_color: Vec4,
    end_color: Vec4,
    translation: Vec2,
    size: Vec2,
    g_start: Vec2,
    g_dir: Vec2,
    start_len: f32,
    end_len: f32,
    hint: f32,
    flags: u32,
}

fn convert_color_to_space(color: LinearRgba, space: InterpolationColorSpace) -> [f32; 4] {
    match space {
        InterpolationColorSpace::Oklaba => {
            let oklaba: Oklaba = color.into();
            [oklaba.lightness, oklaba.a, oklaba.b, oklaba.alpha]
        }
        InterpolationColorSpace::Oklcha | InterpolationColorSpace::OklchaLong => {
            let oklcha: Oklcha = color.into();
            [
                oklcha.lightness,
                oklcha.chroma,
                // The shader expects normalized hues
                oklcha.hue / 360.,
                oklcha.alpha,
            ]
        }
        InterpolationColorSpace::Okhsla | InterpolationColorSpace::OkhslaLong => {
            let okhsla: Okhsla = color.into();
            [
                okhsla.hue / 360.,
                okhsla.saturation,
                okhsla.lightness,
                okhsla.alpha,
            ]
        }
        InterpolationColorSpace::Srgba => {
            let srgba: Srgba = color.into();
            [srgba.red, srgba.green, srgba.blue, srgba.alpha]
        }
        InterpolationColorSpace::LinearRgba => color.to_f32_array(),
        InterpolationColorSpace::Hsla | InterpolationColorSpace::HslaLong => {
            let hsla: Hsla = color.into();
            // The shader expects normalized hues
            [hsla.hue / 360., hsla.saturation, hsla.lightness, hsla.alpha]
        }
        InterpolationColorSpace::Hsva | InterpolationColorSpace::HsvaLong => {
            let hsva: Hsva = color.into();
            // The shader expects normalized hues
            [hsva.hue / 360., hsva.saturation, hsva.value, hsva.alpha]
        }
    }
}

pub type DrawGradientFns = (
    SetItemPipeline,
    SetUiViewBindGroup<ExtractedGradient, 0>,
    DrawUiRenderObject<ExtractedGradient, 1>,
);

fn calculate_rendered_stop_indices(stops: &[(LinearRgba, f32, f32)]) -> Vec<u32> {
    let mut rendered_stop_indices = vec![];
    for stop_index in 0..(stops.len() - 1) {
        if stop_index + 2 == stops.len() || stops[stop_index].1 != stops[stop_index + 1].1 {
            rendered_stop_indices.push(stop_index as u32);
        }
    }
    rendered_stop_indices
}
