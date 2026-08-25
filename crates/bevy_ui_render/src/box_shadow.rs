//! Box shadows rendering

use core::{hash::Hash, ops::Range};

use bevy_app::prelude::*;
use bevy_asset::*;
use bevy_camera::visibility::InheritedVisibility;
use bevy_color::{Alpha, ColorToComponents, LinearRgba};
use bevy_ecs::prelude::*;
use bevy_ecs::{prelude::Component, system::*};
use bevy_math::{vec2, Affine2, FloatOrd, Vec2, Vec4};
use bevy_mesh::VertexBufferLayout;
use bevy_render::render_asset::{RenderAsset, RenderAssets};
use bevy_render::sync_world::{MainEntity, MainEntityHashSet};
use bevy_render::texture::GpuImage;
use bevy_render::{
    render_phase::*,
    render_resource::{binding_types::uniform_buffer, *},
    view::*,
    Extract, ExtractSchedule, Render, RenderSystems,
};
use bevy_render::{GpuResourceAppExt, RenderApp, RenderStartup};
use bevy_shader::{Shader, ShaderDefVal};
use bevy_ui::{
    BoxShadow, CalculatedClip, ComputedNode, ComputedStackIndex, ComputedUiRenderTargetInfo,
    ComputedUiTargetCamera, ResolvedBorderRadius, UiGlobalTransform, Val,
};
use bevy_utils::default;
use bytemuck::{Pod, Zeroable};

use crate::{
    prepare_uinodes, queue_ui_items, wipe_phase_items_if_camera_component_changed,
    BoxShadowSamples, CachedCameraView, ChangedUiObject, DrawUiRenderObject, RenderUiSystems,
    SetUiViewBindGroup, TransparentUi, UiCameraMap, UiMeta, UiRenderObject, UiRenderObjects,
};

use super::{stack_z_offsets, QUAD_VERTEX_POSITIONS};

/// A plugin that enables the rendering of box shadows.
pub struct BoxShadowPlugin;

impl Plugin for BoxShadowPlugin {
    fn build(&self, app: &mut App) {
        embedded_asset!(app, "box_shadow.wesl");

        if let Some(render_app) = app.get_sub_app_mut(RenderApp) {
            render_app
                .add_render_command::<TransparentUi, DrawBoxShadows>()
                .init_resource::<ExtractedBoxShadows>()
                .init_gpu_resource::<UiMeta<ExtractedBoxShadow>>()
                .init_gpu_resource::<SpecializedRenderPipelines<BoxShadowPipeline>>()
                .add_systems(RenderStartup, init_box_shadow_pipeline)
                .add_systems(
                    ExtractSchedule,
                    (
                        extract_shadows.in_set(RenderUiSystems::ExtractBoxShadows),
                        wipe_phase_items_if_camera_component_changed::<
                            ExtractedBoxShadow,
                            BoxShadowSamples,
                        >
                            .in_set(
                                RenderUiSystems::ExtractWipePhaseItemsIfCameraComponentsChanged,
                            ),
                    ),
                )
                .add_systems(
                    Render,
                    (
                        queue_ui_items::<ExtractedBoxShadow>.in_set(RenderSystems::Queue),
                        prepare_uinodes::<ExtractedBoxShadow>
                            .in_set(RenderSystems::PrepareBindGroups),
                    ),
                );
        }
    }
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct BoxShadowVertex {
    position: [f32; 3],
    uvs: [f32; 2],
    vertex_color: [f32; 4],
    size: [f32; 2],
    radius: [[f32; 4]; 2],
    blur: f32,
    bounds: [f32; 2],
}

#[derive(Component)]
pub struct UiShadowsBatch {
    pub range: Range<u32>,
    pub camera: Entity,
}

#[derive(Clone, Default)]
pub struct BoxShadowInstanceData {
    color: Vec4,
    size: Vec2,
    radius: [Vec4; 2],
    blur: f32,
    bounds: Vec2,
}

#[derive(Resource)]
pub struct BoxShadowPipeline {
    pub view_layout: BindGroupLayoutDescriptor,
    pub shader: Handle<Shader>,
}

pub fn init_box_shadow_pipeline(mut commands: Commands, asset_server: Res<AssetServer>) {
    let view_layout = BindGroupLayoutDescriptor::new(
        "box_shadow_view_layout",
        &BindGroupLayoutEntries::single(
            ShaderStages::VERTEX_FRAGMENT,
            uniform_buffer::<ViewUniform>(true),
        ),
    );

    commands.insert_resource(BoxShadowPipeline {
        view_layout,
        shader: load_embedded_asset!(asset_server.as_ref(), "box_shadow.wesl"),
    });
}

#[derive(Clone, Copy, Hash, PartialEq, Eq)]
pub struct BoxShadowPipelineKey {
    pub target_format: TextureFormat,
    /// Number of samples, a higher value results in better quality shadows.
    pub samples: u32,
}

impl SpecializedRenderPipeline for BoxShadowPipeline {
    type Key = BoxShadowPipelineKey;

    fn specialize(&self, key: Self::Key) -> RenderPipelineDescriptor {
        let vertex_layout = VertexBufferLayout::from_vertex_formats(
            VertexStepMode::Vertex,
            vec![
                // position
                VertexFormat::Float32x3,
                // uv
                VertexFormat::Float32x2,
                // color
                VertexFormat::Float32x4,
                // target rect size
                VertexFormat::Float32x2,
                // corner radius x values (top left, top right, bottom right, bottom left)
                VertexFormat::Float32x4,
                // corner radius y values (top left, top right, bottom right, bottom left)
                VertexFormat::Float32x4,
                // blur radius
                VertexFormat::Float32,
                // outer size
                VertexFormat::Float32x2,
            ],
        );
        let shader_defs = vec![ShaderDefVal::UInt("SHADOW_SAMPLES".into(), key.samples)];

        RenderPipelineDescriptor {
            vertex: VertexState {
                shader: self.shader.clone(),
                shader_defs: shader_defs.clone(),
                buffers: vec![vertex_layout],
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
            layout: vec![self.view_layout.clone()],
            label: Some("box_shadow_pipeline".into()),
            ..default()
        }
    }
}

/// Description of a shadow to be sorted and queued for rendering
pub struct ExtractedBoxShadow {
    pub stack_index: u32,
    pub transform: Affine2,
    pub bounds: Vec2,
    pub clip: Option<CalculatedClip>,
    pub color: LinearRgba,
    pub radius: ResolvedBorderRadius,
    pub blur_radius: f32,
    pub size: Vec2,
}

impl UiRenderObject for ExtractedBoxShadow {
    type DrawFunctions = DrawBoxShadows;
    type ViewPipelineKeyBuilder = UiBoxShadowViewPipelineKeyBuilder;
    type ViewQueryData = Option<&'static BoxShadowSamples>;
    type SpecializedRenderPipeline = BoxShadowPipeline;
    type PipelineKeySystemParam = ();
    type Vertex = BoxShadowVertex;
    type InstanceData = BoxShadowInstanceData;
    type TexturedGpuAsset = GpuImage;

    fn get_sort_key(&self) -> FloatOrd {
        FloatOrd(self.stack_index as f32 + stack_z_offsets::BOX_SHADOW)
    }

    fn create_view_pipeline_key_builder<'w, 's>(
        box_shadow_samples: Option<&BoxShadowSamples>,
    ) -> Self::ViewPipelineKeyBuilder {
        UiBoxShadowViewPipelineKeyBuilder {
            box_shadow_samples: box_shadow_samples.cloned(),
        }
    }

    fn create_pipeline_key(
        &self,
        cached_camera_view: &CachedCameraView<Self::ViewPipelineKeyBuilder>,
        _: &mut SystemParamItem<Self::PipelineKeySystemParam>,
    ) -> Option<BoxShadowPipelineKey> {
        Some(BoxShadowPipelineKey {
            target_format: cached_camera_view.extracted_view.target_format,
            samples: cached_camera_view
                .pipeline_key_builder
                .box_shadow_samples
                .unwrap_or_default()
                .0,
        })
    }

    fn view_bind_group_layout(
        pipeline: &Self::SpecializedRenderPipeline,
    ) -> &BindGroupLayoutDescriptor {
        &pipeline.view_layout
    }

    fn clip(&self) -> Option<&CalculatedClip> {
        self.clip.as_ref()
    }

    fn populate_quad(
        &self,
        out_quad: &mut crate::UiQuad<Self::InstanceData>,
        quad_index: usize,
        _: &RenderAssets<Self::TexturedGpuAsset>,
        _: &AssetId<<Self::TexturedGpuAsset as RenderAsset>::SourceAsset>,
    ) {
        debug_assert_eq!(quad_index, 0);

        let rect_size = self.bounds;

        // Specify the corners of the node
        let positions =
            QUAD_VERTEX_POSITIONS.map(|pos| self.transform.transform_point2(pos * rect_size));

        let uvs = [
            Vec2::ZERO,
            Vec2::new(self.bounds.x, 0.),
            self.bounds,
            Vec2::new(0., self.bounds.y),
        ]
        .map(|pos| pos / self.bounds);

        out_quad.instance_data = BoxShadowInstanceData {
            color: self.color.to_vec4(),
            size: self.size,
            radius: self.radius.into(),
            blur: self.blur_radius,
            bounds: rect_size,
        };

        for (&mut (ref mut out_position, ref mut out_uvs), (position, uv)) in out_quad
            .vertices
            .iter_mut()
            .zip(positions.iter().zip(uvs.iter()))
        {
            *out_position = *position;
            out_uvs.uv = *uv;
            out_uvs.point = default();
        }
    }

    fn create_vertex(
        quad: &crate::UiQuad<Self::InstanceData>,
        position: Vec2,
        uvs: &crate::UiQuadInterpolants,
    ) -> Self::Vertex {
        BoxShadowVertex {
            position: position.extend(0.0).into(),
            uvs: uvs.uv.into(),
            vertex_color: quad.instance_data.color.into(),
            size: quad.instance_data.size.into(),
            radius: quad.instance_data.radius.map(Into::into),
            blur: quad.instance_data.blur,
            bounds: quad.instance_data.bounds.into(),
        }
    }
}

/// List of extracted shadows to be sorted and queued for rendering
pub type ExtractedBoxShadows = UiRenderObjects<ExtractedBoxShadow>;

pub fn extract_shadows(
    mut commands: Commands,
    mut extracted_box_shadows: ResMut<ExtractedBoxShadows>,
    box_shadow_query: Extract<
        Query<
            (
                Entity,
                &ComputedNode,
                &ComputedStackIndex,
                &UiGlobalTransform,
                &InheritedVisibility,
                &BoxShadow,
                Option<&CalculatedClip>,
                &ComputedUiTargetCamera,
                &ComputedUiRenderTargetInfo,
            ),
            Or<(
                Changed<ComputedNode>,
                Changed<ComputedStackIndex>,
                Changed<UiGlobalTransform>,
                Changed<InheritedVisibility>,
                Changed<BoxShadow>,
                Changed<CalculatedClip>,
                Changed<ComputedUiTargetCamera>,
                Changed<ComputedUiRenderTargetInfo>,
            )>,
        >,
    >,
    unfiltered_box_shadow_query: Extract<
        Query<(
            Entity,
            &ComputedNode,
            &ComputedStackIndex,
            &UiGlobalTransform,
            &InheritedVisibility,
            &BoxShadow,
            Option<&CalculatedClip>,
            &ComputedUiTargetCamera,
            &ComputedUiRenderTargetInfo,
        )>,
    >,
    camera_map: Extract<UiCameraMap>,
    (
        mut removed_computed_node_query,
        mut removed_computed_stack_index_query,
        mut removed_ui_global_transform_query,
        mut removed_inherited_visibility_query,
        mut removed_box_shadow_query,
        mut removed_calculated_clip_query,
        mut removed_computed_ui_target_camera_query,
        mut removed_computed_ui_render_target_info_query,
    ): (
        Extract<RemovedComponents<ComputedNode>>,
        Extract<RemovedComponents<ComputedStackIndex>>,
        Extract<RemovedComponents<UiGlobalTransform>>,
        Extract<RemovedComponents<InheritedVisibility>>,
        Extract<RemovedComponents<BoxShadow>>,
        Extract<RemovedComponents<CalculatedClip>>,
        Extract<RemovedComponents<ComputedUiTargetCamera>>,
        Extract<RemovedComponents<ComputedUiRenderTargetInfo>>,
    ),
    mut nodes_processed_this_frame: Local<MainEntityHashSet>,
) {
    nodes_processed_this_frame.clear();
    extracted_box_shadows.changed.clear();

    let mut mapping = camera_map.get_mapper();

    for (entity, uinode, stack_index, transform, visibility, box_shadow, clip, camera, target) in
        box_shadow_query.iter().chain(
            removed_calculated_clip_query
                .read()
                .filter_map(|entity| unfiltered_box_shadow_query.get(entity).ok()),
        )
    {
        let main_entity = MainEntity::from(entity);

        // If there were any previous box shadows for this entity, despawn them
        // and record them as changed so the render phase entry can be removed.
        if let Some((prev_camera_entity, mut shadows)) =
            extracted_box_shadows.objects.remove(&main_entity)
        {
            let changed = extracted_box_shadows
                .changed
                .entry(main_entity)
                .or_default();
            for (render_entity, _) in shadows.drain(..) {
                commands.entity(render_entity).despawn();
                changed.push(ChangedUiObject {
                    render_entity,
                    camera_entity: prev_camera_entity,
                });
            }
        }

        // Skip if no visible shadows
        if !visibility.get() || box_shadow.is_empty() || uinode.is_empty() {
            continue;
        }

        let Some(extracted_camera_entity) = mapping.map(camera) else {
            continue;
        };
        if let Some((camera_entity, _)) = extracted_box_shadows.objects.get_mut(&main_entity) {
            *camera_entity = extracted_camera_entity;
        }

        let ui_physical_viewport_size = target.physical_size().as_vec2();
        let scale_factor = target.scale_factor();

        for drop_shadow in box_shadow.iter() {
            if drop_shadow.color.is_fully_transparent() {
                continue;
            }

            let resolve_val = |val, base, scale_factor| match val {
                Val::Auto => 0.,
                Val::Px(px) => px * scale_factor,
                Val::Percent(percent) => percent / 100. * base,
                Val::Vw(percent) => percent / 100. * ui_physical_viewport_size.x,
                Val::Vh(percent) => percent / 100. * ui_physical_viewport_size.y,
                Val::VMin(percent) => percent / 100. * ui_physical_viewport_size.min_element(),
                Val::VMax(percent) => percent / 100. * ui_physical_viewport_size.max_element(),
                Val::Em(em) => em * uinode.em_size.0 * scale_factor,
                Val::Rem(rem) => rem * uinode.rem_size.0 * scale_factor,
            };

            let spread_x = resolve_val(drop_shadow.spread_radius, uinode.size().x, scale_factor);
            let spread_ratio = (spread_x + uinode.size().x) / uinode.size().x;

            let spread = vec2(spread_x, uinode.size().y * spread_ratio - uinode.size().y);

            let blur_radius = resolve_val(drop_shadow.blur_radius, uinode.size().x, scale_factor);
            let offset = vec2(
                resolve_val(drop_shadow.x_offset, uinode.size().x, scale_factor),
                resolve_val(drop_shadow.y_offset, uinode.size().y, scale_factor),
            );

            let shadow_size = uinode.size() + spread;
            if shadow_size.cmple(Vec2::ZERO).any() {
                continue;
            }

            nodes_processed_this_frame.insert(main_entity);

            let radius = ResolvedBorderRadius {
                top_left: uinode.border_radius.top_left * spread_ratio,
                top_right: uinode.border_radius.top_right * spread_ratio,
                bottom_left: uinode.border_radius.bottom_left * spread_ratio,
                bottom_right: uinode.border_radius.bottom_right * spread_ratio,
            };

            extracted_box_shadows.add(
                &mut commands,
                main_entity,
                extracted_camera_entity,
                ExtractedBoxShadow {
                    stack_index: stack_index.0,
                    transform: Affine2::from(transform) * Affine2::from_translation(offset),
                    color: drop_shadow.color.into(),
                    bounds: shadow_size + 6. * blur_radius,
                    clip: clip.cloned(),
                    radius,
                    blur_radius,
                    size: shadow_size,
                },
            );
        }
    }

    // Only remove the render-world data if we didn't handle the node above.
    // It's possible that a relevant component was removed and added in the same
    // frame.
    for main_entity in removed_computed_node_query
        .read()
        .chain(removed_computed_stack_index_query.read())
        .chain(removed_ui_global_transform_query.read())
        .chain(removed_inherited_visibility_query.read())
        .chain(removed_box_shadow_query.read())
        .chain(removed_computed_ui_target_camera_query.read())
        .chain(removed_computed_ui_render_target_info_query.read())
    {
        let main_entity = MainEntity::from(main_entity);
        if nodes_processed_this_frame.contains(&main_entity) {
            continue;
        }
        let Some((prev_camera_entity, mut extracted_nodes)) =
            extracted_box_shadows.objects.remove(&main_entity)
        else {
            continue;
        };
        let changed = extracted_box_shadows
            .changed
            .entry(main_entity)
            .or_default();
        for (render_entity, _) in extracted_nodes.drain(..) {
            commands.entity(render_entity).despawn();
            changed.push(ChangedUiObject {
                render_entity,
                camera_entity: prev_camera_entity,
            });
        }
    }
}

/// Information that the box shadow renderer needs from each view to construct
/// the pipeline key.
pub struct UiBoxShadowViewPipelineKeyBuilder {
    /// The number of samples that this view requests to render box shadows
    /// with.
    box_shadow_samples: Option<BoxShadowSamples>,
}

pub type DrawBoxShadows = (
    SetItemPipeline,
    SetUiViewBindGroup<ExtractedBoxShadow, 0>,
    DrawUiRenderObject<ExtractedBoxShadow>,
);
