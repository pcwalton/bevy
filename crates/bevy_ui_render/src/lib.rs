#![expect(missing_docs, reason = "Not all docs are written yet, see #3492.")]
#![cfg_attr(docsrs, feature(doc_cfg))]
#![doc(
    html_logo_url = "https://bevyengine.org/assets/icon.png",
    html_favicon_url = "https://bevyengine.org/assets/icon.png"
)]

//! Provides rendering functionality for `bevy_ui`.

pub mod box_shadow;
pub mod clipping;
mod gradient;
mod image;
use bevy_ecs::query::QueryData;
use bevy_ecs::system::lifetimeless::SRes;
use bevy_render::batching::gpu_preprocessing::IndirectParametersIndexed;
use bevy_render::render_phase::DrawFunctionId;
use bevy_render::render_resource::SpecializedRenderPipeline;
use bevy_utils::default;
pub use image::ImageNodeAssetChangedSystems;
mod pipeline;
pub mod render_pass;
mod text;
pub mod ui_material;
mod ui_material_pipeline;
pub mod ui_texture_slice_pipeline;

#[cfg(feature = "bevy_ui_debug")]
mod debug_overlay;

use bevy_a11y::AccessibilitySystems;
use bevy_camera::visibility::InheritedVisibility;
use bevy_camera::{Camera, Camera2d, Camera3d, RenderTarget};
use bevy_ecs::entity::{EntityHashMap, EntityHashSet, EntityIndexMap};
use bevy_reflect::prelude::ReflectDefault;
use bevy_reflect::Reflect;
use bevy_render::camera::{extract_cameras, CameraMainPassTextureFormats};
use bevy_render::sync_world::{MainEntityHashMap, MainEntityHashSet};
use bevy_shader::load_shader_library;
use bevy_ui::widget::{ImageNode, ImageNodeSize, NodeImageMode, Text, TextShadow, ViewportNode};
use bevy_ui::{
    BackgroundColor, BackgroundGradient, BorderColor, BorderGradient, BoxShadow, CalculatedClip,
    ComputedNode, ComputedStackIndex, ComputedUiTargetCamera, Display, Node, OuterColor, Outline,
    ResolvedBorderRadius, UiGlobalTransform, UiSystems, VisualBox,
};

use bevy_app::prelude::*;
use bevy_asset::{AssetEventSystems, AssetId, Assets};
use bevy_color::{Alpha, ColorToComponents, LinearRgba};
use bevy_core_pipeline::schedule::{Core2d, Core2dSystems, Core3d, Core3dSystems};
use bevy_core_pipeline::upscaling::upscaling;
use bevy_ecs::prelude::*;
use bevy_ecs::schedule::IntoScheduleConfigs;
use bevy_ecs::system::{StaticSystemParam, SystemParam, SystemParamItem};
use bevy_image::{prelude::*, TRANSPARENT_IMAGE_HANDLE};
use bevy_math::{proj, vec4, Affine2, FloatOrd, Rect, UVec4, Vec2, Vec4, Vec4Swizzles as _};
use bevy_render::{
    globals::GlobalsBuffer,
    render_asset::{ExtractedAssets, RenderAsset, RenderAssets},
    render_phase::{
        sort_phase_system, AddRenderCommand, DrawFunctions, PhaseItem, PhaseItemExtraIndex,
        ViewSortedRenderPhases,
    },
    render_resource::*,
    renderer::{RenderDevice, RenderQueue},
    sync_world::{MainEntity, RenderEntity},
    texture::GpuImage,
    view::{ExtractedView, RetainedViewEntity, ViewUniforms},
    Extract, ExtractSchedule, GpuResourceAppExt, Render, RenderApp, RenderStartup, RenderSystems,
};
use bevy_sprite::BorderRect;
#[cfg(feature = "bevy_ui_debug")]
pub use debug_overlay::{GlobalUiDebugOptions, UiDebugOptions};

use gradient::GradientPlugin;

use bevy_platform::collections::{HashMap, HashSet};
use bevy_text::{
    ComputedTextBlock, EditableText, PositionedGlyph, Strikethrough, StrikethroughColor,
    TextBackgroundColor, TextColor, TextCursorStyle, TextLayoutInfo, TextSpan, Underline,
    UnderlineColor,
};
use bevy_transform::components::GlobalTransform;
use box_shadow::BoxShadowPlugin;
use bytemuck::{Pod, Zeroable};
use core::ops::Range;
use smallvec::SmallVec;
use std::marker::{PhantomData, Send};
use std::{array, mem};

pub use pipeline::*;
pub use render_pass::*;
pub use ui_material_pipeline::*;
use ui_texture_slice_pipeline::UiTextureSlicerPlugin;

use crate::clipping::clip_polygon;
use crate::shader_flags::INVERT;
use crate::text::{calculate_text_scroll_clip, extract_preedit_underlines, extract_text_cursor};

pub mod prelude {
    #[cfg(feature = "bevy_ui_debug")]
    pub use crate::debug_overlay::{GlobalUiDebugOptions, UiDebugOptions};

    pub use crate::{
        ui_material::*, ui_material_pipeline::UiMaterialPlugin, BoxShadowSamples, UiAntiAlias,
    };
}

/// Local Z offsets of "extracted nodes" for a given entity. These exist to allow rendering multiple "extracted nodes"
/// for a given source entity (ex: render both a background color _and_ a custom material for a given node).
///
/// When possible these offsets should be defined in _this_ module to ensure z-index coordination across contexts.
/// When this is _not_ possible, pick a suitably unique index unlikely to clash with other things (ex: `0.1826823` not `0.1`).
///
/// Offsets should be unique for a given node entity to avoid z fighting.
/// These should pretty much _always_ be larger than -0.5 and smaller than 0.5 to avoid clipping into nodes
/// above / below the current node in the stack.
///
/// A z-index of 0.0 is the baseline, which is used as the primary "background color" of the node.
///
/// Note that nodes "stack" on each other, so a negative offset on the node above could clip _into_
/// a positive offset on a node below.
pub mod stack_z_offsets {
    pub const BOX_SHADOW: f32 = -0.1;
    pub const BACKGROUND_COLOR: f32 = 0.0;
    pub const BORDER: f32 = 0.01;
    pub const GRADIENT: f32 = 0.02;
    pub const BORDER_GRADIENT: f32 = 0.03;
    pub const IMAGE: f32 = 0.04;
    pub const MATERIAL: f32 = 0.05;
    pub const TEXT_SHADOW: f32 = 0.0525;
    pub const TEXT_SELECTION: f32 = 0.055;
    pub const TEXT: f32 = 0.06;
    pub const TEXT_STRIKETHROUGH: f32 = 0.07;
    pub const TEXT_CURSOR: f32 = 0.08;
}

#[derive(Debug, Hash, PartialEq, Eq, Clone, SystemSet)]
pub enum RenderUiSystems {
    ExtractChanges,
    ExtractCameraViews,
    ExtractBoxShadows,
    ExtractBackgrounds,
    ExtractImages,
    ExtractTextureSlice,
    ExtractBorders,
    ExtractViewportNodes,
    ExtractTextBackgrounds,
    ExtractTextShadows,
    ExtractText,
    ExtractCursor,
    ExtractDebug,
    ExtractGradient,
    ExtractWipePhaseItemsIfCameraComponentsChanged,
}

/// Marker for controlling whether UI is rendered with or without anti-aliasing
/// in a camera. By default, UI is always anti-aliased.
///
/// **Note:** This does not affect text anti-aliasing. For that, use the `font_smoothing` property of the [`TextFont`](bevy_text::TextFont) component.
///
/// ```
/// use bevy_camera::prelude::*;
/// use bevy_ecs::prelude::*;
/// use bevy_ui::prelude::*;
/// use bevy_ui_render::prelude::*;
///
/// fn spawn_camera(mut commands: Commands) {
///     commands.spawn((
///         Camera2d,
///         // This will cause all UI in this camera to be rendered without
///         // anti-aliasing
///         UiAntiAlias::Off,
///     ));
/// }
/// ```
#[derive(Component, Clone, Copy, Default, Debug, Reflect, Eq, PartialEq)]
#[reflect(Component, Default, PartialEq, Clone)]
pub enum UiAntiAlias {
    /// UI will render with anti-aliasing
    #[default]
    On,
    /// UI will render without anti-aliasing
    Off,
}

/// Number of shadow samples.
/// A larger value will result in higher quality shadows.
/// Default is 4, values higher than ~10 offer diminishing returns.
///
/// ```
/// use bevy_camera::prelude::*;
/// use bevy_ecs::prelude::*;
/// use bevy_ui::prelude::*;
/// use bevy_ui_render::prelude::*;
///
/// fn spawn_camera(mut commands: Commands) {
///     commands.spawn((
///         Camera2d,
///         BoxShadowSamples(6),
///     ));
/// }
/// ```
#[derive(Component, Clone, Copy, Debug, Reflect, Eq, PartialEq)]
#[reflect(Component, Default, PartialEq, Clone)]
pub struct BoxShadowSamples(pub u32);

impl Default for BoxShadowSamples {
    fn default() -> Self {
        Self(4)
    }
}

#[derive(Default)]
pub struct UiRenderPlugin;

impl Plugin for UiRenderPlugin {
    fn build(&self, app: &mut App) {
        load_shader_library!(app, "ui.wesl");

        #[cfg(feature = "bevy_ui_debug")]
        app.init_resource::<GlobalUiDebugOptions>();

        app.add_systems(
            PostUpdate,
            (
                image::mark_images_as_changed_if_their_assets_changed,
                image::update_texture_atlas_layout_components,
            )
                .chain()
                .in_set(ImageNodeAssetChangedSystems)
                .after(UiSystems::Content)
                .after(AssetEventSystems)
                .after(AccessibilitySystems::Update),
        );

        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };

        render_app
            .init_gpu_resource::<SpecializedRenderPipelines<UiPipeline>>()
            .init_gpu_resource::<UiTexturedBindGroups<ExtractedUiNode>>()
            .init_gpu_resource::<UiMeta<ExtractedUiNode>>()
            .init_resource::<ExtractedUiNodes>()
            .allow_ambiguous_resource::<ExtractedUiNodes>()
            .init_resource::<DrawFunctions<TransparentUi>>()
            .init_resource::<ViewSortedRenderPhases<TransparentUi>>()
            .allow_ambiguous_resource::<ViewSortedRenderPhases<TransparentUi>>()
            .add_render_command::<TransparentUi, DrawUi>()
            .configure_sets(
                ExtractSchedule,
                (
                    RenderUiSystems::ExtractChanges,
                    RenderUiSystems::ExtractCameraViews,
                    RenderUiSystems::ExtractBoxShadows,
                    RenderUiSystems::ExtractBackgrounds,
                    RenderUiSystems::ExtractViewportNodes,
                    RenderUiSystems::ExtractImages,
                    RenderUiSystems::ExtractTextureSlice,
                    RenderUiSystems::ExtractBorders,
                    RenderUiSystems::ExtractTextBackgrounds,
                    RenderUiSystems::ExtractTextShadows,
                    RenderUiSystems::ExtractText,
                    RenderUiSystems::ExtractCursor,
                    RenderUiSystems::ExtractDebug,
                    RenderUiSystems::ExtractGradient,
                    RenderUiSystems::ExtractWipePhaseItemsIfCameraComponentsChanged,
                )
                    .chain_weak(),
            )
            .add_systems(RenderStartup, init_ui_pipeline)
            .add_systems(
                ExtractSchedule,
                (
                    extract_uinode_changes.in_set(RenderUiSystems::ExtractChanges),
                    extract_ui_camera_view
                        .after(extract_cameras)
                        .in_set(RenderUiSystems::ExtractCameraViews),
                    extract_uinode_background_colors.in_set(RenderUiSystems::ExtractBackgrounds),
                    extract_uinode_images.in_set(RenderUiSystems::ExtractImages),
                    extract_uinode_borders.in_set(RenderUiSystems::ExtractBorders),
                    extract_viewport_nodes.in_set(RenderUiSystems::ExtractViewportNodes),
                    extract_text_decorations.in_set(RenderUiSystems::ExtractTextBackgrounds),
                    extract_text_shadows.in_set(RenderUiSystems::ExtractTextShadows),
                    extract_text_sections.in_set(RenderUiSystems::ExtractText),
                    extract_text_cursor.in_set(RenderUiSystems::ExtractCursor),
                    extract_preedit_underlines.in_set(RenderUiSystems::ExtractCursor),
                    wipe_phase_items_if_camera_component_changed::<ExtractedUiNode, UiAntiAlias>
                        .in_set(RenderUiSystems::ExtractWipePhaseItemsIfCameraComponentsChanged),
                    #[cfg(feature = "bevy_ui_debug")]
                    debug_overlay::extract_debug_overlay.in_set(RenderUiSystems::ExtractDebug),
                ),
            )
            .add_systems(
                Render,
                (
                    queue_ui_items::<ExtractedUiNode>.in_set(RenderSystems::Queue),
                    sort_phase_system::<TransparentUi>.in_set(RenderSystems::PhaseSort),
                    prepare_uinodes::<ExtractedUiNode>.in_set(RenderSystems::PrepareBindGroups),
                    clear_batches::<ExtractedUiNode>.in_set(RenderSystems::Cleanup),
                ),
            )
            .add_systems(
                Core2d,
                ui_pass.after(Core2dSystems::PostProcess).before(upscaling),
            )
            .add_systems(
                Core3d,
                ui_pass.after(Core3dSystems::PostProcess).before(upscaling),
            );

        app.add_plugins(UiTextureSlicerPlugin);
        app.add_plugins(GradientPlugin);
        app.add_plugins(BoxShadowPlugin);
    }
}

#[derive(SystemParam)]
pub struct UiCameraMap<'w, 's> {
    mapping: Query<'w, 's, RenderEntity>,
}

impl<'w, 's> UiCameraMap<'w, 's> {
    /// Creates a [`UiCameraMapper`] for performing repeated camera-to-render-entity lookups.
    ///
    /// The last successful mapping is cached to avoid redundant queries.
    pub fn get_mapper(&'w self) -> UiCameraMapper<'w, 's> {
        UiCameraMapper {
            mapping: &self.mapping,
            camera_entity: None,
            render_entity: None,
        }
    }
}

/// Helper for mapping UI target camera entities to their corresponding render entities,
/// with caching to avoid repeated lookups for the same camera.
pub struct UiCameraMapper<'w, 's> {
    mapping: &'w Query<'w, 's, RenderEntity>,
    /// Cached camera entity from the last successful `map` call.
    camera_entity: Option<Entity>,
    /// Cached camera entity from the last successful `map` call.
    render_entity: Option<Entity>,
}

impl<'w, 's> UiCameraMapper<'w, 's> {
    /// Returns the render entity corresponding to the given [`ComputedUiTargetCamera`]'s camera, or none if no corresponding entity was found.
    pub fn map(&mut self, computed_target: &ComputedUiTargetCamera) -> Option<Entity> {
        let camera_entity = computed_target.get()?;
        if self.camera_entity != Some(camera_entity) {
            let new_render_camera_entity = self.mapping.get(camera_entity).ok()?;
            self.render_entity = Some(new_render_camera_entity);
            self.camera_entity = Some(camera_entity);
        }

        self.render_entity
    }

    /// Returns the cached camera entity from the last successful `map` call.
    pub fn current_camera(&self) -> Option<Entity> {
        self.camera_entity
    }
}

pub struct ExtractedUiNode {
    pub z_order: f32,
    pub image: AssetId<Image>,
    pub clip: Option<CalculatedClip>,
    pub item: ExtractedUiItem,
    pub transform: Affine2,
}

#[derive(Copy, Clone, Debug, Default, Pod, Zeroable)]
#[repr(C)]
pub struct ExtractedUiNodeInstanceData {
    world_from_local: Vec4,
    color: Vec4,
    border: Vec4,
    radius: [Vec4; 2],
    uv_scale: Vec2,
    uv_offset: Vec2,
    translation: Vec2,
    size: Vec2,
    flags: u32,
    pad: [u32; 3],
}

impl UiRenderObject for ExtractedUiNode {
    type DrawFunctions = DrawUi;
    type ViewPipelineKeyBuilder = UiNodePipelineKeyBuilder;
    type ViewQueryData = Option<&'static UiAntiAlias>;
    type SpecializedRenderPipeline = UiPipeline;
    type PipelineKeySystemParam = SRes<UiMeta<ExtractedUiNode>>;
    type InstanceData = ExtractedUiNodeInstanceData;
    type TexturedGpuAsset = GpuImage;

    const TEXTURED: bool = true;

    fn get_sort_key(&self) -> FloatOrd {
        FloatOrd(self.z_order)
    }

    fn create_view_pipeline_key_builder<'w, 's>(
        item: <<Self::ViewQueryData as QueryData>::ReadOnly as QueryData>::Item<'w, 's>,
    ) -> Self::ViewPipelineKeyBuilder {
        UiNodePipelineKeyBuilder {
            anti_alias: item.cloned(),
        }
    }

    fn create_pipeline_key(
        &self,
        cached_camera_view: &CachedCameraView<Self::ViewPipelineKeyBuilder>,
        ui_meta: &mut SystemParamItem<Self::PipelineKeySystemParam>,
    ) -> Option<<Self::SpecializedRenderPipeline as SpecializedRenderPipeline>::Key> {
        let mut flags = UiPipelineKeyFlags::empty();
        if matches!(
            cached_camera_view.pipeline_key_builder.anti_alias,
            None | Some(UiAntiAlias::On)
        ) {
            flags.insert(UiPipelineKeyFlags::ANTI_ALIAS);
        }
        if matches!(ui_meta.instances, UiInstances::Retained { .. }) {
            flags.insert(UiPipelineKeyFlags::RETAINED_INSTANCES);
        }

        Some(UiPipelineKey {
            target_format: cached_camera_view.extracted_view.target_format,
            flags,
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

    fn textured_asset_id(&self) -> AssetId<Image> {
        self.image
    }

    fn textured_bind_group_layout(
        pipeline: &Self::SpecializedRenderPipeline,
    ) -> Option<&BindGroupLayoutDescriptor> {
        Some(&pipeline.image_layout)
    }

    fn get_or_create_textured_bind_group(
        render_device: &RenderDevice,
        layout: &BindGroupLayout,
        gpu_image: &GpuImage,
    ) -> Option<BindGroup> {
        Some(render_device.create_bind_group(
            "ui_material_bind_group",
            layout,
            &BindGroupEntries::sequential((&gpu_image.texture_view, &gpu_image.sampler)),
        ))
    }

    fn clip(&self) -> Option<&CalculatedClip> {
        self.clip.as_ref()
    }

    fn quad_count(&self) -> usize {
        match &self.item {
            ExtractedUiItem::Node { .. } => 1,
            ExtractedUiItem::Glyphs { glyphs } => glyphs.len(),
        }
    }

    fn populate_quad(
        &self,
        out_quad: &mut UiQuad<Self::InstanceData>,
        index: usize,
        gpu_image: Option<&GpuImage>,
    ) {
        let positions;

        match &self.item {
            ExtractedUiItem::Node {
                atlas_scaling,
                flip_x,
                flip_y,
                border_radius,
                border,
                node_type,
                rect,
                color,
            } => {
                debug_assert_eq!(index, 0);

                let mut flags = if self.image != AssetId::default() {
                    shader_flags::TEXTURED
                } else {
                    shader_flags::UNTEXTURED
                };

                let rect_size = rect.size();

                let transform = self.transform;

                // Specify the corners of the node
                let points = QUAD_VERTEX_POSITIONS.map(|pos| pos * rect_size);
                positions = points.map(|pos| transform.transform_point2(pos));

                let uvs = if flags == shader_flags::UNTEXTURED {
                    [Vec2::ZERO, Vec2::X, Vec2::ONE, Vec2::Y]
                } else {
                    let mut uinode_rect = *rect;
                    let image = gpu_image
                        .expect("Image was checked during batching and should still exist");
                    // Rescale atlases. This is done here because we need texture data that might not be available in Extract.
                    let atlas_extent = atlas_scaling
                        .map(|scaling| image.size_2d().as_vec2() * scaling)
                        .unwrap_or(uinode_rect.max);
                    if *flip_x {
                        mem::swap(&mut uinode_rect.max.x, &mut uinode_rect.min.x);
                    }
                    if *flip_y {
                        mem::swap(&mut uinode_rect.max.y, &mut uinode_rect.min.y);
                    }
                    [
                        Vec2::new(uinode_rect.min.x, uinode_rect.min.y),
                        Vec2::new(uinode_rect.max.x, uinode_rect.min.y),
                        Vec2::new(uinode_rect.max.x, uinode_rect.max.y),
                        Vec2::new(uinode_rect.min.x, uinode_rect.max.y),
                    ]
                    .map(|pos| pos / atlas_extent)
                };

                let color = color.to_vec4();
                match node_type {
                    NodeType::Border(border_flags) => {
                        flags |= border_flags;
                    }
                    NodeType::Inverted => {
                        flags |= INVERT;
                    }
                    _ => {}
                }

                out_quad.instance_data = ExtractedUiNodeInstanceData {
                    world_from_local: pack_transform(transform, rect_size),
                    color,
                    flags,
                    radius: (*border_radius).into(),
                    border: vec4(
                        border.min_inset.x,
                        border.min_inset.y,
                        border.max_inset.x,
                        border.max_inset.y,
                    ),
                    size: rect_size,
                    uv_scale: uvs[2] - uvs[0],
                    uv_offset: uvs[0],
                    translation: transform.translation,
                    pad: default(),
                };
            }
            ExtractedUiItem::Glyphs { glyphs } => {
                let image =
                    gpu_image.expect("Image was checked during batching and should still exist");

                let atlas_extent = image.size_2d().as_vec2();

                let glyph = &glyphs[index];
                let color = glyph.color.to_vec4();
                let glyph_rect = glyph.rect;
                let rect_size = glyph_rect.size();

                // Specify the corners of the glyph
                positions = QUAD_VERTEX_POSITIONS.map(|pos| {
                    self.transform
                        .transform_point2(glyph.translation + pos * glyph_rect.size())
                });

                let uvs = [
                    Vec2::new(glyph.rect.min.x, glyph.rect.min.y) / atlas_extent,
                    Vec2::new(glyph.rect.max.x, glyph.rect.min.y) / atlas_extent,
                    Vec2::new(glyph.rect.max.x, glyph.rect.max.y) / atlas_extent,
                    Vec2::new(glyph.rect.min.x, glyph.rect.max.y) / atlas_extent,
                ];

                out_quad.instance_data = ExtractedUiNodeInstanceData {
                    world_from_local: pack_transform(self.transform, rect_size),
                    color,
                    flags: shader_flags::TEXTURED | shader_flags::TEXT,
                    radius: default(),
                    border: default(),
                    size: rect_size,
                    uv_scale: uvs[2] - uvs[0],
                    uv_offset: uvs[0],
                    translation: self.transform.transform_point2(glyph.translation),
                    pad: default(),
                };
            }
        }

        out_quad.positions = positions;
    }
}

/// The type of UI node.
/// This is used to determine how to render the UI node.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum NodeType {
    Rect,
    Inverted,
    Border(u32), // shader flags
}

pub enum ExtractedUiItem {
    Node {
        color: LinearRgba,
        rect: Rect,
        atlas_scaling: Option<Vec2>,
        flip_x: bool,
        flip_y: bool,
        /// Border radius of the UI node.
        /// Ordering: top left, top right, bottom right, bottom left.
        border_radius: ResolvedBorderRadius,
        /// Border thickness of the UI node.
        /// Ordering: left, top, right, bottom.
        border: BorderRect,
        node_type: NodeType,
    },
    /// A contiguous sequence of text glyphs from the same section
    Glyphs {
        /// The color, position, and UV rect of each glyph.
        glyphs: Vec<ExtractedGlyph>,
    },
}

pub struct ExtractedGlyph {
    pub color: LinearRgba,
    pub translation: Vec2,
    pub rect: Rect,
}

/// The list of UI nodes, as well as the set of nodes that changed.
///
/// This is a two-level data structure so that we can quickly remove all
/// nodes associated with a main-world entity when it changes.
pub type ExtractedUiNodes = UiRenderObjects<ExtractedUiNode>;

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct ChangedUiObject {
    render_entity: Entity,
    camera_entity: Entity,
}

/// A query filter that matches all UI nodes.
type UiNodeQueryFilter = (
    With<ComputedNode>,
    With<ComputedStackIndex>,
    With<UiGlobalTransform>,
    With<InheritedVisibility>,
    With<ComputedUiTargetCamera>,
);

// Note: Whenever you add a new component that affects UI rendering, make sure
// to add a `Changed` query filter and a reference to the `RemovedComponents`
// resource to `extract_uinode_changes` below.
//
// Note: We don't have to match on `AssetChanged` for images or texture atlas
// layouts because the
// `bevy_ui::widget::mark_images_as_changed_as_their_assets_changed` image marks
// the `ImageNode` for us automatically as changed when those assets change.

/// A render-world system that scans for any UI nodes that have changed and
/// removes the render world data associated with them.
pub fn extract_uinode_changes(
    mut commands: Commands,
    mut extracted_uinodes: ResMut<ExtractedUiNodes>,
    all_uinodes_query: Extract<Query<Entity, UiNodeQueryFilter>>,
    changed_uinodes_query: Extract<
        Query<
            Entity,
            (
                UiNodeQueryFilter,
                Or<(
                    Or<(
                        Changed<ComputedNode>,
                        Changed<ComputedStackIndex>,
                        Changed<UiGlobalTransform>,
                        Changed<InheritedVisibility>,
                        Changed<CalculatedClip>,
                        Changed<ComputedUiTargetCamera>,
                        Changed<BackgroundColor>,
                        Changed<OuterColor>,
                    )>,
                    Or<(
                        Changed<ImageNode>,
                        Changed<ImageNodeSize>,
                        Changed<BorderColor>,
                        Changed<Outline>,
                        Changed<ViewportNode>,
                        Changed<ComputedTextBlock>,
                        Changed<TextColor>,
                        Changed<TextLayoutInfo>,
                    )>,
                    Or<(
                        Changed<TextCursorStyle>,
                        Changed<TextShadow>,
                        Changed<BackgroundGradient>,
                        Changed<BorderGradient>,
                        Changed<BoxShadow>,
                        Changed<EditableText>,
                        Changed<Underline>,
                        Changed<Strikethrough>,
                    )>,
                    Or<(Changed<StrikethroughColor>, Changed<UnderlineColor>)>,
                )>,
            ),
        >,
    >,
    #[cfg(feature = "bevy_ui_debug")] changed_debug_options_query: Extract<
        Query<Entity, (UiNodeQueryFilter, Changed<UiDebugOptions>)>,
    >,
    text_span_query: Extract<
        Query<
            Entity,
            (
                With<TextSpan>,
                Or<(
                    Changed<TextColor>,
                    Changed<TextBackgroundColor>,
                    Changed<Underline>,
                    Changed<Strikethrough>,
                    Changed<StrikethroughColor>,
                    Changed<UnderlineColor>,
                )>,
            ),
        >,
    >,
    text_span_parent_query: Extract<Query<&ChildOf, With<TextSpan>>>,
    text_query: Extract<Query<Entity, With<Text>>>,
    (
        mut removed_computed_node_query,
        mut removed_computed_stack_index_query,
        mut removed_ui_global_transform_query,
        mut removed_inherited_visibility_query,
        mut removed_calculated_clip_query,
        mut removed_computed_ui_target_camera_query,
        mut removed_background_color_query,
        mut removed_outer_color_query,
    ): (
        Extract<RemovedComponents<ComputedNode>>,
        Extract<RemovedComponents<ComputedStackIndex>>,
        Extract<RemovedComponents<UiGlobalTransform>>,
        Extract<RemovedComponents<InheritedVisibility>>,
        Extract<RemovedComponents<CalculatedClip>>,
        Extract<RemovedComponents<ComputedUiTargetCamera>>,
        Extract<RemovedComponents<BackgroundColor>>,
        Extract<RemovedComponents<OuterColor>>,
    ),
    (
        mut removed_image_node_query,
        mut removed_image_node_size_query,
        mut removed_border_color_query,
        mut removed_outline_query,
        mut removed_viewport_node_query,
        mut removed_computed_text_block_query,
        mut removed_text_color_query,
        mut removed_text_layout_info_query,
    ): (
        Extract<RemovedComponents<ImageNode>>,
        Extract<RemovedComponents<ImageNodeSize>>,
        Extract<RemovedComponents<BorderColor>>,
        Extract<RemovedComponents<Outline>>,
        Extract<RemovedComponents<ViewportNode>>,
        Extract<RemovedComponents<ComputedTextBlock>>,
        Extract<RemovedComponents<TextColor>>,
        Extract<RemovedComponents<TextLayoutInfo>>,
    ),
    (
        mut removed_text_cursor_style_query,
        mut removed_text_shadow_query,
        mut removed_background_gradient_query,
        mut removed_border_gradient_query,
        mut removed_box_shadow_query,
        mut removed_editable_text_query,
        mut removed_underline_query,
        mut removed_strikethrough_query,
    ): (
        Extract<RemovedComponents<TextCursorStyle>>,
        Extract<RemovedComponents<TextShadow>>,
        Extract<RemovedComponents<BackgroundGradient>>,
        Extract<RemovedComponents<BorderGradient>>,
        Extract<RemovedComponents<BoxShadow>>,
        Extract<RemovedComponents<EditableText>>,
        Extract<RemovedComponents<Underline>>,
        Extract<RemovedComponents<Strikethrough>>,
    ),
    (mut removed_strikethrough_color_query, mut removed_underline_color_query): (
        Extract<RemovedComponents<StrikethroughColor>>,
        Extract<RemovedComponents<UnderlineColor>>,
    ),
    #[cfg(feature = "bevy_ui_debug")] mut removed_debug_options_query: Extract<
        RemovedComponents<UiDebugOptions>,
    >,
    #[cfg(feature = "bevy_ui_debug")] global_ui_debug_options: Extract<Res<GlobalUiDebugOptions>>,
    mut extra_nodes_to_invalidate: Local<MainEntityHashSet>,
) {
    extracted_uinodes.changed.clear();

    // If the debug options changed, we wipe everything.
    // That's a bit coarse-grained, but having the debug options change is rare
    // and should only happen in, well, debugging.
    #[cfg(feature = "bevy_ui_debug")]
    let must_wipe_all_nodes = global_ui_debug_options.is_changed();
    #[cfg(not(feature = "bevy_ui_debug"))]
    let must_wipe_all_nodes = false;

    if must_wipe_all_nodes {
        for main_entity in &all_uinodes_query {
            process_changed_entity(
                main_entity.into(),
                &mut commands,
                &text_span_parent_query,
                &text_query,
                &mut extracted_uinodes,
                Some(&mut extra_nodes_to_invalidate),
            );
        }
    } else {
        // Go through all nodes that have changed and invalidate any render world
        // data associated with them.
        for main_entity in changed_uinodes_query
            .iter()
            .chain(text_span_query.iter())
            .chain(removed_computed_node_query.read())
            .chain(removed_computed_stack_index_query.read())
            .chain(removed_ui_global_transform_query.read())
            .chain(removed_inherited_visibility_query.read())
            .chain(removed_calculated_clip_query.read())
            .chain(removed_computed_ui_target_camera_query.read())
            .chain(removed_background_color_query.read())
            .chain(removed_outer_color_query.read())
            .chain(removed_image_node_query.read())
            .chain(removed_image_node_size_query.read())
            .chain(removed_border_color_query.read())
            .chain(removed_outline_query.read())
            .chain(removed_viewport_node_query.read())
            .chain(removed_computed_text_block_query.read())
            .chain(removed_text_color_query.read())
            .chain(removed_text_layout_info_query.read())
            .chain(removed_text_cursor_style_query.read())
            .chain(removed_text_shadow_query.read())
            .chain(removed_background_gradient_query.read())
            .chain(removed_border_gradient_query.read())
            .chain(removed_box_shadow_query.read())
            .chain(removed_editable_text_query.read())
            .chain(removed_underline_query.read())
            .chain(removed_strikethrough_query.read())
            .chain(removed_strikethrough_color_query.read())
            .chain(removed_underline_color_query.read())
        {
            process_changed_entity(
                main_entity.into(),
                &mut commands,
                &text_span_parent_query,
                &text_query,
                &mut extracted_uinodes,
                Some(&mut extra_nodes_to_invalidate),
            );
        }

        // Process nodes that have changed debug options too, if that feature is
        // enabled.
        #[cfg(feature = "bevy_ui_debug")]
        for main_entity in changed_debug_options_query
            .iter()
            .chain(removed_debug_options_query.read())
        {
            process_changed_entity(
                main_entity.into(),
                &mut commands,
                &text_span_parent_query,
                &text_query,
                &mut extracted_uinodes,
                Some(&mut extra_nodes_to_invalidate),
            );
        }
    }

    for main_entity in extra_nodes_to_invalidate.drain() {
        process_changed_entity(
            main_entity,
            &mut commands,
            &text_span_parent_query,
            &text_query,
            &mut extracted_uinodes,
            None,
        );
    }

    fn process_changed_entity(
        mut main_entity: MainEntity,
        commands: &mut Commands,
        text_span_parent_query: &Query<&ChildOf, With<TextSpan>>,
        text_query: &Query<Entity, With<Text>>,
        extracted_uinodes: &mut ExtractedUiNodes,
        maybe_extra_nodes_to_invalidate: Option<&mut MainEntityHashSet>,
    ) {
        // Mark the node as changed so that the other `extract_` systems will
        // know to process it.
        let changed_ui_nodes = extracted_uinodes.changed.entry(main_entity).or_default();

        if let Some((prev_camera_entity, mut render_entities)) =
            extracted_uinodes.objects.remove(&main_entity)
        {
            for (render_entity, _) in render_entities.drain(..) {
                commands.entity(render_entity).despawn();
                changed_ui_nodes.push(ChangedUiObject {
                    render_entity,
                    camera_entity: prev_camera_entity,
                });
            }
        }

        // If this node is a `TextSpan`, then we need to invalidate the ancestor
        // `Text` node too. This is because `extract_text_decorations` only
        // looks at the `text_background_colors_query` for the text spans if the
        // `uinode_query` that it's iterating over matched the ancestor `Text`
        // node.
        if let Some(extra_nodes_to_invalidate) = maybe_extra_nodes_to_invalidate
            && let Ok(parent) = text_span_parent_query.get(main_entity.entity())
        {
            main_entity = parent.parent().into();
            loop {
                if text_query.contains(main_entity.entity()) {
                    extra_nodes_to_invalidate.insert(main_entity);
                    break;
                }
                match text_span_parent_query.get(main_entity.entity()) {
                    Ok(parent) => main_entity = parent.parent().into(),
                    Err(_) => break,
                }
            }
        }
    }
}

pub fn extract_uinode_background_colors(
    mut commands: Commands,
    extracted_uinodes: ResMut<ExtractedUiNodes>,
    uinode_query: Extract<
        Query<(
            Entity,
            &ComputedNode,
            &ComputedStackIndex,
            &UiGlobalTransform,
            &InheritedVisibility,
            Option<&CalculatedClip>,
            &ComputedUiTargetCamera,
            &BackgroundColor,
            Option<&OuterColor>,
        )>,
    >,
    camera_map: Extract<UiCameraMap>,
) {
    let extracted_uinodes = extracted_uinodes.into_inner();
    let mut camera_mapper = camera_map.get_mapper();

    for (
        (
            entity,
            uinode,
            stack_index,
            transform,
            inherited_visibility,
            clip,
            camera,
            background_color,
            maybe_outer_color,
        ),
        changed_objects,
    ) in extracted_uinodes
        .changed
        .iter_mut()
        .filter_map(|(main_entity, changed_objects)| {
            Some((
                uinode_query.get(main_entity.entity()).ok()?,
                changed_objects,
            ))
        })
    {
        // Skip invisible backgrounds
        if !inherited_visibility.get()
            || (background_color.is_fully_transparent()
                && maybe_outer_color.is_none_or(|outer| outer.is_fully_transparent()))
            || uinode.is_empty()
        {
            continue;
        }

        let Some(extracted_camera_entity) = camera_mapper.map(camera) else {
            continue;
        };

        if !background_color.is_fully_transparent() {
            UiRenderObjects::<ExtractedUiNode>::add_render_entity(
                changed_objects,
                &mut extracted_uinodes.objects,
                &mut commands,
                entity.into(),
                extracted_camera_entity,
                ExtractedUiNode {
                    z_order: stack_index.0 as f32 + stack_z_offsets::BACKGROUND_COLOR,
                    clip: clip.cloned(),
                    image: AssetId::default(),
                    transform: transform.into(),
                    item: ExtractedUiItem::Node {
                        color: background_color.0.into(),
                        rect: Rect {
                            min: Vec2::ZERO,
                            max: uinode.size,
                        },
                        atlas_scaling: None,
                        flip_x: false,
                        flip_y: false,
                        border: uinode.border(),
                        border_radius: uinode.border_radius(),
                        node_type: NodeType::Rect,
                    },
                },
            );
        }

        if let Some(outer_color) = maybe_outer_color
            && !outer_color.0.is_fully_transparent()
        {
            UiRenderObjects::<ExtractedUiNode>::add_render_entity(
                changed_objects,
                &mut extracted_uinodes.objects,
                &mut commands,
                entity.into(),
                extracted_camera_entity,
                ExtractedUiNode {
                    z_order: stack_index.0 as f32 + stack_z_offsets::BACKGROUND_COLOR,
                    clip: clip.cloned(),
                    image: AssetId::default(),
                    transform: transform.into(),
                    item: ExtractedUiItem::Node {
                        color: outer_color.0.into(),
                        rect: Rect {
                            min: Vec2::ZERO,
                            max: uinode.size,
                        },
                        atlas_scaling: None,
                        flip_x: false,
                        flip_y: false,
                        border: BorderRect::ZERO,
                        border_radius: uinode.border_radius(),
                        node_type: NodeType::Inverted,
                    },
                },
            );
        }
    }
}

pub fn extract_uinode_images(
    mut commands: Commands,
    extracted_uinodes: ResMut<ExtractedUiNodes>,
    texture_atlases: Extract<Res<Assets<TextureAtlasLayout>>>,
    uinode_query: Extract<
        Query<(
            Entity,
            &ComputedNode,
            &ComputedStackIndex,
            &UiGlobalTransform,
            &InheritedVisibility,
            Option<&CalculatedClip>,
            &ComputedUiTargetCamera,
            &ImageNode,
            &ImageNodeSize,
        )>,
    >,
    camera_map: Extract<UiCameraMap>,
) {
    let extracted_uinodes = extracted_uinodes.into_inner();
    let mut camera_mapper = camera_map.get_mapper();

    for (
        (
            entity,
            uinode,
            stack_index,
            transform,
            inherited_visibility,
            clip,
            camera,
            image,
            image_size,
        ),
        changed_objects,
    ) in extracted_uinodes
        .changed
        .iter_mut()
        .filter_map(|(main_entity, changed_objects)| {
            Some((
                uinode_query.get(main_entity.entity()).ok()?,
                changed_objects,
            ))
        })
    {
        let visual_box = match image.visual_box {
            VisualBox::ContentBox => uinode.content_box(),
            VisualBox::PaddingBox => uinode.padding_box(),
            VisualBox::BorderBox => uinode.border_box(),
        };

        // Skip invisible images
        if !inherited_visibility.get()
            || image.color.is_fully_transparent()
            || image.image.id() == TRANSPARENT_IMAGE_HANDLE.id()
            || image.image_mode.uses_slices()
            || visual_box.size().cmple(Vec2::ZERO).any()
        {
            continue;
        }

        let Some(extracted_camera_entity) = camera_mapper.map(camera) else {
            continue;
        };

        let size = if matches!(image.image_mode, NodeImageMode::Auto) {
            let source = image_size.size().as_vec2();
            if source.cmple(Vec2::ZERO).any() {
                visual_box.size()
            } else {
                source * (visual_box.size() / source).min_element()
            }
        } else {
            visual_box.size()
        };

        // The node's border radius is subtracted from the visual box target's edge insets
        // and then clamped to get the corner radius for the image. Ideally this should be handled
        // on the GPU, but that might need changes to `ui.wesl`'s UV calculations.
        let mut inset = match image.visual_box {
            VisualBox::ContentBox => uinode.content_inset(),
            VisualBox::PaddingBox => uinode.border(),
            VisualBox::BorderBox => BorderRect::ZERO,
        };
        let image_inset = 0.5 * (visual_box.size() - size);
        inset.min_inset += image_inset;
        inset.max_inset += image_inset;

        let radius = uinode.border_radius();
        let clamped_radius = ResolvedBorderRadius {
            top_left: (radius.top_left - inset.min_inset).clamp(Vec2::ZERO, 0.5 * size),
            top_right: (radius.top_right - Vec2::new(inset.max_inset.x, inset.min_inset.y))
                .clamp(Vec2::ZERO, 0.5 * size),
            bottom_right: (radius.bottom_right - inset.max_inset).clamp(Vec2::ZERO, 0.5 * size),
            bottom_left: (radius.bottom_left - Vec2::new(inset.min_inset.x, inset.max_inset.y))
                .clamp(Vec2::ZERO, 0.5 * size),
        };

        let atlas_rect = image
            .texture_atlas
            .as_ref()
            .and_then(|s| s.texture_rect(&texture_atlases))
            .map(|r| r.as_rect());

        let mut rect = match (atlas_rect, image.rect) {
            (None, None) => Rect {
                min: Vec2::ZERO,
                max: size,
            },
            (None, Some(image_rect)) => image_rect,
            (Some(atlas_rect), None) => atlas_rect,
            (Some(atlas_rect), Some(mut image_rect)) => {
                image_rect.min += atlas_rect.min;
                image_rect.max += atlas_rect.min;
                image_rect
            }
        };

        let atlas_scaling = if atlas_rect.is_some() || image.rect.is_some() {
            let atlas_scaling = size / rect.size();
            rect.min *= atlas_scaling;
            rect.max *= atlas_scaling;
            Some(atlas_scaling)
        } else {
            None
        };

        UiRenderObjects::<ExtractedUiNode>::add_render_entity(
            changed_objects,
            &mut extracted_uinodes.objects,
            &mut commands,
            entity.into(),
            extracted_camera_entity,
            ExtractedUiNode {
                z_order: stack_index.0 as f32 + stack_z_offsets::IMAGE,
                clip: clip.cloned(),
                image: image.image.id(),
                transform: Affine2::from(*transform)
                    * Affine2::from_translation(visual_box.center()),
                item: ExtractedUiItem::Node {
                    color: image.color.into(),
                    rect,
                    atlas_scaling,
                    flip_x: image.flip_x,
                    flip_y: image.flip_y,
                    border: BorderRect::ZERO,
                    border_radius: clamped_radius,
                    node_type: NodeType::Rect,
                },
            },
        );
    }
}

pub fn extract_uinode_borders(
    mut commands: Commands,
    extracted_uinodes: ResMut<ExtractedUiNodes>,
    uinode_query: Extract<
        Query<(
            Entity,
            Option<&Node>,
            &ComputedNode,
            &ComputedStackIndex,
            &UiGlobalTransform,
            &InheritedVisibility,
            Option<&CalculatedClip>,
            &ComputedUiTargetCamera,
            AnyOf<(&BorderColor, &Outline)>,
        )>,
    >,
    camera_map: Extract<UiCameraMap>,
) {
    let extracted_uinodes = extracted_uinodes.into_inner();
    let image = AssetId::<Image>::default();
    let mut camera_mapper = camera_map.get_mapper();

    for (
        (
            entity,
            node,
            computed_node,
            stack_index,
            transform,
            inherited_visibility,
            maybe_clip,
            camera,
            (maybe_border_color, maybe_outline),
        ),
        changed_objects,
    ) in extracted_uinodes
        .changed
        .iter_mut()
        .filter_map(|(main_entity, changed_objects)| {
            Some((
                uinode_query.get(main_entity.entity()).ok()?,
                changed_objects,
            ))
        })
    {
        // Skip invisible borders and removed nodes
        if !inherited_visibility.get() || node.is_some_and(|node| node.display == Display::None) {
            continue;
        }

        let Some(extracted_camera_entity) = camera_mapper.map(camera) else {
            continue;
        };

        // Don't extract borders with zero width along all edges
        if computed_node.border() != BorderRect::ZERO
            && let Some(border_color) = maybe_border_color
        {
            let border_colors = [
                border_color.left.to_linear(),
                border_color.top.to_linear(),
                border_color.right.to_linear(),
                border_color.bottom.to_linear(),
            ];

            const BORDER_FLAGS: [u32; 4] = [
                shader_flags::BORDER_LEFT,
                shader_flags::BORDER_TOP,
                shader_flags::BORDER_RIGHT,
                shader_flags::BORDER_BOTTOM,
            ];
            let mut completed_flags = 0;

            for (i, &color) in border_colors.iter().enumerate() {
                if color.is_fully_transparent() {
                    continue;
                }

                let mut border_flags = BORDER_FLAGS[i];

                if completed_flags & border_flags != 0 {
                    continue;
                }

                for j in i + 1..4 {
                    if color == border_colors[j] {
                        border_flags |= BORDER_FLAGS[j];
                    }
                }
                completed_flags |= border_flags;

                let node = ExtractedUiNode {
                    z_order: stack_index.0 as f32 + stack_z_offsets::BORDER,
                    image,
                    clip: maybe_clip.cloned(),
                    transform: transform.into(),
                    item: ExtractedUiItem::Node {
                        color,
                        rect: Rect {
                            max: computed_node.size(),
                            ..Default::default()
                        },
                        atlas_scaling: None,
                        flip_x: false,
                        flip_y: false,
                        border: computed_node.border(),
                        border_radius: computed_node.border_radius(),
                        node_type: NodeType::Border(border_flags),
                    },
                };

                UiRenderObjects::<ExtractedUiNode>::add_render_entity(
                    changed_objects,
                    &mut extracted_uinodes.objects,
                    &mut commands,
                    entity.into(),
                    extracted_camera_entity,
                    node,
                );
            }
        }

        if computed_node.outline_width() <= 0. {
            continue;
        }

        if let Some(outline) = maybe_outline.filter(|outline| !outline.color.is_fully_transparent())
        {
            let outline_size = computed_node.outlined_node_size();
            UiRenderObjects::<ExtractedUiNode>::add_render_entity(
                changed_objects,
                &mut extracted_uinodes.objects,
                &mut commands,
                entity.into(),
                extracted_camera_entity,
                ExtractedUiNode {
                    z_order: stack_index.0 as f32 + stack_z_offsets::BORDER,
                    image,
                    clip: maybe_clip.cloned(),
                    transform: transform.into(),
                    item: ExtractedUiItem::Node {
                        color: outline.color.into(),
                        rect: Rect {
                            max: outline_size,
                            ..Default::default()
                        },
                        atlas_scaling: None,
                        flip_x: false,
                        flip_y: false,
                        border: BorderRect::all(computed_node.outline_width()),
                        border_radius: computed_node.outline_radius(),
                        node_type: NodeType::Border(shader_flags::BORDER_ALL),
                    },
                },
            );
        }
    }
}

/// The UI camera is "moved back" by this many units (plus the [`UI_CAMERA_TRANSFORM_OFFSET`]) and also has a view
/// distance of this many units. This ensures that with a left-handed projection,
/// as UI elements are "stacked on top of each other", they are within the camera's view
/// and have room to grow.
// TODO: Consider computing this value at runtime based on the maximum z-value.
const UI_CAMERA_FAR: f32 = 1000.0;

// This value is subtracted from the far distance for the camera's z-position to ensure nodes at z == 0.0 are rendered
// TODO: Evaluate if we still need this.
const UI_CAMERA_TRANSFORM_OFFSET: f32 = -0.1;

/// The ID of the subview associated with a camera on which UI is to be drawn.
///
/// When UI is present, cameras extract to two views: the main 2D/3D one and a
/// UI one. The main 2D or 3D camera gets subview 0, and the corresponding UI
/// camera gets this subview, 1.
const UI_CAMERA_SUBVIEW: u32 = 1;

/// A render-world component that lives on the main render target view and
/// specifies the corresponding UI view.
///
/// For example, if UI is being rendered to a 3D camera, this component lives on
/// the 3D camera and contains the entity corresponding to the UI view.
#[derive(Component)]
/// Entity id of the temporary render entity with the corresponding extracted UI view.
pub struct UiCameraView(pub Entity);

/// A render-world component that lives on the UI view and specifies the
/// corresponding main render target view.
///
/// For example, if the UI is being rendered to a 3D camera, this component
/// lives on the UI view and contains the entity corresponding to the 3D camera.
///
/// This is the inverse of [`UiCameraView`].
#[derive(Component)]
pub struct UiViewTarget(pub Entity);

/// Information that [`extract_ui_camera_view`] maintains about each view that
/// it has seen.
pub struct CachedUiViewData {
    /// The render-world [`ExtractedView`].
    extracted_view_entity: Entity,
    /// The unique, stable identifier for the view across frames.
    retained_view_entity: RetainedViewEntity,
}

/// Extracts all UI elements associated with a camera into the render world.
pub fn extract_ui_camera_view(
    mut commands: Commands,
    mut transparent_render_phases: ResMut<ViewSortedRenderPhases<TransparentUi>>,
    query: Extract<
        Query<
            (
                Entity,
                RenderEntity,
                &Camera,
                Option<&UiAntiAlias>,
                Option<&BoxShadowSamples>,
            ),
            Or<(With<Camera2d>, With<Camera3d>)>,
        >,
    >,
    main_pass_formats: Res<CameraMainPassTextureFormats>,
    mut live_entities: Local<HashSet<RetainedViewEntity>>,
    mut cached_ui_view_data: Local<MainEntityHashMap<CachedUiViewData>>,
    mut removed_cameras_query: Extract<RemovedComponents<Camera>>,
    mut cameras_updated_this_frame: Local<MainEntityHashSet>,
) {
    cameras_updated_this_frame.clear();
    for (main_entity, render_entity, camera, ui_anti_alias, shadow_samples) in &query {
        let main_entity = MainEntity::from(main_entity);
        let retained_view_entity = RetainedViewEntity::new(main_entity, None, UI_CAMERA_SUBVIEW);

        // ignore inactive cameras
        if let (Some(physical_viewport_rect), Some(target_size), Some(target_format)) = (
            camera.physical_viewport_rect(),
            camera.physical_target_size(),
            main_pass_formats.get(&render_entity).copied(),
        ) && target_size.x != 0
            && target_size.y != 0
            && camera.physical_viewport_size().is_some()
            && camera.is_active
        {
            cameras_updated_this_frame.insert(main_entity);
            transparent_render_phases.prepare_for_new_frame(retained_view_entity);

            // use a projection matrix with the origin in the top left instead of the bottom left that comes with OrthographicProjection
            let projection_matrix = proj::orthographic(
                0.0,
                physical_viewport_rect.width() as f32,
                physical_viewport_rect.height() as f32,
                0.0,
                0.0,
                UI_CAMERA_FAR,
            );
            // We use `UI_CAMERA_SUBVIEW` here so as not to conflict with the
            // main 3D or 2D camera, which will have subview index 0.
            // Creates the UI view.
            let extracted_view = ExtractedView {
                retained_view_entity,
                clip_from_view: projection_matrix,
                world_from_view: GlobalTransform::from_xyz(
                    0.0,
                    0.0,
                    UI_CAMERA_FAR + UI_CAMERA_TRANSFORM_OFFSET,
                ),
                clip_from_world: None,
                target_format,
                viewport: UVec4::from((physical_viewport_rect.min, physical_viewport_rect.size())),
                color_grading: Default::default(),
                invert_culling: false,
            };
            // Link to the main camera view.
            let ui_view_target_component = UiViewTarget(render_entity);

            let ui_camera_view = match cached_ui_view_data.get(&main_entity) {
                Some(cached_ui_view_data) => commands
                    .entity(cached_ui_view_data.extracted_view_entity)
                    .insert((extracted_view, ui_view_target_component))
                    .id(),
                None => commands
                    .spawn((extracted_view, ui_view_target_component))
                    .id(),
            };

            let mut entity_commands = commands
                .get_entity(render_entity)
                .expect("Camera entity wasn't synced.");
            // Link from the main 2D/3D camera view to the UI view.
            entity_commands.insert(UiCameraView(ui_camera_view));
            if let Some(ui_anti_alias) = ui_anti_alias {
                entity_commands.insert(*ui_anti_alias);
            }
            if let Some(shadow_samples) = shadow_samples {
                entity_commands.insert(*shadow_samples);
            }

            live_entities.insert(retained_view_entity);
            cached_ui_view_data.insert(
                main_entity,
                CachedUiViewData {
                    extracted_view_entity: ui_camera_view,
                    retained_view_entity,
                },
            );
            continue;
        }

        // If we got here, the camera no longer exists or is no longer
        // renderable. Remove its associated render-world data.
        commands
            .get_entity(render_entity)
            .expect("Camera entity wasn't synced.")
            .remove::<(UiCameraView, UiAntiAlias, BoxShadowSamples)>();
        live_entities.remove(&retained_view_entity);
        if let Some(cached_ui_view_data) = cached_ui_view_data.remove(&main_entity) {
            commands
                .entity(cached_ui_view_data.extracted_view_entity)
                .despawn();
        }
    }

    // Only remove the render-world data for a camera if we didn't handle the
    // camera above.
    // It's possible that the `Camera` component was removed and added in the
    // same frame.
    for main_entity in removed_cameras_query.read() {
        let main_entity = MainEntity::from(main_entity);
        if cameras_updated_this_frame.contains(&main_entity) {
            continue;
        }

        if let Some(cached_ui_view_data) = cached_ui_view_data.remove(&main_entity) {
            commands
                .entity(cached_ui_view_data.extracted_view_entity)
                .despawn();
            live_entities.remove(&cached_ui_view_data.retained_view_entity);
        }
    }

    // Clean up render phases belonging to cameras that no longer exist.
    transparent_render_phases.retain(|entity, _| live_entities.contains(entity));
}

pub fn wipe_phase_items_if_camera_component_changed<E, C>(
    changed_cameras_query: Extract<
        Query<Entity, (Changed<C>, Or<(With<Camera2d>, With<Camera3d>)>)>,
    >,
    all_cameras_query: Extract<Query<RenderEntity, Or<(With<Camera2d>, With<Camera3d>)>>>,
    mut removed_components: Extract<RemovedComponents<C>>,
    render_objects: ResMut<UiRenderObjects<E>>,
    mut cameras_to_invalidate: Local<EntityHashSet>,
) where
    E: UiRenderObject,
    C: Component,
{
    for main_entity in changed_cameras_query
        .iter()
        .chain(removed_components.read())
    {
        let main_entity = MainEntity::from(main_entity);
        if let Ok(camera_render_entity) = all_cameras_query.get(main_entity.entity()) {
            cameras_to_invalidate.insert(camera_render_entity);
        }
    }

    if cameras_to_invalidate.is_empty() {
        return;
    }

    let render_objects = render_objects.into_inner();
    for (main_entity, (camera_entity, render_entities)) in render_objects.objects.iter() {
        if !cameras_to_invalidate.contains(camera_entity) {
            continue;
        }
        for render_entity in render_entities.keys() {
            render_objects
                .changed
                .entry(*main_entity)
                .or_default()
                .push(ChangedUiObject {
                    render_entity: *render_entity,
                    camera_entity: *camera_entity,
                });
        }
    }

    cameras_to_invalidate.clear();
}

pub fn extract_viewport_nodes(
    mut commands: Commands,
    extracted_uinodes: ResMut<ExtractedUiNodes>,
    camera_query: Extract<Query<(&Camera, &RenderTarget)>>,
    uinode_query: Extract<
        Query<(
            Entity,
            &ComputedNode,
            &ComputedStackIndex,
            &UiGlobalTransform,
            &InheritedVisibility,
            Option<&CalculatedClip>,
            &ComputedUiTargetCamera,
            &ViewportNode,
        )>,
    >,
    camera_map: Extract<UiCameraMap>,
) {
    let extracted_uinodes = extracted_uinodes.into_inner();
    let mut camera_mapper = camera_map.get_mapper();

    for (
        (entity, uinode, stack_index, transform, inherited_visibility, clip, camera, viewport_node),
        changed_objects,
    ) in extracted_uinodes
        .changed
        .iter_mut()
        .filter_map(|(main_entity, changed_objects)| {
            Some((
                uinode_query.get(main_entity.entity()).ok()?,
                changed_objects,
            ))
        })
    {
        // Skip invisible images
        if !inherited_visibility.get() || uinode.is_empty() {
            continue;
        }

        let Some(extracted_camera_entity) = camera_mapper.map(camera) else {
            continue;
        };
        let Some(camera_entity) = viewport_node.camera else {
            continue;
        };

        let Some(image) = camera_query
            .get(camera_entity)
            .ok()
            .and_then(|(_, render_target)| render_target.as_image())
        else {
            continue;
        };

        UiRenderObjects::<ExtractedUiNode>::add_render_entity(
            changed_objects,
            &mut extracted_uinodes.objects,
            &mut commands,
            entity.into(),
            extracted_camera_entity,
            ExtractedUiNode {
                z_order: stack_index.0 as f32 + stack_z_offsets::IMAGE,
                clip: clip.cloned(),
                image: image.id(),
                transform: transform.into(),
                item: ExtractedUiItem::Node {
                    color: LinearRgba::WHITE,
                    rect: Rect {
                        min: Vec2::ZERO,
                        max: uinode.size,
                    },
                    atlas_scaling: None,
                    flip_x: false,
                    flip_y: false,
                    border: uinode.border(),
                    border_radius: uinode.border_radius(),
                    node_type: NodeType::Rect,
                },
            },
        );
    }
}

pub fn extract_text_sections(
    mut commands: Commands,
    extracted_uinodes: ResMut<ExtractedUiNodes>,
    uinode_query: Extract<
        Query<(
            Entity,
            &ComputedNode,
            &ComputedStackIndex,
            &UiGlobalTransform,
            &InheritedVisibility,
            Option<&CalculatedClip>,
            &ComputedUiTargetCamera,
            &ComputedTextBlock,
            &TextColor,
            &TextLayoutInfo,
            Option<&EditableText>,
            Option<&TextCursorStyle>,
        )>,
    >,
    text_styles: Extract<Query<&TextColor>>,
    camera_map: Extract<UiCameraMap>,
) {
    let extracted_uinodes = extracted_uinodes.into_inner();
    let mut camera_mapper = camera_map.get_mapper();

    let mut glyphs = vec![];

    for (
        (
            entity,
            uinode,
            stack_index,
            global_transform,
            inherited_visibility,
            maybe_clip,
            camera,
            computed_block,
            text_color,
            text_layout_info,
            editable_text,
            cursor_style,
        ),
        changed_objects,
    ) in extracted_uinodes
        .changed
        .iter_mut()
        .filter_map(|(main_entity, changed_objects)| {
            Some((
                uinode_query.get(main_entity.entity()).ok()?,
                changed_objects,
            ))
        })
    {
        // Skip if not visible or if size is set to zero (e.g. when a parent is set to `Display::None`)
        if !inherited_visibility.get() || uinode.is_empty() {
            continue;
        }

        let Some(extracted_camera_entity) = camera_mapper.map(camera) else {
            continue;
        };

        let transform = Affine2::from(*global_transform)
            * Affine2::from_translation(
                uinode.content_box().min
                    - editable_text.map_or(Vec2::ZERO, |text| text.viewport.offset),
            );

        let clip = calculate_text_scroll_clip(editable_text, maybe_clip, uinode, global_transform);

        let mut color = text_color.0.to_linear();

        let selected_text_color = cursor_style
            .and_then(|cursor_style| cursor_style.selected_text_color)
            .map(|selected_text_color| selected_text_color.to_linear());

        let mut current_section_index = 0;

        for (
            i,
            PositionedGlyph {
                position,
                atlas_info,
                section_index,
                ..
            },
        ) in text_layout_info.glyphs.iter().enumerate()
        {
            if current_section_index != *section_index
                && let Some(section_entity) = computed_block
                    .entities()
                    .get(*section_index as usize)
                    .map(|t| t.entity)
            {
                color = text_styles
                    .get(section_entity)
                    .map(|text_color| LinearRgba::from(text_color.0))
                    .unwrap_or_default();
                current_section_index = *section_index;
            }

            let color = if !atlas_info.is_alpha_mask {
                LinearRgba::WHITE
            } else if let Some(selected_text_color) = selected_text_color
                && text_layout_info
                    .selection_rects
                    .iter()
                    .any(|selection_rect| {
                        let glyph_rect = Rect::from_center_size(*position, atlas_info.rect.size());
                        selection_rect.contains(glyph_rect.min)
                            && selection_rect.contains(glyph_rect.max)
                    })
            {
                selected_text_color
            } else {
                color
            };

            glyphs.push(ExtractedGlyph {
                color,
                translation: *position,
                rect: atlas_info.rect,
            });

            if text_layout_info
                .glyphs
                .get(i + 1)
                .is_none_or(|info| info.atlas_info.texture != atlas_info.texture)
            {
                UiRenderObjects::<ExtractedUiNode>::add_render_entity(
                    changed_objects,
                    &mut extracted_uinodes.objects,
                    &mut commands,
                    entity.into(),
                    extracted_camera_entity,
                    ExtractedUiNode {
                        z_order: stack_index.0 as f32 + stack_z_offsets::TEXT,
                        image: atlas_info.texture,
                        clip: clip.clone(),
                        item: ExtractedUiItem::Glyphs {
                            glyphs: mem::take(&mut glyphs),
                        },
                        transform,
                    },
                );
            }
        }
    }
}

pub fn extract_text_shadows(
    mut commands: Commands,
    extracted_uinodes: ResMut<ExtractedUiNodes>,
    uinode_query: Extract<
        Query<(
            Entity,
            &ComputedNode,
            &ComputedStackIndex,
            &UiGlobalTransform,
            &ComputedUiTargetCamera,
            &InheritedVisibility,
            Option<&CalculatedClip>,
            &TextLayoutInfo,
            &TextShadow,
            &ComputedTextBlock,
            Option<&EditableText>,
        )>,
    >,
    text_decoration_query: Extract<Query<(Has<Strikethrough>, Has<Underline>)>>,
    camera_map: Extract<UiCameraMap>,
) {
    let extracted_uinodes = extracted_uinodes.into_inner();
    let mut camera_mapper = camera_map.get_mapper();

    let mut glyphs = vec![];

    for (
        (
            entity,
            uinode,
            stack_index,
            global_transform,
            target,
            inherited_visibility,
            maybe_clip,
            text_layout_info,
            shadow,
            computed_block,
            editable_text,
        ),
        changed_objects,
    ) in extracted_uinodes
        .changed
        .iter_mut()
        .filter_map(|(main_entity, changed_objects)| {
            Some((
                uinode_query.get(main_entity.entity()).ok()?,
                changed_objects,
            ))
        })
    {
        // Skip if not visible or if size is set to zero (e.g. when a parent is set to `Display::None`)
        if !inherited_visibility.get() || uinode.is_empty() {
            continue;
        }

        let Some(extracted_camera_entity) = camera_mapper.map(target) else {
            continue;
        };

        let node_transform = Affine2::from(*global_transform)
            * Affine2::from_translation(
                uinode.content_box().min + shadow.offset / uinode.inverse_scale_factor()
                    - editable_text.map_or(Vec2::ZERO, |text| text.viewport.offset),
            );

        let clip = calculate_text_scroll_clip(editable_text, maybe_clip, uinode, global_transform);

        for (
            i,
            PositionedGlyph {
                position,
                atlas_info,
                section_index,
                ..
            },
        ) in text_layout_info.glyphs.iter().enumerate()
        {
            glyphs.push(ExtractedGlyph {
                color: shadow.color.into(),
                translation: *position,
                rect: atlas_info.rect,
            });

            if text_layout_info.glyphs.get(i + 1).is_none_or(|info| {
                info.section_index != *section_index
                    || info.atlas_info.texture != atlas_info.texture
            }) {
                UiRenderObjects::<ExtractedUiNode>::add_render_entity(
                    changed_objects,
                    &mut extracted_uinodes.objects,
                    &mut commands,
                    entity.into(),
                    extracted_camera_entity,
                    ExtractedUiNode {
                        transform: node_transform,
                        z_order: stack_index.0 as f32 + stack_z_offsets::TEXT,
                        image: atlas_info.texture,
                        clip: clip.clone(),
                        item: ExtractedUiItem::Glyphs {
                            glyphs: mem::take(&mut glyphs),
                        },
                    },
                );
            }
        }

        for run in text_layout_info.run_geometry.iter() {
            let Some(section_entity) = computed_block
                .entities()
                .get(run.section_index as usize)
                .map(|t| t.entity)
            else {
                continue;
            };
            let Ok((has_strikethrough, has_underline)) = text_decoration_query.get(section_entity)
            else {
                continue;
            };

            if has_strikethrough {
                UiRenderObjects::<ExtractedUiNode>::add_render_entity(
                    changed_objects,
                    &mut extracted_uinodes.objects,
                    &mut commands,
                    entity.into(),
                    extracted_camera_entity,
                    ExtractedUiNode {
                        z_order: stack_index.0 as f32 + stack_z_offsets::TEXT,
                        clip: clip.clone(),
                        image: AssetId::default(),
                        transform: node_transform
                            * Affine2::from_translation(run.strikethrough_position()),
                        item: ExtractedUiItem::Node {
                            color: shadow.color.into(),
                            rect: Rect {
                                min: Vec2::ZERO,
                                max: run.strikethrough_size(),
                            },
                            atlas_scaling: None,
                            flip_x: false,
                            flip_y: false,
                            border: BorderRect::ZERO,
                            border_radius: ResolvedBorderRadius::ZERO,
                            node_type: NodeType::Rect,
                        },
                    },
                );
            }

            if has_underline {
                UiRenderObjects::<ExtractedUiNode>::add_render_entity(
                    changed_objects,
                    &mut extracted_uinodes.objects,
                    &mut commands,
                    entity.into(),
                    extracted_camera_entity,
                    ExtractedUiNode {
                        z_order: stack_index.0 as f32 + stack_z_offsets::TEXT,
                        clip: clip.clone(),
                        image: AssetId::default(),
                        transform: node_transform
                            * Affine2::from_translation(run.underline_position()),
                        item: ExtractedUiItem::Node {
                            color: shadow.color.into(),
                            rect: Rect {
                                min: Vec2::ZERO,
                                max: run.underline_size(),
                            },
                            atlas_scaling: None,
                            flip_x: false,
                            flip_y: false,
                            border: BorderRect::ZERO,
                            border_radius: ResolvedBorderRadius::ZERO,
                            node_type: NodeType::Rect,
                        },
                    },
                );
            }
        }
    }
}

pub fn extract_text_decorations(
    mut commands: Commands,
    extracted_uinodes: ResMut<ExtractedUiNodes>,
    uinode_query: Extract<
        Query<(
            Entity,
            &ComputedNode,
            &ComputedStackIndex,
            &ComputedTextBlock,
            &UiGlobalTransform,
            &InheritedVisibility,
            Option<&CalculatedClip>,
            &ComputedUiTargetCamera,
            &TextLayoutInfo,
            Option<&EditableText>,
        )>,
    >,
    text_background_colors_query: Extract<
        Query<(
            AnyOf<(&TextBackgroundColor, &Strikethrough, &Underline)>,
            &TextColor,
            Option<&StrikethroughColor>,
            Option<&UnderlineColor>,
        )>,
    >,
    camera_map: Extract<UiCameraMap>,
) {
    let extracted_uinodes = extracted_uinodes.into_inner();
    let mut camera_mapper = camera_map.get_mapper();

    for (
        (
            entity,
            uinode,
            stack_index,
            computed_block,
            global_transform,
            inherited_visibility,
            maybe_clip,
            camera,
            text_layout_info,
            editable_text,
        ),
        changed_objects,
    ) in extracted_uinodes
        .changed
        .iter_mut()
        .filter_map(|(main_entity, changed_objects)| {
            Some((
                uinode_query.get(main_entity.entity()).ok()?,
                changed_objects,
            ))
        })
    {
        // Skip if not visible or if size is set to zero (e.g. when a parent is set to `Display::None`)
        if !inherited_visibility.get() || uinode.is_empty() {
            continue;
        }

        let Some(extracted_camera_entity) = camera_mapper.map(camera) else {
            continue;
        };

        let transform = Affine2::from(global_transform)
            * Affine2::from_translation(
                uinode.content_box().min
                    - editable_text.map_or(Vec2::ZERO, |text| text.viewport.offset),
            );

        let clip = calculate_text_scroll_clip(editable_text, maybe_clip, uinode, global_transform);

        for run in text_layout_info.run_geometry.iter() {
            let Some(section_entity) = computed_block
                .entities()
                .get(run.section_index as usize)
                .map(|t| t.entity)
            else {
                continue;
            };
            let Ok((
                (text_background_color, maybe_strikethrough, maybe_underline),
                text_color,
                maybe_strikethrough_color,
                maybe_underline_color,
            )) = text_background_colors_query.get(section_entity)
            else {
                continue;
            };

            if let Some(text_background_color) = text_background_color {
                UiRenderObjects::<ExtractedUiNode>::add_render_entity(
                    changed_objects,
                    &mut extracted_uinodes.objects,
                    &mut commands,
                    entity.into(),
                    extracted_camera_entity,
                    ExtractedUiNode {
                        z_order: stack_index.0 as f32 + stack_z_offsets::TEXT,
                        clip: clip.clone(),
                        image: AssetId::default(),
                        transform: transform * Affine2::from_translation(run.bounds.center()),
                        item: ExtractedUiItem::Node {
                            color: text_background_color.0.to_linear(),
                            rect: Rect {
                                min: Vec2::ZERO,
                                max: run.bounds.size(),
                            },
                            atlas_scaling: None,
                            flip_x: false,
                            flip_y: false,
                            border: BorderRect::ZERO,
                            border_radius: ResolvedBorderRadius::ZERO,
                            node_type: NodeType::Rect,
                        },
                    },
                );
            }

            if maybe_strikethrough.is_some() {
                let color = maybe_strikethrough_color
                    .map(|sc| sc.0)
                    .unwrap_or(text_color.0)
                    .to_linear();

                UiRenderObjects::<ExtractedUiNode>::add_render_entity(
                    changed_objects,
                    &mut extracted_uinodes.objects,
                    &mut commands,
                    entity.into(),
                    extracted_camera_entity,
                    ExtractedUiNode {
                        z_order: stack_index.0 as f32 + stack_z_offsets::TEXT_STRIKETHROUGH,
                        clip: clip.clone(),
                        image: AssetId::default(),
                        transform: transform
                            * Affine2::from_translation(run.strikethrough_position()),
                        item: ExtractedUiItem::Node {
                            color,
                            rect: Rect {
                                min: Vec2::ZERO,
                                max: run.strikethrough_size(),
                            },
                            atlas_scaling: None,
                            flip_x: false,
                            flip_y: false,
                            border: BorderRect::ZERO,
                            border_radius: ResolvedBorderRadius::ZERO,
                            node_type: NodeType::Rect,
                        },
                    },
                );
            }

            if maybe_underline.is_some() {
                let color = maybe_underline_color
                    .map(|uc| uc.0)
                    .unwrap_or(text_color.0)
                    .to_linear();

                UiRenderObjects::<ExtractedUiNode>::add_render_entity(
                    changed_objects,
                    &mut extracted_uinodes.objects,
                    &mut commands,
                    entity.into(),
                    extracted_camera_entity,
                    ExtractedUiNode {
                        z_order: stack_index.0 as f32 + stack_z_offsets::TEXT_STRIKETHROUGH,
                        clip: clip.clone(),
                        image: AssetId::default(),
                        transform: transform * Affine2::from_translation(run.underline_position()),
                        item: ExtractedUiItem::Node {
                            color,
                            rect: Rect {
                                min: Vec2::ZERO,
                                max: run.underline_size(),
                            },
                            atlas_scaling: None,
                            flip_x: false,
                            flip_y: false,
                            border: BorderRect::ZERO,
                            border_radius: ResolvedBorderRadius::ZERO,
                            node_type: NodeType::Rect,
                        },
                    },
                );
            }
        }
    }
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct UiVertex {
    pub position: Vec2,
}

#[derive(Resource)]
pub struct UiMeta<E>
where
    E: UiRenderObject,
{
    vertices: RawBufferVec<UiVertex>,
    indices: RawBufferVec<u32>,
    instances: UiInstances<E::InstanceData>,
    view_bind_group: Option<BindGroup>,
}

impl<E> FromWorld for UiMeta<E>
where
    E: UiRenderObject,
{
    fn from_world(world: &mut World) -> Self {
        let render_device = world.resource::<RenderDevice>();

        Self {
            vertices: RawBufferVec::new(BufferUsages::VERTEX),
            indices: RawBufferVec::new(BufferUsages::INDEX),
            instances: if render_device.limits().max_storage_buffers_per_shader_stage > 0 {
                UiInstances::Retained {
                    instances: SparseBufferVec::new(
                        BufferUsages::STORAGE,
                        "UI retained instances".into(),
                    ),
                    instances_free_list: vec![],
                    entity_to_instance_index: default(),
                    instance_index_buffer: RawBufferVec::new(BufferUsages::VERTEX),
                    bind_group: None,
                }
            } else {
                UiInstances::Immediate {
                    instances: RawBufferVec::new(BufferUsages::VERTEX | BufferUsages::COPY_DST),
                }
            },
            view_bind_group: None,
        }
    }
}

#[expect(clippy::large_enum_variant, reason = "it only ever wastes stack space")]
pub(crate) enum UiInstances<T>
where
    T: Pod + Default,
{
    Retained {
        instances: SparseBufferVec<T>,
        instances_free_list: Vec<u32>,
        entity_to_instance_index: EntityHashMap<SmallVec<[u32; 1]>>,
        instance_index_buffer: RawBufferVec<u32>,
        bind_group: Option<BindGroup>,
    },
    Immediate {
        instances: RawBufferVec<T>,
    },
}

impl<T> UiInstances<T>
where
    T: Pod + Default,
{
    fn get_entity_mut(&mut self, entity: Entity) -> UiEntityInstances<'_, T> {
        match *self {
            UiInstances::Retained {
                ref mut instances,
                ref mut instances_free_list,
                ref mut entity_to_instance_index,
                ref mut instance_index_buffer,
                ..
            } => UiEntityInstances::Retained {
                instances,
                instances_free_list,
                entity_instance_indices: entity_to_instance_index.entry(entity).or_default(),
                instance_index_buffer,
            },
            UiInstances::Immediate { ref mut instances } => {
                UiEntityInstances::Immediate { instances }
            }
        }
    }

    fn free_entity(&mut self, entity: Entity) {
        if let UiInstances::Retained {
            ref mut instances_free_list,
            ref mut entity_to_instance_index,
            ..
        } = *self
            && let Some(indices) = entity_to_instance_index.remove(&entity)
        {
            instances_free_list.extend(indices);
        }
    }
}

enum UiEntityInstances<'a, T>
where
    T: Pod + Default,
{
    Retained {
        instances: &'a mut SparseBufferVec<T>,
        instances_free_list: &'a mut Vec<u32>,
        entity_instance_indices: &'a mut SmallVec<[u32; 1]>,
        instance_index_buffer: &'a mut RawBufferVec<u32>,
    },
    Immediate {
        instances: &'a mut RawBufferVec<T>,
    },
}

impl<'a, T> UiEntityInstances<'a, T>
where
    T: Pod + Default,
{
    fn get_or_insert_instance(
        &mut self,
        entity_index: usize,
        create_instance: impl FnOnce() -> T,
    ) -> u32 {
        match *self {
            UiEntityInstances::Immediate { ref mut instances } => {
                instances.push(create_instance()) as u32
            }
            UiEntityInstances::Retained {
                ref mut instances,
                ref mut instances_free_list,
                ref mut entity_instance_indices,
                ref mut instance_index_buffer,
            } => {
                if let Some(instance_index) = entity_instance_indices.get_mut(entity_index) {
                    return instance_index_buffer.push(*instance_index) as u32;
                }

                let instance_index = match instances_free_list.pop() {
                    Some(instance_index) => {
                        instances.set(instance_index, create_instance());
                        instance_index
                    }
                    None => instances.push(create_instance()),
                };
                debug_assert_eq!(entity_instance_indices.len(), entity_index);
                entity_instance_indices.push(instance_index);
                instance_index_buffer.push(instance_index) as u32
            }
        }
    }
}

pub(crate) const QUAD_VERTEX_POSITIONS: [Vec2; 4] = [
    Vec2::new(-0.5, -0.5),
    Vec2::new(0.5, -0.5),
    Vec2::new(0.5, 0.5),
    Vec2::new(-0.5, 0.5),
];
pub(crate) const QUAD_INDICES: [u32; 6] = [0, 1, 2, 0, 2, 3];

#[derive(Component, Debug)]
pub struct UiBatch<E>
where
    E: UiRenderObject,
{
    pub params: Vec<IndirectParametersIndexed>,
    pub textured_asset_id: AssetId<<E::TexturedGpuAsset as RenderAsset>::SourceAsset>,
    phantom: PhantomData<E>,
}

impl<E> UiBatch<E>
where
    E: UiRenderObject,
{
    fn push_simple_quad(&mut self, instance_index: u32) {
        self.push(0, 0..(QUAD_INDICES.len() as u32), instance_index);
    }

    fn push(&mut self, base_vertex: u32, indices: Range<u32>, instance_index: u32) {
        match self.params.last_mut() {
            Some(params)
                if params.base_vertex == base_vertex
                    && params.first_index == indices.start
                    && params.first_index + params.index_count == indices.end
                    && params.first_instance + params.instance_count == instance_index =>
            {
                params.instance_count += 1;
            }
            _ => self.params.push(IndirectParametersIndexed {
                index_count: indices.end - indices.start,
                instance_count: 1,
                first_index: indices.start,
                base_vertex,
                first_instance: instance_index,
            }),
        }
    }
}

/// The values here should match the values for the constants in `ui.wesl`
pub mod shader_flags {
    /// Texture should be ignored
    pub const UNTEXTURED: u32 = 0;
    /// Textured
    pub const TEXTURED: u32 = 1;
    /// Ordering: top left, top right, bottom right, bottom left.
    pub const CORNERS: [u32; 4] = [0, 2, 2 | 4, 4];
    pub const RADIAL: u32 = 16;
    pub const FILL_START: u32 = 32;
    pub const FILL_END: u32 = 64;
    pub const CONIC: u32 = 128;
    pub const BORDER_LEFT: u32 = 256;
    pub const BORDER_TOP: u32 = 512;
    pub const BORDER_RIGHT: u32 = 1024;
    pub const BORDER_BOTTOM: u32 = 2048;
    pub const BORDER_ALL: u32 = BORDER_LEFT + BORDER_TOP + BORDER_RIGHT + BORDER_BOTTOM;
    pub const INVERT: u32 = 4096;
    pub const TEXT: u32 = 8192;
}

/// Information that the [`queue_ui_items`] system keeps internally.
#[derive(Default)]
pub struct QueueUiItemsLocalData {
    /// A list of all UI objects that were processed this frame.
    processed_ui_objects: HashSet<ChangedUiObject>,

    /// A list of UI objects that couldn't have pipeline keys generated for them
    /// on the previous frame.
    ///
    /// [`queue_ui_items`] will attempt to re-queue them on subsequent frames
    /// until they successfully enqueue.
    ///
    /// Typically, a pipeline key will fail to be generated because a dependent
    /// asset (e.g. a material) hasn't loaded yet.
    ui_objects_to_retry_this_frame: HashSet<(MainEntity, ChangedUiObject)>,

    /// A list of UI objects that couldn't have pipeline keys generated for them
    /// on this frame.
    ///
    /// [`queue_ui_items`] will attempt to re-queue them on subsequent frames.
    ui_objects_to_retry_next_frame: HashSet<(MainEntity, ChangedUiObject)>,
}

/// Processes changed render objects of a single type, inserting and removing
/// sorted phase items as necessary.
///
/// This system runs once per frame for each type of render object (normal UI
/// nodes, box shadows, gradients, etc.) It examines the list of changed render
/// nodes and adds and removes phase items as necessary.
pub fn queue_ui_items<E>(
    extracted_nodes: Res<UiRenderObjects<E>>,
    pipeline: Res<E::SpecializedRenderPipeline>,
    mut pipelines: ResMut<SpecializedRenderPipelines<E::SpecializedRenderPipeline>>,
    mut transparent_render_phases: ResMut<ViewSortedRenderPhases<TransparentUi>>,
    render_views: Query<E::ViewQueryData, With<ExtractedView>>,
    ui_camera_views: Query<&UiCameraView>,
    extracted_views: Query<&ExtractedView>,
    pipeline_cache: Res<PipelineCache>,
    draw_functions: Res<DrawFunctions<TransparentUi>>,
    mut local_data: Local<QueueUiItemsLocalData>,
    system_param: StaticSystemParam<E::PipelineKeySystemParam>,
) where
    E: UiRenderObject,
    <E::SpecializedRenderPipeline as SpecializedRenderPipeline>::Key: Send + Sync,
{
    let mut system_param = system_param.into_inner();
    let local_data = &mut *local_data;

    // Save the list of UI objects we need to attempt to re-queue this frame.
    // After processing current changes, we need to retry those in case they
    // succeed now.
    mem::swap(
        &mut local_data.ui_objects_to_retry_this_frame,
        &mut local_data.ui_objects_to_retry_next_frame,
    );
    local_data.ui_objects_to_retry_next_frame.clear();

    // Quick exit so we don't have to grab the lock on draw functions if there's
    // nothing to do.
    if extracted_nodes.changed.is_empty() && local_data.ui_objects_to_retry_this_frame.is_empty() {
        return;
    }

    let draw_function = draw_functions.read().id::<E::DrawFunctions>();

    // To avoid having to look up information about the camera over and over
    // again for each changed render object, we cache the most recent view we
    // looked at here.
    let mut maybe_cached_camera_view = None;

    // Loop over all changed nodes.
    for (main_entity, extracted_sub_ui_objects) in extracted_nodes.changed.iter() {
        // Examine all changed nodes (which includes nodes that were removed),
        // and remove all the corresponding phase items.
        local_data.processed_ui_objects.clear();
        for changed_ui_object in extracted_sub_ui_objects.iter() {
            if !local_data.processed_ui_objects.insert(*changed_ui_object) {
                continue;
            }

            // Refresh the cached camera view.
            CachedCameraView::<E::ViewPipelineKeyBuilder>::update::<E>(
                &mut maybe_cached_camera_view,
                changed_ui_object.camera_entity,
                &render_views,
                &ui_camera_views,
                &extracted_views,
            );
            let Some(ref mut cached_camera_view) = maybe_cached_camera_view else {
                continue;
            };

            // Fetch the transparent render phase, and remove the appropriate
            // phase items from it.
            let Some(transparent_render_phase) = transparent_render_phases
                .get_mut(&cached_camera_view.extracted_view.retained_view_entity)
            else {
                continue;
            };
            transparent_render_phase.remove(changed_ui_object.render_entity, *main_entity);
        }

        // If the UI node no longer exists, stop here.
        let Some((extracted_camera_entity, extracted_sub_uinodes)) =
            extracted_nodes.objects.get(main_entity)
        else {
            continue;
        };

        // Now look at all the changed UI nodes again. For each, attempt to add
        // the appropriate render objects of this type.
        for (render_entity, extracted_uinode) in extracted_sub_uinodes.iter() {
            try_add_phase_item(
                *main_entity,
                *render_entity,
                extracted_uinode,
                *extracted_camera_entity,
                &mut maybe_cached_camera_view,
                draw_function,
                &pipeline,
                &mut pipelines,
                &mut transparent_render_phases,
                &render_views,
                &ui_camera_views,
                &extracted_views,
                &pipeline_cache,
                &mut local_data.ui_objects_to_retry_next_frame,
                &mut system_param,
            );
        }
    }

    // Finally, attempt to re-queue all UI objects that we couldn't re-queue
    // last frame (usually because a dependent asset hadn't loaded yet).
    for (main_entity, changed_object) in local_data.ui_objects_to_retry_this_frame.drain() {
        let Some((extracted_camera_entity, extracted_sub_nodes)) =
            extracted_nodes.objects.get(&main_entity)
        else {
            continue;
        };
        let Some(extracted_uinode) = extracted_sub_nodes.get(&changed_object.render_entity) else {
            continue;
        };

        try_add_phase_item(
            main_entity,
            changed_object.render_entity,
            extracted_uinode,
            *extracted_camera_entity,
            &mut maybe_cached_camera_view,
            draw_function,
            &pipeline,
            &mut pipelines,
            &mut transparent_render_phases,
            &render_views,
            &ui_camera_views,
            &extracted_views,
            &pipeline_cache,
            &mut local_data.ui_objects_to_retry_next_frame,
            &mut system_param,
        );
    }

    // Attempts to enqueue a single phase item. If enqueuing fails because the
    // pipeline key couldn't be generated, then this function adds the item to
    // `ui_objects_to_retry_next_frame` and bails out.
    fn try_add_phase_item<'w, E>(
        main_entity: MainEntity,
        render_entity: Entity,
        extracted_uinode: &E,
        extracted_camera_entity: Entity,
        maybe_cached_camera_view: &mut Option<CachedCameraView<'w, E::ViewPipelineKeyBuilder>>,
        draw_function: DrawFunctionId,
        pipeline: &E::SpecializedRenderPipeline,
        pipelines: &mut SpecializedRenderPipelines<E::SpecializedRenderPipeline>,
        transparent_render_phases: &mut ViewSortedRenderPhases<TransparentUi>,
        render_views: &'w Query<E::ViewQueryData, With<ExtractedView>>,
        ui_camera_views: &'w Query<&UiCameraView>,
        extracted_views: &'w Query<&ExtractedView>,
        pipeline_cache: &PipelineCache,
        ui_objects_to_retry_next_frame: &mut HashSet<(MainEntity, ChangedUiObject)>,
        system_param: &mut <E::PipelineKeySystemParam as SystemParam>::Item<'_, '_>,
    ) where
        E: UiRenderObject,
    {
        // Refresh the cached camera view.
        CachedCameraView::<E::ViewPipelineKeyBuilder>::update::<E>(
            maybe_cached_camera_view,
            extracted_camera_entity,
            render_views,
            ui_camera_views,
            extracted_views,
        );
        let Some(ref mut cached_camera_view) = *maybe_cached_camera_view else {
            return;
        };

        // Fetch the transparent render phase.
        let Some(transparent_render_phase) = transparent_render_phases
            .get_mut(&cached_camera_view.extracted_view.retained_view_entity)
        else {
            return;
        };

        // Get the pipeline key, and specialize the pipeline. We need a
        // pipeline in order to construct a `TransparentUi` phase item.
        let Some(pipeline_key) =
            extracted_uinode.create_pipeline_key(cached_camera_view, system_param)
        else {
            // If we couldn't create the pipeline key, then make a note of this
            // item so that we will try to enqueue it later, and bail out.
            ui_objects_to_retry_next_frame.insert((
                main_entity,
                ChangedUiObject {
                    render_entity,
                    camera_entity: extracted_camera_entity,
                },
            ));
            return;
        };
        let pipeline = pipelines.specialize(pipeline_cache, pipeline, pipeline_key);

        // Add the phase item. Note that this phase item will be retained
        // from frame to frame.
        transparent_render_phase.add_retained(TransparentUi {
            draw_function,
            pipeline,
            entity: (render_entity, main_entity),
            sort_key: extracted_uinode.get_sort_key(),
            // batch_range will be calculated in prepare_uinodes
            batch_range: 0..0,
            extra_index: PhaseItemExtraIndex::None,
            indexed: true,
        });
    }
}

/// Information from the view necessary to construct the pipeline key for a
/// plain UI node.
pub struct UiNodePipelineKeyBuilder {
    /// Whether anti-aliasing is requested for UI nodes in this view.
    anti_alias: Option<UiAntiAlias>,
}

#[derive(Resource)]
pub struct UiTexturedBindGroups<E>
where
    E: UiRenderObject,
{
    pub values: HashMap<AssetId<<E::TexturedGpuAsset as RenderAsset>::SourceAsset>, BindGroup>,
    phantom: PhantomData<E>,
}

impl<E> Default for UiTexturedBindGroups<E>
where
    E: UiRenderObject,
{
    fn default() -> Self {
        UiTexturedBindGroups {
            values: HashMap::default(),
            phantom: PhantomData,
        }
    }
}

pub fn prepare_uinodes<E>(
    mut commands: Commands,
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
    pipeline_cache: Res<PipelineCache>,
    ui_meta: ResMut<UiMeta<E>>,
    extracted_uinodes: Res<UiRenderObjects<E>>,
    view_uniforms: Res<ViewUniforms>,
    globals_buffer: Res<GlobalsBuffer>,
    ui_pipeline: Res<E::SpecializedRenderPipeline>,
    mut maybe_textured_bind_groups: Option<ResMut<UiTexturedBindGroups<E>>>,
    gpu_assets: Res<RenderAssets<E::TexturedGpuAsset>>,
    mut phases: ResMut<ViewSortedRenderPhases<TransparentUi>>,
    extracted_assets: Res<ExtractedAssets<E::TexturedGpuAsset>>,
    (
        mut sparse_buffer_update_jobs,
        mut sparse_buffer_update_bind_groups,
        sparse_buffer_update_pipelines,
    ): (
        ResMut<SparseBufferUpdateJobs>,
        ResMut<SparseBufferUpdateBindGroups>,
        Res<SparseBufferUpdatePipelines>,
    ),
    mut previous_len: Local<usize>,
) where
    E: UiRenderObject,
{
    let ui_meta = ui_meta.into_inner();

    for changed_ui_objects in extracted_uinodes.changed.values() {
        for changed_ui_object in changed_ui_objects {
            ui_meta
                .instances
                .free_entity(changed_ui_object.render_entity);
        }
    }

    // If the underlying textured asset has changed, the prepared GPU textured
    // asset has (probably) changed
    if E::TEXTURED {
        let texture_bind_groups = maybe_textured_bind_groups.as_mut().unwrap();
        for id in extracted_assets
            .modified
            .iter()
            .chain(extracted_assets.removed.iter())
        {
            texture_bind_groups.values.remove(id);
        }
    }

    let globals_binding = if E::NEEDS_GLOBALS_UNIFORM {
        let Some(globals_binding) = globals_buffer.buffer.binding() else {
            return;
        };
        Some(globals_binding)
    } else {
        None
    };

    if let Some(view_binding) = view_uniforms.uniforms.binding() {
        let mut batches: Vec<(Entity, UiBatch<E>)> = Vec::with_capacity(*previous_len);

        let bind_group_layouts = E::bind_group_layouts(&ui_pipeline);

        ui_meta.vertices.clear();
        ui_meta.indices.clear();
        match ui_meta.instances {
            UiInstances::Immediate { ref mut instances } => instances.clear(),
            UiInstances::Retained {
                ref mut instance_index_buffer,
                ..
            } => instance_index_buffer.clear(),
        };

        let view_bind_group_layout = pipeline_cache.get_bind_group_layout(bind_group_layouts.view);
        ui_meta.view_bind_group = Some(if let Some(globals_binding) = globals_binding {
            render_device.create_bind_group(
                "ui_view_and_globals_bind_group",
                &view_bind_group_layout,
                &BindGroupEntries::sequential((view_binding, globals_binding)),
            )
        } else {
            render_device.create_bind_group(
                "ui_view_bind_group",
                &view_bind_group_layout,
                &BindGroupEntries::single(view_binding),
            )
        });

        ui_meta.vertices.extend(
            QUAD_VERTEX_POSITIONS
                .iter()
                .map(|&position| UiVertex { position }),
        );
        ui_meta.indices.extend(QUAD_INDICES);

        // Buffer indexes
        let mut vertices_index = QUAD_VERTEX_POSITIONS.len() as u32;
        let mut indices_index = QUAD_INDICES.len() as u32;

        for ui_phase in phases.values_mut() {
            let mut batch_item_index = 0;
            let mut batch_textured_asset_handle = None;

            for item_index in 0..ui_phase.items.len() {
                let item = &mut ui_phase.items[item_index];
                let Some(extracted_uinode) = extracted_uinodes
                    .objects
                    .get(&item.main_entity())
                    .and_then(|(_, sub_uinodes)| sub_uinodes.get(&item.entity()))
                else {
                    batch_textured_asset_handle = None;
                    continue;
                };

                // Initialize the batch range to be zero-length initially.
                // We'll extend it as we accumulate items into this batch.
                item.batch_range = (item_index as u32)..(item_index as u32);

                let textured_asset_id = extracted_uinode.textured_asset_id();
                let gpu_textured_asset = if E::TEXTURED {
                    gpu_assets.get(textured_asset_id)
                } else {
                    None
                };

                let mut existing_batch = batches.last_mut();

                if batch_textured_asset_handle.is_none()
                    || existing_batch.is_none()
                    || (batch_textured_asset_handle != Some(AssetId::default())
                        && textured_asset_id != AssetId::default()
                        && batch_textured_asset_handle != Some(textured_asset_id))
                {
                    if E::TEXTURED && gpu_textured_asset.is_none() {
                        continue;
                    }

                    batch_item_index = item_index;
                    batch_textured_asset_handle = Some(textured_asset_id);

                    let new_batch = UiBatch {
                        params: vec![],
                        textured_asset_id,
                        phantom: PhantomData,
                    };
                    batches.push((item.entity(), new_batch));

                    if let Some(gpu_asset) = gpu_textured_asset {
                        maybe_textured_bind_groups
                            .as_mut()
                            .unwrap()
                            .values
                            .entry(textured_asset_id)
                            .or_insert_with(|| {
                                E::get_or_create_textured_bind_group(
                                    &render_device,
                                    &pipeline_cache.get_bind_group_layout(
                                        E::textured_bind_group_layout(&ui_pipeline).expect(
                                            "Textured UI render objects must have texture bind group layouts"
                                        )
                                    ),
                                    gpu_asset,
                                ).expect(
                                    "Textured UI render objects must be able to get or create \
                                    textured bind groups"
                                )
                            });
                    }

                    existing_batch = batches.last_mut();
                } else if E::TEXTURED
                    && batch_textured_asset_handle == Some(AssetId::default())
                    && textured_asset_id != AssetId::default()
                {
                    if let Some(ref mut existing_batch) = existing_batch
                        && let Some(gpu_asset) = gpu_textured_asset
                    {
                        batch_textured_asset_handle = Some(textured_asset_id);
                        existing_batch.1.textured_asset_id = textured_asset_id;

                        maybe_textured_bind_groups
                            .as_mut()
                            .unwrap()
                            .values
                            .entry(textured_asset_id)
                            .or_insert_with(|| {
                                E::get_or_create_textured_bind_group(
                                    &render_device,
                                    &pipeline_cache.get_bind_group_layout(
                                        E::textured_bind_group_layout(&ui_pipeline).expect(
                                            "Textured UI render objects must have texture bind group layouts"
                                        )
                                    ),
                                    gpu_asset,
                                ).expect(
                                    "Textured UI render objects must be able to get or create \
                                    textured bind groups"
                                )
                            });
                    } else {
                        continue;
                    }
                }

                let batch = &mut existing_batch.expect("We should have a batch").1;
                let mut entity_instances = ui_meta.instances.get_entity_mut(item.entity());
                let (mut quad, mut is_invisible) = (UiQuad::default(), true);
                for quad_index in 0..extracted_uinode.quad_count() {
                    extracted_uinode.populate_quad(&mut quad, quad_index, gpu_textured_asset);

                    if extracted_uinode.clip().is_none() {
                        let instance_index = entity_instances
                            .get_or_insert_instance(quad_index, || quad.instance_data);
                        batch.push_simple_quad(instance_index);
                        is_invisible = false;
                        continue;
                    }

                    let vertices: [_; 4] = array::from_fn(|index| {
                        (quad.positions[index], QUAD_VERTEX_POSITIONS[index])
                    });
                    let clipped_quad = clip_polygon(extracted_uinode.clip(), &vertices, Vec2::lerp);
                    if clipped_quad.is_empty() {
                        continue;
                    }

                    let instance_index =
                        entity_instances.get_or_insert_instance(quad_index, || quad.instance_data);
                    let index_count = (clipped_quad.len() as u32 - 2) * 3;
                    batch.push(
                        vertices_index,
                        indices_index..(indices_index + index_count),
                        instance_index,
                    );

                    for &(_, local_position) in &clipped_quad {
                        ui_meta.vertices.push(UiVertex {
                            position: local_position,
                        });
                    }

                    for i in 1..clipped_quad.len() as u32 - 1 {
                        ui_meta.indices.push(0);
                        ui_meta.indices.push(i);
                        ui_meta.indices.push(i + 1);
                    }

                    vertices_index += clipped_quad.len() as u32;
                    indices_index += index_count;
                    is_invisible = false;
                }

                if is_invisible {
                    continue;
                }

                ui_phase.items[batch_item_index].batch_range_mut().end += 1;
            }
        }

        ui_meta.vertices.write_buffer(&render_device, &render_queue);
        ui_meta.indices.write_buffer(&render_device, &render_queue);

        match ui_meta.instances {
            UiInstances::Immediate { ref mut instances } => {
                instances.write_buffer(&render_device, &render_queue);
            }
            UiInstances::Retained {
                ref mut instances,
                ref mut instance_index_buffer,
                ref mut bind_group,
                ..
            } => {
                instances.write_buffers(&render_device, &render_queue);
                instances.prepare_to_populate_buffers(
                    &render_device,
                    &pipeline_cache,
                    &mut sparse_buffer_update_jobs,
                    &mut sparse_buffer_update_bind_groups,
                    &sparse_buffer_update_pipelines,
                );
                instance_index_buffer.write_buffer(&render_device, &render_queue);
                *bind_group = instances.buffer().map(|instances_buffer| {
                    let bind_group_layout =
                        pipeline_cache.get_bind_group_layout(bind_group_layouts.instances);
                    render_device.create_bind_group(
                        "UI instances bind group",
                        &bind_group_layout,
                        &BindGroupEntries::single(instances_buffer.as_entire_binding()),
                    )
                });
            }
        }

        *previous_len = batches.len();
        commands.try_insert_batch(batches);
    }
}

/// A render-world system that removes all [`UiBatch`] components.
///
/// They're currently rebuilt from scratch every frame, so we have to remove
/// them.
///
/// This is run during the render cleanup phase.
pub fn clear_batches<E>(mut commands: Commands, batches_query: Query<Entity, With<UiBatch<E>>>)
where
    E: UiRenderObject,
{
    for entity in &batches_query {
        commands.entity(entity).remove::<UiBatch<E>>();
    }
}

pub struct UiRenderObjectBindGroupLayouts<'a> {
    pub view: &'a BindGroupLayoutDescriptor,
    pub instances: &'a BindGroupLayoutDescriptor,
}

/// A render-world resource that holds all UI render objects of a single type.
///
/// Types of UI render objects include normal UI nodes, box shadows, gradients,
/// texture slices, and so forth.
///
/// This resource is retained from frame to frame. The various `extract_`
/// systems keep it up to date.
#[derive(Resource)]
pub struct UiRenderObjects<E>
where
    E: UiRenderObject,
{
    /// The list of UI objects grouped by their main-world entity, along with
    /// each group's target camera entity.
    ///
    /// This is a two-level data structure so that we can quickly remove all UI
    /// render objects associated with a main-world entity when it changes.
    pub objects: MainEntityHashMap<(Entity, EntityIndexMap<E>)>,

    /// UI render objects that changed this frame.
    pub changed: MainEntityHashMap<SmallVec<[ChangedUiObject; 4]>>,
}

impl<RO> Default for UiRenderObjects<RO>
where
    RO: UiRenderObject,
{
    fn default() -> UiRenderObjects<RO> {
        UiRenderObjects {
            objects: default(),
            changed: default(),
        }
    }
}

impl<E> UiRenderObjects<E>
where
    E: UiRenderObject,
{
    /// Spawns a new render entity corresponding to a UI node that changed since
    /// the previous frame, and records the information needed to render it.
    ///
    /// There can be multiple render entities corresponding to a single
    /// main-world UI node.
    pub fn add(
        &mut self,
        commands: &mut Commands,
        main_entity: MainEntity,
        extracted_camera_entity: Entity,
        object: E,
    ) {
        let render_entity = commands.spawn_empty().id();

        // Associate the newly spawned render world entity with the main world
        // entity and camera.
        self.objects
            .entry(main_entity)
            .or_insert_with(|| (extracted_camera_entity, Default::default()))
            .1
            .insert(render_entity, object);

        // Note that it's changed so that we queue it later.
        self.changed
            .entry(main_entity)
            .or_default()
            .push(ChangedUiObject {
                render_entity,
                camera_entity: extracted_camera_entity,
            });
    }

    pub fn add_render_entity(
        changed_objects: &mut SmallVec<[ChangedUiObject; 4]>,
        objects: &mut MainEntityHashMap<(Entity, EntityIndexMap<E>)>,
        commands: &mut Commands,
        main_entity: MainEntity,
        extracted_camera_entity: Entity,
        object: E,
    ) {
        let render_entity = commands.spawn_empty().id();

        // Associate the newly spawned render world entity with the main world
        // entity and camera.
        objects
            .entry(main_entity)
            .or_insert_with(|| (extracted_camera_entity, Default::default()))
            .1
            .insert(render_entity, object);

        // Note that it's changed so that we queue it later.
        changed_objects.push(ChangedUiObject {
            render_entity,
            camera_entity: extracted_camera_entity,
        });
    }
}

/// A piece of a UI node that renders using a specific shader and can enqueue
/// the phase items necessary to render itself.
///
/// Types of UI render objects include normal UI nodes, box shadows, gradients,
/// texture slices, and so forth.
pub trait UiRenderObject: Send + Sync + 'static {
    /// The set of draw functions that draw render objects of this type.
    type DrawFunctions: 'static;

    /// A type that stores all information taken from a view that will be needed
    /// to create the pipeline key for objects of this type.
    ///
    /// If no such data is required, this can be `()`.
    type ViewPipelineKeyBuilder: Sized;

    /// ECS data from the [`ExtractedView`] needed to construct the
    /// [`Self::ViewPipelineKeyBuilder`].
    ///
    /// If no such data is required, this can be `()`.
    type ViewQueryData: QueryData;

    /// The shader pipeline that renders UI objects of this type.
    type SpecializedRenderPipeline: SpecializedRenderPipeline + Resource;

    /// Any extra system data needed in order to create a pipeline key.
    type PipelineKeySystemParam: SystemParam;

    type InstanceData: Clone + Copy + Default + Pod + Zeroable + Send + Sync + 'static;

    /// If there is no texture, you can set this to `GpuImage`.
    type TexturedGpuAsset: RenderAsset;

    const TEXTURED: bool = false;

    const NEEDS_GLOBALS_UNIFORM: bool = false;

    /// Returns the sort order for this render object.
    fn get_sort_key(&self) -> FloatOrd;

    /// Extracts whatever render-world data is necessary to construct the
    /// pipeline key from a single view.
    fn create_view_pipeline_key_builder<'w, 's>(
        view: <<Self::ViewQueryData as QueryData>::ReadOnly as QueryData>::Item<'w, 's>,
    ) -> Self::ViewPipelineKeyBuilder;

    /// Creates the pipeline key needed to render this object from the
    /// information fetched in [`Self::create_view_pipeline_key_builder`].
    fn create_pipeline_key(
        &self,
        cached_camera_view: &CachedCameraView<Self::ViewPipelineKeyBuilder>,
        system_param: &mut SystemParamItem<Self::PipelineKeySystemParam>,
    ) -> Option<<Self::SpecializedRenderPipeline as SpecializedRenderPipeline>::Key>;

    fn bind_group_layouts(
        pipeline: &Self::SpecializedRenderPipeline,
    ) -> UiRenderObjectBindGroupLayouts<'_>;

    fn textured_asset_id(&self) -> AssetId<<Self::TexturedGpuAsset as RenderAsset>::SourceAsset> {
        AssetId::default()
    }

    fn textured_bind_group_layout(
        _pipeline: &Self::SpecializedRenderPipeline,
    ) -> Option<&BindGroupLayoutDescriptor> {
        None
    }

    fn get_or_create_textured_bind_group(
        _render_device: &RenderDevice,
        _layout: &BindGroupLayout,
        _gpu_asset: &Self::TexturedGpuAsset,
    ) -> Option<BindGroup> {
        None
    }

    fn clip(&self) -> Option<&CalculatedClip>;

    fn quad_count(&self) -> usize {
        1
    }

    fn populate_quad(
        &self,
        out_quad: &mut UiQuad<Self::InstanceData>,
        index: usize,
        gpu_asset: Option<&Self::TexturedGpuAsset>,
    );
}

#[derive(Default)]
pub struct UiQuad<ID>
where
    ID: Copy + Clone + Default,
{
    positions: [Vec2; 4],
    instance_data: ID,
}

/// Information about a single view that [`queue_ui_items`] caches.
///
/// All information needed to construct the pipeline key must be in this structure.
pub struct CachedCameraView<'w, PKB> {
    /// The render-world entity of the current camera.
    current_camera_entity: Entity,

    /// The [`ExtractedView`] structure corresponding to that camera.
    extracted_view: &'w ExtractedView,

    /// Any render-object-specific data needed to construct a pipeline key.
    ///
    /// The [`UiRenderObject::ViewPipelineKeyBuilder`] specifies this type.
    pipeline_key_builder: PKB,
}

impl<'w, PKB> CachedCameraView<'w, PKB> {
    fn update<E>(
        maybe_self: &'_ mut Option<CachedCameraView<'w, PKB>>,
        this_camera_entity: Entity,
        render_views: &'w Query<E::ViewQueryData, With<ExtractedView>>,
        ui_camera_views: &'w Query<&UiCameraView>,
        extracted_views: &'w Query<&ExtractedView>,
    ) where
        E: UiRenderObject<ViewPipelineKeyBuilder = PKB>,
    {
        if maybe_self.as_ref().is_some_and(|cached_camera_view| {
            cached_camera_view.current_camera_entity == this_camera_entity
        }) {
            return;
        }

        let maybe_current_view =
            ui_camera_views
                .get(this_camera_entity)
                .ok()
                .and_then(|default_camera_view| {
                    let view = extracted_views.get(default_camera_view.0).ok()?;
                    let pipeline_key_builder = render_views.get(this_camera_entity).ok()?;
                    Some((view, pipeline_key_builder))
                });

        let Some((extracted_view, pipeline_key_builder_item)) = maybe_current_view else {
            *maybe_self = None;
            return;
        };

        *maybe_self = Some(CachedCameraView {
            current_camera_entity: this_camera_entity,
            extracted_view,
            pipeline_key_builder: E::create_view_pipeline_key_builder(pipeline_key_builder_item),
        });
    }
}

fn pack_transform(transform: Affine2, rect_size: Vec2) -> Vec4 {
    Vec4::ZERO
        .with_xy(transform.matrix2.x_axis * rect_size.x)
        .with_zw(transform.matrix2.y_axis * rect_size.y)
}
