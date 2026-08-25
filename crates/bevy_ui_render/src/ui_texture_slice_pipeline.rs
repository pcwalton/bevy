use core::hash::Hash;

use crate::*;
use bevy_asset::*;
use bevy_color::{ColorToComponents, LinearRgba};
use bevy_ecs::system::*;
use bevy_image::prelude::*;
use bevy_math::{Affine2, FloatOrd, Rect, Vec2, Vec4};
use bevy_mesh::VertexBufferLayout;
use bevy_platform::collections::HashMap;
use bevy_render::{
    render_phase::*,
    render_resource::{binding_types::uniform_buffer, *},
    texture::GpuImage,
    view::*,
    Extract, ExtractSchedule, Render, RenderSystems,
};
use bevy_render::{sync_world::MainEntity, GpuResourceAppExt, RenderStartup};
use bevy_shader::Shader;
use bevy_sprite::{SliceScaleMode, SpriteImageMode, TextureSlicer};
use bevy_ui::widget::NodeImageMode;
use bevy_ui::{ComputedStackIndex, VisualBox};
use bevy_utils::default;
use binding_types::{sampler, texture_2d};
use bytemuck::{Pod, Zeroable};

pub struct UiTextureSlicerPlugin;

impl Plugin for UiTextureSlicerPlugin {
    fn build(&self, app: &mut App) {
        embedded_asset!(app, "ui_texture_slice.wesl");

        if let Some(render_app) = app.get_sub_app_mut(RenderApp) {
            render_app
                .add_render_command::<TransparentUi, DrawUiTextureSlices>()
                .init_resource::<ExtractedUiTextureSlices>()
                .init_gpu_resource::<UiMeta<ExtractedUiTextureSlice>>()
                .init_gpu_resource::<SpecializedRenderPipelines<UiTextureSlicePipeline>>()
                .add_systems(RenderStartup, init_ui_texture_slice_pipeline)
                .add_systems(
                    ExtractSchedule,
                    extract_ui_texture_slices.in_set(RenderUiSystems::ExtractTextureSlice),
                )
                .add_systems(
                    Render,
                    (
                        queue_ui_items::<ExtractedUiTextureSlice>.in_set(RenderSystems::Queue),
                        prepare_uinodes::<ExtractedUiTextureSlice>
                            .in_set(RenderSystems::PrepareBindGroups),
                    ),
                );
        }
    }
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct UiTextureSliceVertex {
    pub position: [f32; 3],
    pub uv: [f32; 2],
    pub color: [f32; 4],
    pub slices: [f32; 4],
    pub border: [f32; 4],
    pub repeat: [f32; 4],
    pub atlas: [f32; 4],
}

#[derive(Resource, Default)]
pub struct UiTextureSliceImageBindGroups {
    pub values: HashMap<AssetId<Image>, BindGroup>,
}

#[derive(Resource)]
pub struct UiTextureSlicePipeline {
    pub view_layout: BindGroupLayoutDescriptor,
    pub image_layout: BindGroupLayoutDescriptor,
    pub shader: Handle<Shader>,
}

pub fn init_ui_texture_slice_pipeline(mut commands: Commands, asset_server: Res<AssetServer>) {
    let view_layout = BindGroupLayoutDescriptor::new(
        "ui_texture_slice_view_layout",
        &BindGroupLayoutEntries::single(
            ShaderStages::VERTEX_FRAGMENT,
            uniform_buffer::<ViewUniform>(true),
        ),
    );

    let image_layout = BindGroupLayoutDescriptor::new(
        "ui_texture_slice_image_layout",
        &BindGroupLayoutEntries::sequential(
            ShaderStages::FRAGMENT,
            (
                texture_2d(TextureSampleType::Float { filterable: true }),
                sampler(SamplerBindingType::Filtering),
            ),
        ),
    );

    commands.insert_resource(UiTextureSlicePipeline {
        view_layout,
        image_layout,
        shader: load_embedded_asset!(asset_server.as_ref(), "ui_texture_slice.wesl"),
    });
}

#[derive(Clone, Copy, Hash, PartialEq, Eq)]
pub struct UiTextureSlicePipelineKey {
    pub target_format: TextureFormat,
}

impl SpecializedRenderPipeline for UiTextureSlicePipeline {
    type Key = UiTextureSlicePipelineKey;

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
                // normalized texture slicing lines (left, top, right, bottom)
                VertexFormat::Float32x4,
                // normalized target slicing lines (left, top, right, bottom)
                VertexFormat::Float32x4,
                // repeat values (horizontal side, vertical side, horizontal center, vertical center)
                VertexFormat::Float32x4,
                // normalized texture atlas rect (left, top, right, bottom)
                VertexFormat::Float32x4,
            ],
        );
        let shader_defs = Vec::new();

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
            layout: vec![self.view_layout.clone(), self.image_layout.clone()],
            label: Some("ui_texture_slice_pipeline".into()),
            ..default()
        }
    }
}

pub struct ExtractedUiTextureSlice {
    pub stack_index: u32,
    pub transform: Affine2,
    pub rect: Rect,
    pub atlas_rect: Option<Rect>,
    pub image: AssetId<Image>,
    pub clip: Option<CalculatedClip>,
    pub color: LinearRgba,
    pub image_scale_mode: SpriteImageMode,
    pub flip_x: bool,
    pub flip_y: bool,
    pub inverse_scale_factor: f32,
}

#[derive(Clone, Copy, Default)]
pub struct UiTextureSliceInstanceData {
    color: Vec4,
    slices: Vec4,
    border: Vec4,
    repeat: Vec4,
    atlas: Vec4,
}

impl UiRenderObject for ExtractedUiTextureSlice {
    type DrawFunctions = DrawUiTextureSlices;
    type ViewQueryData = ();
    type SpecializedRenderPipeline = UiTextureSlicePipeline;
    type ViewPipelineKeyBuilder = ();
    type PipelineKeySystemParam = ();
    type Vertex = UiTextureSliceVertex;
    type InstanceData = UiTextureSliceInstanceData;
    type TexturedGpuAsset = GpuImage;

    const TEXTURED: bool = true;

    fn get_sort_key(&self) -> FloatOrd {
        FloatOrd(self.stack_index as f32 + stack_z_offsets::IMAGE)
    }

    fn create_view_pipeline_key_builder<'w, 's>(_: ()) {}

    fn create_pipeline_key(
        &self,
        cached_camera_view: &CachedCameraView<Self::ViewPipelineKeyBuilder>,
        _: &mut SystemParamItem<Self::PipelineKeySystemParam>,
    ) -> Option<<Self::SpecializedRenderPipeline as SpecializedRenderPipeline>::Key> {
        Some(UiTextureSlicePipelineKey {
            target_format: cached_camera_view.extracted_view.target_format,
        })
    }

    fn view_bind_group_layout(
        pipeline: &Self::SpecializedRenderPipeline,
    ) -> &BindGroupLayoutDescriptor {
        &pipeline.view_layout
    }

    fn textured_asset_id(&self) -> AssetId<Image> {
        self.image
    }

    fn textured_bind_group_layout(
        pipeline: &Self::SpecializedRenderPipeline,
    ) -> Option<&BindGroupLayoutDescriptor> {
        Some(&pipeline.image_layout)
    }

    fn clip(&self) -> Option<&CalculatedClip> {
        self.clip.as_ref()
    }

    fn populate_quad(
        &self,
        out_quad: &mut UiQuad<Self::InstanceData>,
        quad_index: usize,
        gpu_textured_assets: &RenderAssets<Self::TexturedGpuAsset>,
        textured_asset_id: &AssetId<<Self::TexturedGpuAsset as RenderAsset>::SourceAsset>,
    ) {
        debug_assert_eq!(quad_index, 0);

        let uinode_rect = self.rect;

        let rect_size = uinode_rect.size();

        // Specify the corners of the node
        let positions =
            QUAD_VERTEX_POSITIONS.map(|pos| self.transform.transform_point2(pos * rect_size));

        let uvs = [Vec2::ZERO, Vec2::X, Vec2::ONE, Vec2::Y];

        let color = self.color.to_vec4();

        let batch_image_size = gpu_textured_assets
            .get(*textured_asset_id)
            .expect("Image was checked during batching and should still exist")
            .size_2d()
            .as_vec2();

        let (image_size, mut atlas) = if let Some(atlas) = self.atlas_rect {
            (
                atlas.size(),
                [
                    atlas.min.x / batch_image_size.x,
                    atlas.min.y / batch_image_size.y,
                    atlas.max.x / batch_image_size.x,
                    atlas.max.y / batch_image_size.y,
                ],
            )
        } else {
            (batch_image_size, [0., 0., 1., 1.])
        };

        if self.flip_x {
            atlas.swap(0, 2);
        }

        if self.flip_y {
            atlas.swap(1, 3);
        }

        let [slices, border, repeat] = compute_texture_slices(
            image_size,
            uinode_rect.size() * self.inverse_scale_factor,
            &self.image_scale_mode,
        );

        out_quad.instance_data = UiTextureSliceInstanceData {
            color,
            slices: slices.into(),
            border: border.into(),
            repeat: repeat.into(),
            atlas: atlas.into(),
        };

        for (&mut (ref mut out_position, ref mut out_uvs), (position, uv)) in out_quad
            .vertices
            .iter_mut()
            .zip(positions.iter().zip(uvs.iter()))
        {
            *out_position = *position;
            out_uvs.uv = *uv;
            out_uvs.point = Vec2::ZERO;
        }
    }

    fn create_vertex(
        quad: &UiQuad<Self::InstanceData>,
        position: Vec2,
        uvs: &UiQuadInterpolants,
    ) -> Self::Vertex {
        UiTextureSliceVertex {
            position: position.extend(0.0).into(),
            uv: uvs.uv.into(),
            color: quad.instance_data.color.into(),
            slices: quad.instance_data.slices.into(),
            border: quad.instance_data.border.into(),
            repeat: quad.instance_data.repeat.into(),
            atlas: quad.instance_data.atlas.into(),
        }
    }
}

/// A render-world resource that stores all texture slices in the scene.
pub type ExtractedUiTextureSlices = UiRenderObjects<ExtractedUiTextureSlice>;

pub fn extract_ui_texture_slices(
    mut commands: Commands,
    mut extracted_ui_slicers: ResMut<ExtractedUiTextureSlices>,
    texture_atlases: Extract<Res<Assets<TextureAtlasLayout>>>,
    slicers_query: Extract<
        Query<
            (
                Entity,
                &ComputedNode,
                &ComputedStackIndex,
                &UiGlobalTransform,
                &InheritedVisibility,
                Option<&CalculatedClip>,
                &ComputedUiTargetCamera,
                &ImageNode,
            ),
            Or<(
                Changed<ComputedNode>,
                Changed<ComputedStackIndex>,
                Changed<UiGlobalTransform>,
                Changed<InheritedVisibility>,
                Changed<CalculatedClip>,
                Changed<ComputedUiTargetCamera>,
                Changed<ImageNode>,
                // The `bevy_ui::widget::update_image_content_size_system` marks
                // `ImageNodeSize` as changed to indicate that the image metrics
                // and/or texture atlas layout changed, so we need to watch for
                // changes to that component, even though we don't read it.
                Changed<ImageNodeSize>,
            )>,
        >,
    >,
    unfiltered_slicers_query: Extract<
        Query<(
            Entity,
            &ComputedNode,
            &ComputedStackIndex,
            &UiGlobalTransform,
            &InheritedVisibility,
            Option<&CalculatedClip>,
            &ComputedUiTargetCamera,
            &ImageNode,
        )>,
    >,
    camera_map: Extract<UiCameraMap>,
    (
        mut removed_computed_node_query,
        mut removed_computed_stack_index_query,
        mut removed_ui_global_transform_query,
        mut removed_inherited_visibility_query,
        mut removed_calculated_clip_query,
        mut removed_computed_ui_target_camera_query,
        mut removed_image_node_query,
    ): (
        Extract<RemovedComponents<ComputedNode>>,
        Extract<RemovedComponents<ComputedStackIndex>>,
        Extract<RemovedComponents<UiGlobalTransform>>,
        Extract<RemovedComponents<InheritedVisibility>>,
        Extract<RemovedComponents<CalculatedClip>>,
        Extract<RemovedComponents<ComputedUiTargetCamera>>,
        Extract<RemovedComponents<ImageNode>>,
    ),
    mut nodes_processed_this_frame: Local<MainEntityHashSet>,
) {
    nodes_processed_this_frame.clear();
    extracted_ui_slicers.changed.clear();
    let mut camera_mapper = camera_map.get_mapper();

    for (entity, uinode, stack_index, transform, inherited_visibility, clip, camera, image) in
        slicers_query.iter().chain(
            removed_calculated_clip_query
                .read()
                .filter_map(|entity| unfiltered_slicers_query.get(entity).ok()),
        )
    {
        let main_entity = MainEntity::from(entity);

        // If there were any previous UI slices for this entity, despawn them
        // and record them as changed so the render phase entry can be removed.
        if let Some((prev_camera_entity, mut slices)) =
            extracted_ui_slicers.objects.remove(&main_entity)
        {
            let changed = extracted_ui_slicers.changed.entry(main_entity).or_default();
            for (render_entity, _) in slices.drain(..) {
                commands.entity(render_entity).despawn();
                changed.push(ChangedUiObject {
                    render_entity,
                    camera_entity: prev_camera_entity,
                });
            }
        }

        let visual_box = match image.visual_box {
            VisualBox::ContentBox => uinode.content_box(),
            VisualBox::PaddingBox => uinode.padding_box(),
            VisualBox::BorderBox => uinode.border_box(),
        };

        // Skip invisible images
        if !inherited_visibility.get()
            || image.color.is_fully_transparent()
            || image.image.id() == TRANSPARENT_IMAGE_HANDLE.id()
            || visual_box.size().cmple(Vec2::ZERO).any()
        {
            continue;
        }

        let image_scale_mode = match image.image_mode.clone() {
            NodeImageMode::Sliced(texture_slicer) => SpriteImageMode::Sliced(texture_slicer),
            NodeImageMode::Tiled {
                tile_x,
                tile_y,
                stretch_value,
            } => SpriteImageMode::Tiled {
                tile_x,
                tile_y,
                stretch_value,
            },
            _ => continue,
        };

        let Some(extracted_camera_entity) = camera_mapper.map(camera) else {
            continue;
        };
        if let Some((camera_entity, _)) = extracted_ui_slicers.objects.get_mut(&main_entity) {
            *camera_entity = extracted_camera_entity;
        }

        nodes_processed_this_frame.insert(main_entity);

        let atlas_rect = image
            .texture_atlas
            .as_ref()
            .and_then(|s| s.texture_rect(&texture_atlases))
            .map(|r| r.as_rect());

        let atlas_rect = match (atlas_rect, image.rect) {
            (None, None) => None,
            (None, Some(image_rect)) => Some(image_rect),
            (Some(atlas_rect), None) => Some(atlas_rect),
            (Some(atlas_rect), Some(mut image_rect)) => {
                image_rect.min += atlas_rect.min;
                image_rect.max += atlas_rect.min;
                Some(image_rect)
            }
        };

        extracted_ui_slicers.add(
            &mut commands,
            main_entity,
            extracted_camera_entity,
            ExtractedUiTextureSlice {
                stack_index: stack_index.0,
                transform: Affine2::from(*transform)
                    * Affine2::from_translation(visual_box.center()),
                color: image.color.into(),
                rect: Rect {
                    min: Vec2::ZERO,
                    max: visual_box.size(),
                },
                clip: clip.cloned(),
                image: image.image.id(),
                image_scale_mode,
                atlas_rect,
                flip_x: image.flip_x,
                flip_y: image.flip_y,
                inverse_scale_factor: uinode.inverse_scale_factor,
            },
        );
    }

    // Only remove the render-world data if we didn't handle the node above.
    // It's possible that a relevant component was removed and added in the same
    // frame.
    for main_entity in removed_computed_node_query
        .read()
        .chain(removed_computed_stack_index_query.read())
        .chain(removed_ui_global_transform_query.read())
        .chain(removed_inherited_visibility_query.read())
        .chain(removed_computed_ui_target_camera_query.read())
        .chain(removed_image_node_query.read())
    {
        let main_entity = MainEntity::from(main_entity);
        if nodes_processed_this_frame.contains(&main_entity) {
            continue;
        }
        let Some((prev_camera_entity, mut extracted_nodes)) =
            extracted_ui_slicers.objects.remove(&main_entity)
        else {
            continue;
        };
        let changed = extracted_ui_slicers.changed.entry(main_entity).or_default();
        for (render_entity, _) in extracted_nodes.drain(..) {
            commands.entity(render_entity).despawn();
            changed.push(ChangedUiObject {
                render_entity,
                camera_entity: prev_camera_entity,
            });
        }
    }
}

pub type DrawUiTextureSlices = (
    SetItemPipeline,
    SetUiViewBindGroup<ExtractedUiTextureSlice, 0>,
    SetUiTextureBindGroup<ExtractedUiTextureSlice, 1>,
    DrawUiRenderObject<ExtractedUiTextureSlice>,
);

fn compute_texture_slices(
    image_size: Vec2,
    target_size: Vec2,
    image_scale_mode: &SpriteImageMode,
) -> [[f32; 4]; 3] {
    match image_scale_mode {
        SpriteImageMode::Sliced(TextureSlicer {
            border: border_rect,
            center_scale_mode,
            sides_scale_mode,
            max_corner_scale,
        }) => {
            let min_coeff = (target_size / image_size)
                .min_element()
                .min(*max_corner_scale);

            // calculate the normalized extents of the nine-patched image slices
            let slices = [
                border_rect.min_inset.x / image_size.x,
                border_rect.min_inset.y / image_size.y,
                1. - border_rect.max_inset.x / image_size.x,
                1. - border_rect.max_inset.y / image_size.y,
            ];

            // calculate the normalized extents of the target slices
            let border = [
                (border_rect.min_inset.x / target_size.x) * min_coeff,
                (border_rect.min_inset.y / target_size.y) * min_coeff,
                1. - (border_rect.max_inset.x / target_size.x) * min_coeff,
                1. - (border_rect.max_inset.y / target_size.y) * min_coeff,
            ];

            let image_side_width = image_size.x * (slices[2] - slices[0]);
            let image_side_height = image_size.y * (slices[3] - slices[1]);
            let target_side_width = target_size.x * (border[2] - border[0]);
            let target_side_height = target_size.y * (border[3] - border[1]);

            // compute the number of times to repeat the side and center slices when tiling along each axis
            // if the returned value is `1.` the slice will be stretched to fill the axis.
            let repeat_side_x =
                compute_tiled_subaxis(image_side_width, target_side_width, sides_scale_mode);
            let repeat_side_y =
                compute_tiled_subaxis(image_side_height, target_side_height, sides_scale_mode);
            let repeat_center_x =
                compute_tiled_subaxis(image_side_width, target_side_width, center_scale_mode);
            let repeat_center_y =
                compute_tiled_subaxis(image_side_height, target_side_height, center_scale_mode);

            [
                slices,
                border,
                [
                    repeat_side_x,
                    repeat_side_y,
                    repeat_center_x,
                    repeat_center_y,
                ],
            ]
        }
        SpriteImageMode::Tiled {
            tile_x,
            tile_y,
            stretch_value,
        } => {
            let rx = compute_tiled_axis(*tile_x, image_size.x, target_size.x, *stretch_value);
            let ry = compute_tiled_axis(*tile_y, image_size.y, target_size.y, *stretch_value);
            [[0., 0., 1., 1.], [0., 0., 1., 1.], [1., 1., rx, ry]]
        }
        SpriteImageMode::Auto => {
            unreachable!("Slices can not be computed for SpriteImageMode::Stretch")
        }
        SpriteImageMode::Scale(_) => {
            unreachable!("Slices can not be computed for SpriteImageMode::Scale")
        }
    }
}

fn compute_tiled_axis(tile: bool, image_extent: f32, target_extent: f32, stretch: f32) -> f32 {
    if tile {
        let s = image_extent * stretch;
        target_extent / s
    } else {
        1.
    }
}

fn compute_tiled_subaxis(image_extent: f32, target_extent: f32, mode: &SliceScaleMode) -> f32 {
    match mode {
        SliceScaleMode::Stretch => 1.,
        SliceScaleMode::Tile { stretch_value } => {
            let s = image_extent * *stretch_value;
            target_extent / s
        }
    }
}
