use crate::ui_material::{MaterialNode, UiMaterial, UiMaterialKey};
use crate::*;
use bevy_asset::*;
use bevy_ecs::system::{lifetimeless::SRes, *};
use bevy_math::{vec4, Affine2, FloatOrd, Rect, Vec2, Vec4};
use bevy_mesh::VertexBufferLayout;
use bevy_render::material_bind_groups::FallbackBuffer;
use bevy_render::storage::GpuShaderBuffer;
use bevy_render::{
    globals::GlobalsUniform,
    render_asset::{PrepareAssetError, RenderAsset, RenderAssetPlugin, RenderAssets},
    render_phase::*,
    render_resource::{binding_types::uniform_buffer, *},
    renderer::RenderDevice,
    sync_world::MainEntity,
    view::*,
    Extract, ExtractSchedule, Render, RenderSystems,
};
use bevy_render::{GpuResourceAppExt, RenderApp, RenderStartup};
use bevy_shader::{load_shader_library, Shader, ShaderRef};
use bevy_sprite::BorderRect;
use bevy_ui::ComputedStackIndex;
use bevy_utils::default;
use bytemuck::{Pod, Zeroable};
use core::{hash::Hash, marker::PhantomData};

/// Adds the necessary ECS resources and render logic to enable rendering entities using the given
/// [`UiMaterial`] asset type (which includes [`UiMaterial`] types).
pub struct UiMaterialPlugin<M: UiMaterial>(PhantomData<M>);

impl<M: UiMaterial> Default for UiMaterialPlugin<M> {
    fn default() -> Self {
        Self(Default::default())
    }
}

impl<M: UiMaterial> Plugin for UiMaterialPlugin<M>
where
    M::Data: PartialEq + Eq + Hash + Clone,
{
    fn build(&self, app: &mut App) {
        load_shader_library!(app, "ui_vertex_output.wesl");

        embedded_asset!(app, "ui_material.wesl");

        app.init_asset::<M>()
            .register_type::<MaterialNode<M>>()
            .add_plugins(RenderAssetPlugin::<
                PreparedUiMaterial<M>,
                (GpuImage, GpuShaderBuffer),
            >::default());

        if let Some(render_app) = app.get_sub_app_mut(RenderApp) {
            render_app
                .add_render_command::<TransparentUi, DrawUiMaterial<M>>()
                .init_resource::<ExtractedUiMaterialNodes<M>>()
                .init_gpu_resource::<UiMeta<ExtractedUiMaterialNode<M>>>()
                .init_gpu_resource::<UiTexturedBindGroups<ExtractedUiMaterialNode<M>>>()
                .init_gpu_resource::<SpecializedRenderPipelines<UiMaterialPipeline<M>>>()
                .add_systems(RenderStartup, init_ui_material_pipeline::<M>)
                .add_systems(
                    ExtractSchedule,
                    extract_ui_material_nodes::<M>.in_set(RenderUiSystems::ExtractBackgrounds),
                )
                .add_systems(
                    Render,
                    (
                        queue_ui_items::<ExtractedUiMaterialNode<M>>.in_set(RenderSystems::Queue),
                        prepare_uinodes::<ExtractedUiMaterialNode<M>>
                            .in_set(RenderSystems::PrepareBindGroups),
                        clear_batches::<ExtractedUiMaterialNode<M>>.in_set(RenderSystems::Cleanup),
                    ),
                );
        }
    }
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct UiMaterialVertex {
    pub position: [f32; 3],
    pub uv: [f32; 2],
    pub size: [f32; 2],
    pub border: [f32; 4],
    pub radius: [[f32; 4]; 2],
}

#[derive(Clone, Default)]
pub struct ExtractedUiMaterialNodeInstanceData {
    size: Vec2,
    border: Vec4,
    radius: [Vec4; 2],
}

/// Render pipeline data for a given [`UiMaterial`]
#[derive(Resource)]
pub struct UiMaterialPipeline<M: UiMaterial> {
    pub ui_layout: BindGroupLayoutDescriptor,
    pub view_layout: BindGroupLayoutDescriptor,
    pub vertex_shader: Handle<Shader>,
    pub fragment_shader: Handle<Shader>,
    marker: PhantomData<M>,
}

impl<M: UiMaterial> SpecializedRenderPipeline for UiMaterialPipeline<M>
where
    M::Data: PartialEq + Eq + Hash + Clone,
{
    type Key = UiMaterialKey<M>;

    fn specialize(&self, key: Self::Key) -> RenderPipelineDescriptor {
        let vertex_layout = VertexBufferLayout::from_vertex_formats(
            VertexStepMode::Vertex,
            vec![
                // position
                VertexFormat::Float32x3,
                // uv
                VertexFormat::Float32x2,
                // size
                VertexFormat::Float32x2,
                // border widths
                VertexFormat::Float32x4,
                // border radius x values (top left, top right, bottom right, bottom left)
                VertexFormat::Float32x4,
                // border radius y values (top left, top right, bottom right, bottom left)
                VertexFormat::Float32x4,
            ],
        );
        let shader_defs = Vec::new();

        let mut descriptor = RenderPipelineDescriptor {
            vertex: VertexState {
                shader: self.vertex_shader.clone(),
                shader_defs: shader_defs.clone(),
                buffers: vec![vertex_layout],
                ..default()
            },
            fragment: Some(FragmentState {
                shader: self.fragment_shader.clone(),
                shader_defs,
                targets: vec![Some(ColorTargetState {
                    format: key.target_format,
                    blend: Some(BlendState::ALPHA_BLENDING),
                    write_mask: ColorWrites::ALL,
                })],
                ..default()
            }),
            label: Some("ui_material_pipeline".into()),
            ..default()
        };

        descriptor.layout = vec![self.view_layout.clone(), self.ui_layout.clone()];

        M::specialize(&mut descriptor, key);

        descriptor
    }
}

pub fn init_ui_material_pipeline<M: UiMaterial>(
    mut commands: Commands,
    asset_server: Res<AssetServer>,
    render_device: Res<RenderDevice>,
) {
    let ui_layout = M::bind_group_layout_descriptor(&render_device);

    let view_layout = BindGroupLayoutDescriptor::new(
        "ui_view_layout",
        &BindGroupLayoutEntries::sequential(
            ShaderStages::VERTEX_FRAGMENT,
            (
                uniform_buffer::<ViewUniform>(true),
                uniform_buffer::<GlobalsUniform>(false),
            ),
        ),
    );

    let load_default = || load_embedded_asset!(asset_server.as_ref(), "ui_material.wesl");

    commands.insert_resource(UiMaterialPipeline::<M> {
        ui_layout,
        view_layout,
        vertex_shader: match M::vertex_shader() {
            ShaderRef::Default => load_default(),
            ShaderRef::Handle(handle) => handle,
            ShaderRef::Path(path) => asset_server.load(path),
        },
        fragment_shader: match M::fragment_shader() {
            ShaderRef::Default => load_default(),
            ShaderRef::Handle(handle) => handle,
            ShaderRef::Path(path) => asset_server.load(path),
        },
        marker: PhantomData,
    });
}

pub type DrawUiMaterial<M> = (
    SetItemPipeline,
    SetUiViewBindGroup<ExtractedUiMaterialNode<M>, 0>,
    SetUiTextureBindGroup<ExtractedUiMaterialNode<M>, 1>,
    DrawUiRenderObject<ExtractedUiMaterialNode<M>>,
);

pub struct ExtractedUiMaterialNode<M: UiMaterial> {
    pub stack_index: u32,
    pub transform: Affine2,
    pub rect: Rect,
    pub border: BorderRect,
    pub border_radius: [Vec4; 2],
    pub material: AssetId<M>,
    pub clip: Option<CalculatedClip>,
}

/// A render-world resource that stores all material nodes in the scene.
pub type ExtractedUiMaterialNodes<M> = UiRenderObjects<ExtractedUiMaterialNode<M>>;

impl<M> UiRenderObject for ExtractedUiMaterialNode<M>
where
    M: UiMaterial,
    M::Data: PartialEq + Eq + Hash + Clone,
{
    type DrawFunctions = DrawUiMaterial<M>;
    type ViewPipelineKeyBuilder = ();
    type ViewQueryData = ();
    type SpecializedRenderPipeline = UiMaterialPipeline<M>;
    type PipelineKeySystemParam = Res<'static, RenderAssets<PreparedUiMaterial<M>>>;
    type Vertex = UiMaterialVertex;
    type InstanceData = ExtractedUiMaterialNodeInstanceData;
    type TexturedGpuAsset = PreparedUiMaterial<M>;

    const TEXTURED: bool = true;

    fn get_sort_key(&self) -> FloatOrd {
        FloatOrd(self.stack_index as f32 + M::stack_z_offset())
    }

    fn create_view_pipeline_key_builder<'w, 's>(_: ()) {}

    fn create_pipeline_key(
        &self,
        cached_camera_view: &CachedCameraView<Self::ViewPipelineKeyBuilder>,
        render_materials: &mut SystemParamItem<Self::PipelineKeySystemParam>,
    ) -> Option<<Self::SpecializedRenderPipeline as SpecializedRenderPipeline>::Key> {
        render_materials
            .get(self.material)
            .map(|material| UiMaterialKey {
                target_format: cached_camera_view.extracted_view.target_format,
                bind_group_data: material.key.clone(),
            })
    }

    fn view_bind_group_layout(
        pipeline: &Self::SpecializedRenderPipeline,
    ) -> &BindGroupLayoutDescriptor {
        &pipeline.view_layout
    }

    const NEEDS_GLOBALS_UNIFORM: bool = true;

    fn textured_asset_id(&self) -> AssetId<M> {
        self.material
    }

    fn textured_bind_group_layout(
        pipeline: &Self::SpecializedRenderPipeline,
    ) -> Option<&BindGroupLayoutDescriptor> {
        Some(&pipeline.ui_layout)
    }

    fn get_or_create_textured_bind_group(
        _: &RenderDevice,
        _: &BindGroupLayout,
        gpu_asset: &PreparedUiMaterial<M>,
    ) -> Option<BindGroup> {
        Some(gpu_asset.bind_group.clone())
    }

    fn clip(&self) -> Option<&CalculatedClip> {
        self.clip.as_ref()
    }

    fn populate_quad(
        &self,
        out_quad: &mut UiQuad<Self::InstanceData>,
        index: usize,
        _gpu_asset: Option<&PreparedUiMaterial<M>>,
    ) {
        debug_assert_eq!(index, 0);

        let rect_size = self.rect.size();

        let positions =
            QUAD_VERTEX_POSITIONS.map(|pos| self.transform.transform_point2(pos * rect_size));

        let uvs = [
            Vec2::new(self.rect.min.x, self.rect.min.y),
            Vec2::new(self.rect.max.x, self.rect.min.y),
            Vec2::new(self.rect.max.x, self.rect.max.y),
            Vec2::new(self.rect.min.x, self.rect.max.y),
        ]
        .map(|pos| pos / self.rect.max);

        out_quad.instance_data = ExtractedUiMaterialNodeInstanceData {
            size: rect_size,
            border: vec4(
                self.border.min_inset.x,
                self.border.min_inset.y,
                self.border.max_inset.x,
                self.border.max_inset.y,
            ),
            radius: self.border_radius,
        };

        for (&mut (ref mut out_position, ref mut out_uvs), (position, uv)) in out_quad
            .vertices
            .iter_mut()
            .zip(positions.iter().zip(uvs.iter()))
        {
            *out_position = *position;
            out_uvs.uv_a = *uv;
            out_uvs.uv_b = default();
        }
    }

    fn create_vertex(
        quad: &UiQuad<Self::InstanceData>,
        position: Vec2,
        uvs: &UiQuadUvs,
    ) -> Self::Vertex {
        UiMaterialVertex {
            position: position.extend(1.0).into(),
            uv: uvs.uv_a.into(),
            size: quad.instance_data.size.into(),
            border: quad.instance_data.border.into(),
            radius: quad.instance_data.radius.map(Into::into),
        }
    }
}

pub fn extract_ui_material_nodes<M>(
    mut commands: Commands,
    mut extracted_uinodes: ResMut<ExtractedUiMaterialNodes<M>>,
    materials: Extract<Res<Assets<M>>>,
    uinode_query: Extract<
        Query<
            (
                Entity,
                &ComputedNode,
                &ComputedStackIndex,
                &UiGlobalTransform,
                &MaterialNode<M>,
                &InheritedVisibility,
                Option<&CalculatedClip>,
                &ComputedUiTargetCamera,
            ),
            Or<(
                Changed<ComputedNode>,
                Changed<ComputedStackIndex>,
                Changed<UiGlobalTransform>,
                Changed<MaterialNode<M>>,
                Changed<InheritedVisibility>,
                Changed<CalculatedClip>,
                Changed<ComputedUiTargetCamera>,
            )>,
        >,
    >,
    unfiltered_uinode_query: Extract<
        Query<(
            Entity,
            &ComputedNode,
            &ComputedStackIndex,
            &UiGlobalTransform,
            &MaterialNode<M>,
            &InheritedVisibility,
            Option<&CalculatedClip>,
            &ComputedUiTargetCamera,
        )>,
    >,
    camera_map: Extract<UiCameraMap>,
    (
        mut removed_computed_node_query,
        mut removed_computed_stack_index_query,
        mut removed_ui_global_transform_query,
        mut removed_material_node_query,
        mut removed_inherited_visibility_query,
        mut removed_calculated_clip_query,
        mut removed_computed_ui_target_camera_query,
    ): (
        Extract<RemovedComponents<ComputedNode>>,
        Extract<RemovedComponents<ComputedStackIndex>>,
        Extract<RemovedComponents<UiGlobalTransform>>,
        Extract<RemovedComponents<MaterialNode<M>>>,
        Extract<RemovedComponents<InheritedVisibility>>,
        Extract<RemovedComponents<CalculatedClip>>,
        Extract<RemovedComponents<ComputedUiTargetCamera>>,
    ),
    mut nodes_to_reextract_next_frame: Local<MainEntityHashSet>,
    mut nodes_processed_this_frame: Local<MainEntityHashSet>,
) where
    M: UiMaterial,
    M::Data: PartialEq + Eq + Hash + Clone,
{
    nodes_processed_this_frame.clear();
    extracted_uinodes.changed.clear();
    let mut camera_mapper = camera_map.get_mapper();
    let nodes_to_reextract = mem::take(&mut *nodes_to_reextract_next_frame);

    for (
        entity,
        computed_node,
        stack_index,
        transform,
        handle,
        inherited_visibility,
        clip,
        camera,
    ) in uinode_query.iter().chain(
        nodes_to_reextract
            .into_iter()
            .map(|main_entity| main_entity.entity())
            .chain(removed_calculated_clip_query.read())
            .filter_map(|entity| unfiltered_uinode_query.get(entity).ok()),
    ) {
        let main_entity = MainEntity::from(entity);

        // Make sure we don't process the same node more than once.
        // This is possible if the node was marked for reextraction on the
        // previous frame and was also otherwise changed on this frame.
        if nodes_processed_this_frame.contains(&main_entity) {
            continue;
        }
        // If there were any previous UI nodes for this entity, despawn them
        // and record them as changed so the render phase entry can be
        // removed.
        if let Some((prev_camera_entity, mut nodes)) =
            extracted_uinodes.objects.remove(&main_entity)
        {
            let changed = extracted_uinodes.changed.entry(main_entity).or_default();
            for (render_entity, _) in nodes.drain(..) {
                commands.entity(render_entity).despawn();
                changed.push(ChangedUiObject {
                    render_entity,
                    camera_entity: prev_camera_entity,
                });
            }
        }

        // skip invisible nodes
        if !inherited_visibility.get() || computed_node.is_empty() {
            continue;
        }

        // If the material hasn't finished loading, skip the entity, and
        // remember that we did so that we reextract the node next frame.
        if !materials.contains(handle) {
            nodes_to_reextract_next_frame.insert(main_entity);
            continue;
        }

        let Some(extracted_camera_entity) = camera_mapper.map(camera) else {
            continue;
        };
        if let Some((camera_entity, _)) = extracted_uinodes.objects.get_mut(&main_entity) {
            *camera_entity = extracted_camera_entity;
        }

        nodes_processed_this_frame.insert(main_entity);

        extracted_uinodes.add(
            &mut commands,
            main_entity,
            extracted_camera_entity,
            ExtractedUiMaterialNode {
                stack_index: stack_index.0,
                transform: transform.into(),
                material: handle.id(),
                rect: Rect {
                    min: Vec2::ZERO,
                    max: computed_node.size(),
                },
                border: computed_node.border(),
                border_radius: computed_node.border_radius().into(),
                clip: clip.cloned(),
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
        .chain(removed_material_node_query.read())
        .chain(removed_inherited_visibility_query.read())
        .chain(removed_computed_ui_target_camera_query.read())
    {
        let main_entity = MainEntity::from(main_entity);
        if nodes_processed_this_frame.contains(&main_entity) {
            continue;
        }
        let Some((prev_camera_entity, mut extracted_nodes)) =
            extracted_uinodes.objects.remove(&main_entity)
        else {
            continue;
        };
        let changed = extracted_uinodes.changed.entry(main_entity).or_default();
        for (render_entity, _) in extracted_nodes.drain(..) {
            commands.entity(render_entity).despawn();
            changed.push(ChangedUiObject {
                render_entity,
                camera_entity: prev_camera_entity,
            });
        }
    }
}

pub struct PreparedUiMaterial<T: UiMaterial> {
    pub bindings: BindingResources,
    pub bind_group: BindGroup,
    pub key: T::Data,
}

impl<M: UiMaterial> RenderAsset for PreparedUiMaterial<M> {
    type SourceAsset = M;

    type Param = (
        SRes<RenderDevice>,
        SRes<PipelineCache>,
        SRes<FallbackBuffer>,
        SRes<RenderAssets<GpuShaderBuffer>>,
        SRes<UiMaterialPipeline<M>>,
        M::Param,
    );

    fn prepare_asset(
        material: Self::SourceAsset,
        _: AssetId<Self::SourceAsset>,
        (
            render_device,
            pipeline_cache,
            fallback_buffer,
            shader_buffer_assets,
            pipeline,
            material_param,
        ): &mut SystemParamItem<Self::Param>,
        _: Option<&Self>,
    ) -> Result<Self, PrepareAssetError<Self::SourceAsset>> {
        let bind_group_data = material.bind_group_data();
        match material.as_bind_group(
            &pipeline.ui_layout.clone(),
            render_device,
            pipeline_cache,
            fallback_buffer,
            shader_buffer_assets,
            material_param,
        ) {
            Ok(prepared) => Ok(PreparedUiMaterial {
                bindings: prepared.bindings,
                bind_group: prepared.bind_group,
                key: bind_group_data,
            }),
            Err(AsBindGroupError::RetryNextUpdate) => {
                Err(PrepareAssetError::RetryNextUpdate(material))
            }
            Err(other) => Err(PrepareAssetError::AsBindGroupError(other)),
        }
    }
}
