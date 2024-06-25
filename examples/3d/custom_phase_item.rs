//! Demonstrates enqueuing custom phase items.

use bevy::{
    core_pipeline::core_3d::{Opaque3d, Opaque3dBinKey},
    ecs::{
        query::ROQueryItem,
        system::{
            lifetimeless::{Read, SRes},
            SystemParamItem,
        },
    },
    math::vec3,
    prelude::*,
    render::{
        extract_component::{ExtractComponent, ExtractComponentPlugin},
        render_phase::{
            AddRenderCommand, CachedRenderPipelinePhaseItem, DrawFunctions, PhaseItem,
            RenderCommand, RenderCommandResult, TrackedRenderPass, ViewBinnedRenderPhases,
        },
        render_resource::{
            BufferUsages, CachedRenderPipelineId, ColorTargetState, ColorWrites, FragmentState,
            IndexFormat, MultisampleState, PipelineCache, PrimitiveState, RawBufferVec,
            RenderPipelineDescriptor, SpecializedRenderPipeline, SpecializedRenderPipelines,
            TextureFormat, VertexState,
        },
        renderer::{RenderDevice, RenderQueue},
        texture::BevyDefault as _,
        view::ExtractedView,
        Render, RenderApp, RenderSet,
    },
};
use bytemuck::{Pod, Zeroable};

#[derive(Clone, Component, ExtractComponent)]
struct CustomRenderedEntity;

#[derive(Component)]
struct CustomPhaseRenderPipeline {
    pipeline_id: CachedRenderPipelineId,
}

#[derive(Resource)]
struct CustomPhasePipeline {
    shader: Handle<Shader>,
}

struct SetCustomPhaseItemPipeline;

struct DrawCustomPhaseItem;

#[derive(Resource)]
struct CustomPhaseItemBuffers {
    vertices: RawBufferVec<Vertex>,
    indices: RawBufferVec<u32>,
}

#[derive(Clone, Copy, Pod, Zeroable)]
#[repr(C)]
struct Vertex {
    position: Vec3,
    pad0: u32,
    color: Vec3,
    pad1: u32,
}

type DrawCustomPhaseItemCommands = (SetCustomPhaseItemPipeline, DrawCustomPhaseItem);

static VERTICES: [Vertex; 3] = [
    Vertex::new(vec3(-0.866, -0.5, 0.0), vec3(1.0, 0.0, 0.0)),
    Vertex::new(vec3(0.866, -0.5, 0.0), vec3(0.0, 1.0, 0.0)),
    Vertex::new(vec3(0.0, 1.0, 0.0), vec3(0.0, 0.0, 1.0)),
];

fn main() {
    let mut app = App::new();
    app.add_plugins(DefaultPlugins)
        .add_plugins(ExtractComponentPlugin::<CustomRenderedEntity>::default())
        .add_systems(Startup, setup);

    app.get_sub_app_mut(RenderApp)
        .unwrap()
        .init_resource::<CustomPhasePipeline>()
        .init_resource::<CustomPhaseItemBuffers>()
        .add_render_command::<Opaque3d, DrawCustomPhaseItemCommands>()
        .add_systems(
            Render,
            prepare_custom_phase_pipelines.in_set(RenderSet::Prepare),
        )
        .add_systems(Render, queue_custom_phase_item.in_set(RenderSet::Queue));

    app.run();
}

fn setup(mut commands: Commands) {
    commands.spawn(CustomRenderedEntity);

    commands.spawn(Camera3dBundle {
        transform: Transform::from_xyz(0.0, 0.0, 1.0).looking_at(Vec3::ZERO, Vec3::Y),
        ..default()
    });
}

fn prepare_custom_phase_pipelines(
    mut commands: Commands,
    pipeline_cache: Res<PipelineCache>,
    mut specialized_render_pipelines: ResMut<SpecializedRenderPipelines<CustomPhasePipeline>>,
    custom_phase_pipeline: Res<CustomPhasePipeline>,
    custom_rendered_entities: Query<Entity, With<CustomRenderedEntity>>,
) {
    for entity in custom_rendered_entities.iter() {
        let pipeline_id =
            specialized_render_pipelines.specialize(&pipeline_cache, &custom_phase_pipeline, ());

        commands
            .entity(entity)
            .insert(CustomPhaseRenderPipeline { pipeline_id });
    }
}

fn queue_custom_phase_item(
    mut opaque_render_phases: ResMut<ViewBinnedRenderPhases<Opaque3d>>,
    opaque_draw_functions: Res<DrawFunctions<Opaque3d>>,
    views: Query<Entity, With<ExtractedView>>,
    custom_rendered_entities: Query<
        (Entity, &CustomPhaseRenderPipeline),
        With<CustomRenderedEntity>,
    >,
) {
    let draw_custom_phase_item = opaque_draw_functions
        .read()
        .id::<DrawCustomPhaseItemCommands>();

    for view_entity in views.iter() {
        let Some(opaque_phase) = opaque_render_phases.get_mut(&view_entity) else {
            continue;
        };

        for (entity, custom_phase_render_pipeline) in custom_rendered_entities.iter() {
            opaque_phase.add(
                Opaque3dBinKey {
                    draw_function: draw_custom_phase_item,
                    pipeline: custom_phase_render_pipeline.pipeline_id,
                    asset_id: AssetId::<Mesh>::invalid().untyped(),
                    material_bind_group_id: None,
                    lightmap_image: None,
                },
                entity,
                false,
            );
        }
    }
}

impl SpecializedRenderPipeline for CustomPhasePipeline {
    type Key = ();

    fn specialize(&self, _: Self::Key) -> RenderPipelineDescriptor {
        RenderPipelineDescriptor {
            label: Some("custom render pipeline".into()),
            layout: vec![],
            push_constant_ranges: vec![],
            vertex: VertexState {
                shader: self.shader.clone(),
                shader_defs: vec![],
                entry_point: "vertex".into(),
                buffers: vec![],
            },
            fragment: Some(FragmentState {
                shader: self.shader.clone(),
                shader_defs: vec![],
                entry_point: "fragment".into(),
                targets: vec![Some(ColorTargetState {
                    format: TextureFormat::bevy_default(),
                    blend: None,
                    write_mask: ColorWrites::ALL,
                })],
            }),
            primitive: PrimitiveState::default(),
            depth_stencil: None,
            multisample: MultisampleState::default(),
        }
    }
}

impl<P> RenderCommand<P> for SetCustomPhaseItemPipeline
where
    P: CachedRenderPipelinePhaseItem,
{
    type Param = SRes<PipelineCache>;

    type ViewQuery = ();

    type ItemQuery = Read<CustomPhaseRenderPipeline>;

    fn render<'w>(
        _: &P,
        _: ROQueryItem<'w, Self::ViewQuery>,
        maybe_custom_phase_render_pipeline_id: Option<ROQueryItem<'w, Self::ItemQuery>>,
        pipeline_cache: SystemParamItem<'w, '_, Self::Param>,
        pass: &mut TrackedRenderPass<'w>,
    ) -> RenderCommandResult {
        // Borrow check workaround.
        let pipeline_cache = pipeline_cache.into_inner();

        let Some(custom_phase_render_pipeline_id) = maybe_custom_phase_render_pipeline_id else {
            return RenderCommandResult::Failure;
        };
        let Some(custom_phase_render_pipeline) =
            pipeline_cache.get_render_pipeline(custom_phase_render_pipeline_id.pipeline_id)
        else {
            return RenderCommandResult::Failure;
        };

        pass.set_render_pipeline(custom_phase_render_pipeline);
        RenderCommandResult::Success
    }
}

impl<P> RenderCommand<P> for DrawCustomPhaseItem
where
    P: PhaseItem,
{
    type Param = SRes<CustomPhaseItemBuffers>;

    type ViewQuery = ();

    type ItemQuery = ();

    fn render<'w>(
        _: &P,
        _: ROQueryItem<'w, Self::ViewQuery>,
        _: Option<ROQueryItem<'w, Self::ItemQuery>>,
        custom_phase_item_buffers: SystemParamItem<'w, '_, Self::Param>,
        pass: &mut TrackedRenderPass<'w>,
    ) -> RenderCommandResult {
        // Borrow check workaround.
        let custom_phase_item_buffers = custom_phase_item_buffers.into_inner();

        pass.set_vertex_buffer(
            0,
            custom_phase_item_buffers
                .vertices
                .buffer()
                .unwrap()
                .slice(..),
        );

        pass.set_index_buffer(
            custom_phase_item_buffers
                .indices
                .buffer()
                .unwrap()
                .slice(..),
            0,
            IndexFormat::Uint32,
        );

        pass.draw_indexed(0..3, 0, 0..1);

        RenderCommandResult::Success
    }
}

impl FromWorld for CustomPhaseItemBuffers {
    fn from_world(world: &mut World) -> Self {
        let render_device = world.resource::<RenderDevice>();
        let render_queue = world.resource::<RenderQueue>();

        let mut vbo = RawBufferVec::new(BufferUsages::VERTEX);
        let mut ibo = RawBufferVec::new(BufferUsages::INDEX);

        for vertex in &VERTICES {
            vbo.push(*vertex);
        }
        for index in 0..3 {
            ibo.push(index);
        }

        vbo.write_buffer(render_device, render_queue);
        ibo.write_buffer(render_device, render_queue);

        CustomPhaseItemBuffers {
            vertices: vbo,
            indices: ibo,
        }
    }
}

impl Vertex {
    const fn new(position: Vec3, color: Vec3) -> Vertex {
        Vertex {
            position,
            color,
            pad0: 0,
            pad1: 0,
        }
    }
}

impl FromWorld for CustomPhasePipeline {
    fn from_world(world: &mut World) -> Self {
        let asset_server = world.resource::<AssetServer>();

        CustomPhasePipeline {
            shader: asset_server.load("shaders/custom_phase_item.wgsl"),
        }
    }
}
