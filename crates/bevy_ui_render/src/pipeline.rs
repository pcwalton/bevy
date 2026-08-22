use bevy_asset::{load_embedded_asset, AssetServer, Handle};
use bevy_ecs::prelude::*;
use bevy_mesh::VertexBufferLayout;
use bevy_render::{
    render_resource::{
        binding_types::{sampler, storage_buffer_read_only_sized, texture_2d, uniform_buffer},
        *,
    },
    view::ViewUniform,
};
use bevy_shader::Shader;
use bevy_utils::default;

#[derive(Resource)]
pub struct UiPipeline {
    pub view_layout: BindGroupLayoutDescriptor,
    pub image_layout: BindGroupLayoutDescriptor,
    pub instances_layout: BindGroupLayoutDescriptor,
    pub shader: Handle<Shader>,
}

pub fn init_ui_pipeline(mut commands: Commands, asset_server: Res<AssetServer>) {
    let view_layout = BindGroupLayoutDescriptor::new(
        "ui_view_layout",
        &BindGroupLayoutEntries::single(
            ShaderStages::VERTEX_FRAGMENT,
            uniform_buffer::<ViewUniform>(true),
        ),
    );

    let image_layout = BindGroupLayoutDescriptor::new(
        "ui_image_layout",
        &BindGroupLayoutEntries::sequential(
            ShaderStages::FRAGMENT,
            (
                texture_2d(TextureSampleType::Float { filterable: true }),
                sampler(SamplerBindingType::Filtering),
            ),
        ),
    );

    let instances_layout = BindGroupLayoutDescriptor::new(
        "ui_instances_layout",
        &BindGroupLayoutEntries::single(
            ShaderStages::VERTEX,
            storage_buffer_read_only_sized(false, None),
        ),
    );

    commands.insert_resource(UiPipeline {
        view_layout,
        image_layout,
        instances_layout,
        shader: load_embedded_asset!(asset_server.as_ref(), "ui.wesl"),
    });
}

#[derive(Clone, Copy, Hash, PartialEq, Eq)]
pub struct UiPipelineKey {
    pub target_format: TextureFormat,
    pub anti_alias: bool,
    pub retained: bool,
}

impl SpecializedRenderPipeline for UiPipeline {
    type Key = UiPipelineKey;

    fn specialize(&self, key: Self::Key) -> RenderPipelineDescriptor {
        let vertex_layout = VertexBufferLayout::from_vertex_formats(
            VertexStepMode::Vertex,
            vec![
                // position
                VertexFormat::Float32x2,
            ],
        );
        let instance_vertex_format = if key.retained {
            vec![
                // instance index
                VertexFormat::Uint32,
            ]
        } else {
            vec![
                // world_from_local
                VertexFormat::Float32x4,
                // color
                VertexFormat::Float32x4,
                // border thickness
                VertexFormat::Float32x4,
                // border radius x values (top left, top right, bottom right, bottom left)
                VertexFormat::Float32x4,
                // border radius y values (top left, top right, bottom right, bottom left)
                VertexFormat::Float32x4,
                // uv_scale
                VertexFormat::Float32x2,
                // uv_offset
                VertexFormat::Float32x2,
                // translation
                VertexFormat::Float32x2,
                // size
                VertexFormat::Float32x2,
                // flags
                VertexFormat::Uint32,
                // pad
                VertexFormat::Uint32x3,
            ]
        };
        let instance_layout = VertexBufferLayout::from_vertex_formats(
            VertexStepMode::Instance,
            instance_vertex_format,
        )
        .offset_locations_by(1);

        let mut shader_defs = vec![];
        if key.anti_alias {
            shader_defs.push("ANTI_ALIAS".into());
        }
        if key.retained {
            shader_defs.push("RETAINED_INSTANCES".into());
        }

        let mut layout = vec![self.view_layout.clone(), self.image_layout.clone()];
        if key.retained {
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
            label: Some("ui_pipeline".into()),
            ..default()
        }
    }
}
