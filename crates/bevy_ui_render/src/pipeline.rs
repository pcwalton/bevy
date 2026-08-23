use alloc::borrow::Cow;
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
use bevy_shader::{Shader, ShaderDefVal};
use bevy_utils::default;
use bitflags::bitflags;

use crate::UiMeta;

#[derive(Resource)]
pub struct UiPipeline {
    pub view_layout: BindGroupLayoutDescriptor,
    pub image_bindless_layout: BindGroupLayoutDescriptor,
    pub image_non_bindless_layout: BindGroupLayoutDescriptor,
    pub instances_layout: BindGroupLayoutDescriptor,
    pub shader: Handle<Shader>,
}

impl UiPipeline {
    pub fn image_bind_group_layout(&self, bindless: bool) -> &BindGroupLayoutDescriptor {
        if bindless {
            &self.image_bindless_layout
        } else {
            &self.image_non_bindless_layout
        }
    }
}

pub(crate) static IMAGE_BINDLESS_DESCRIPTOR: BindlessDescriptor = BindlessDescriptor {
    resources: Cow::Borrowed(&[
        BindlessResourceType::Texture2d,
        BindlessResourceType::SamplerFiltering,
    ]),
    buffers: Cow::Borrowed(&[]),
    index_tables: Cow::Borrowed(&[BindlessIndexTableDescriptor {
        indices: BindlessIndex(0)..BindlessIndex(2),
        binding_number: BindingNumber(0),
    }]),
};

pub fn init_ui_pipeline(
    mut commands: Commands,
    asset_server: Res<AssetServer>,
    ui_meta: Res<UiMeta>,
) {
    let view_layout = BindGroupLayoutDescriptor::new(
        "ui_view_layout",
        &BindGroupLayoutEntries::single(
            ShaderStages::VERTEX_FRAGMENT,
            uniform_buffer::<ViewUniform>(true),
        ),
    );

    let image_non_bindless_layout = BindGroupLayoutDescriptor::new(
        "ui_image_non_bindless_layout",
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
        image_bindless_layout: ui_meta.image_bindless_bind_group_layout_descriptor.clone(),
        image_non_bindless_layout,
        instances_layout,
        shader: load_embedded_asset!(asset_server.as_ref(), "ui.wesl"),
    });
}

#[derive(Clone, Copy, Hash, PartialEq, Eq)]
pub struct UiPipelineKey {
    pub target_format: TextureFormat,
    pub flags: UiPipelineKeyFlags,
}

bitflags! {
    #[derive(Clone, Copy, PartialEq, Eq, Hash)]
    pub struct UiPipelineKeyFlags: u8 {
        const ANTI_ALIAS = 1 << 0;
        const RETAINED = 1 << 1;
        const BINDLESS = 1 << 2;
    }
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
        let instance_vertex_format = if key.flags.contains(UiPipelineKeyFlags::RETAINED) {
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
        if key.flags.contains(UiPipelineKeyFlags::ANTI_ALIAS) {
            shader_defs.push("ANTI_ALIAS".into());
        }
        if key.flags.contains(UiPipelineKeyFlags::RETAINED) {
            shader_defs.push("RETAINED_INSTANCES".into());
        }
        if key.flags.contains(UiPipelineKeyFlags::BINDLESS) {
            shader_defs.push("BINDLESS".into());
            shader_defs.push(ShaderDefVal::UInt("MATERIAL_BIND_GROUP".into(), 1));
        }

        let mut layout = vec![
            self.view_layout.clone(),
            if key.flags.contains(UiPipelineKeyFlags::BINDLESS) {
                self.image_bindless_layout.clone()
            } else {
                self.image_non_bindless_layout.clone()
            },
        ];
        if key.flags.contains(UiPipelineKeyFlags::RETAINED) {
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
