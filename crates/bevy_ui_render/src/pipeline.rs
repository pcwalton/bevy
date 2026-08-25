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
use bitflags::bitflags;

#[derive(Resource)]
pub struct UiPipeline {
    pub view_layout: BindGroupLayoutDescriptor,
    /// The bind group layout for the buffer that stores data that's unique to
    /// each instance.
    ///
    /// This is only present in retained mode instance rendering (see
    /// [`crate::UiInstances`]). In immediate mode, this is unused.
    pub instances_layout: BindGroupLayoutDescriptor,
    pub image_layout: BindGroupLayoutDescriptor,
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

    let instances_layout = BindGroupLayoutDescriptor::new(
        "ui_instances_layout",
        &BindGroupLayoutEntries::single(
            ShaderStages::VERTEX,
            storage_buffer_read_only_sized(false, None),
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

    commands.insert_resource(UiPipeline {
        view_layout,
        instances_layout,
        image_layout,
        shader: load_embedded_asset!(asset_server.as_ref(), "ui.wesl"),
    });
}

#[derive(Clone, Copy, Hash, PartialEq, Eq)]
pub struct UiPipelineKey {
    pub target_format: TextureFormat,
    pub flags: UiPipelineKeyFlags,
}

bitflags! {
    #[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
    pub struct UiPipelineKeyFlags: u8 {
        const ANTI_ALIAS = 1 << 0;
        /// Set if we're using retained instances; unset if we're using
        /// immediate instances.
        ///
        /// See [`crate::UiInstances`] for more information.
        const RETAINED_INSTANCES = 1 << 1;
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

        // Specify the layout of a single quad (`UiNodeInstanceData`).
        let mut instance_layout = VertexBufferLayout::from_vertex_formats(
            VertexStepMode::Instance,
            if key.flags.contains(UiPipelineKeyFlags::RETAINED_INSTANCES) {
                vec![
                    // instance index
                    VertexFormat::Uint32,
                ]
            } else {
                vec![
                    // transform
                    VertexFormat::Float32x4,
                    // color
                    VertexFormat::Float32x4,
                    // border
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
                ]
            },
        )
        .offset_locations_by(1);
        // Account for padding if needed.
        if !key.flags.contains(UiPipelineKeyFlags::RETAINED_INSTANCES) {
            instance_layout.array_stride += 12;
        }

        let mut shader_defs = vec![];
        if key.flags.contains(UiPipelineKeyFlags::ANTI_ALIAS) {
            shader_defs.push("ANTI_ALIAS".into());
        }
        if key.flags.contains(UiPipelineKeyFlags::RETAINED_INSTANCES) {
            shader_defs.push("RETAINED_INSTANCES".into());
        }

        let mut layout = vec![self.view_layout.clone(), self.image_layout.clone()];
        if key.flags.contains(UiPipelineKeyFlags::RETAINED_INSTANCES) {
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
