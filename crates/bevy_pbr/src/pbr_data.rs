//! The data relevant to PBR rendering that's cached on entities in the main
//! world.

use bevy_asset::{AssetId, Assets, Handle, UntypedAssetId};
use bevy_ecs::{
    component::Component,
    entity::Entity,
    query::{Has, QueryItem},
    system::{lifetimeless::Read, Commands, Query, Res},
};
use bevy_math::Vec3;
use bevy_render::{
    alpha::AlphaMode,
    extract_instances::ExtractInstance,
    mesh::{Mesh, MeshVertexBufferLayout},
    render_resource::AsBindGroup,
};
use bevy_transform::components::GlobalTransform;

use crate::{
    alpha_mode_pipeline_key, DefaultOpaqueRendererMethod, Lightmap, Material, MeshPipelineKey,
    OpaqueRendererMethod,
};

/// The data relevant to PBR rendering that's cached on entities in the main
/// world.
///
/// Bevy internally manages this. Usually an app will have no need to mess with
/// the data here.
#[derive(Clone, Component)]
pub struct PbrData<M>
where
    M: Material,
    <M as AsBindGroup>::Data: Clone,
{
    pub(crate) mesh_pipeline_key: MeshPipelineKey,
    pub(crate) opaque_renderer_method: OpaqueRendererMethod,
    pub(crate) bind_group_data: <M as AsBindGroup>::Data,
    pub(crate) mesh_asset_id: AssetId<Mesh>,
    pub(crate) material_asset_id: UntypedAssetId,
    // TODO: Just look at the appropriate bits in `MeshPipelineKey`?
    pub(crate) alpha_mode: AlphaMode,
    pub(crate) mesh_layout: MeshVertexBufferLayout,
    pub(crate) depth_bias: f32,
}

// FIXME: This is really ugly.
pub struct RenderPbrData<M>
where
    M: Material,
    <M as AsBindGroup>::Data: Clone,
{
    pub(crate) pbr_data: PbrData<M>,
    pub(crate) translation: Vec3,
}

impl<M> ExtractInstance for RenderPbrData<M>
where
    M: Material,
    <M as AsBindGroup>::Data: Clone,
{
    type QueryData = (Read<PbrData<M>>, Read<GlobalTransform>);

    type QueryFilter = ();

    fn extract((pbr_data, transform): QueryItem<'_, Self::QueryData>) -> Option<Self> {
        Some(RenderPbrData {
            pbr_data: pbr_data.clone(),
            translation: transform.translation(),
        })
    }
}

pub fn update_pbr_data<M>(
    mut commands: Commands,
    mut query: Query<(Entity, &Handle<M>, &Handle<Mesh>, Has<Lightmap>)>,
    materials: Res<Assets<M>>,
    meshes: Res<Assets<Mesh>>,
    default_opaque_renderer_method: Res<DefaultOpaqueRendererMethod>,
) where
    M: Material,
    <M as AsBindGroup>::Data: Clone,
{
    for (entity, material_handle, mesh_handle, has_lightmap) in query.iter_mut() {
        let Some(material) = materials.get(material_handle) else {
            continue;
        };
        let Some(mesh) = meshes.get(mesh_handle) else {
            continue;
        };

        let mut opaque_renderer_method = material.opaque_render_method();
        if opaque_renderer_method == OpaqueRendererMethod::Auto {
            opaque_renderer_method = default_opaque_renderer_method.0;
        }

        let alpha_mode = material.alpha_mode();
        let mesh_layout = mesh.get_mesh_vertex_buffer_layout();
        let depth_bias = material.depth_bias();
        let mesh_asset_id = mesh_handle.id();
        let material_asset_id = material_handle.id().untyped();
        let bind_group_data: <M as AsBindGroup>::Data = material.data();

        let mut mesh_pipeline_key =
            MeshPipelineKey::from_primitive_topology(mesh.primitive_topology());

        if mesh.has_morph_targets() {
            mesh_pipeline_key |= MeshPipelineKey::MORPH_TARGETS;
        }

        if material.reads_view_transmission_texture() {
            mesh_pipeline_key |= MeshPipelineKey::READS_VIEW_TRANSMISSION_TEXTURE;
        }

        mesh_pipeline_key |= alpha_mode_pipeline_key(alpha_mode);

        if has_lightmap {
            mesh_pipeline_key |= MeshPipelineKey::LIGHTMAPPED;
        }

        commands.entity(entity).insert(PbrData::<M> {
            mesh_pipeline_key,
            opaque_renderer_method,
            bind_group_data,
            mesh_asset_id,
            material_asset_id,
            alpha_mode,
            mesh_layout,
            depth_bias,
        });
    }
}
