#[allow(clippy::module_inception)]
mod mesh;
pub mod morph;
pub mod primitives;

use bevy_utils::HashSet;
pub use mesh::*;
pub use primitives::*;
use std::{
    borrow::Borrow,
    hash::{Hash, Hasher},
    sync::Arc,
};

use crate::{prelude::Image, render_asset::RenderAssetPlugin, RenderApp};
use bevy_app::{App, Plugin};
use bevy_asset::{AssetApp, Handle};
use bevy_ecs::{entity::Entity, system::Resource};

/// Adds the [`Mesh`] as an asset and makes sure that they are extracted and prepared for the GPU.
pub struct MeshPlugin;

impl Plugin for MeshPlugin {
    fn build(&self, app: &mut App) {
        app.init_asset::<Mesh>()
            .init_asset::<skinning::SkinnedMeshInverseBindposes>()
            .register_asset_reflect::<Mesh>()
            .register_type::<Option<Handle<Image>>>()
            .register_type::<Option<Vec<String>>>()
            .register_type::<Option<Indices>>()
            .register_type::<Indices>()
            .register_type::<skinning::SkinnedMesh>()
            .register_type::<Vec<Entity>>()
            // 'Mesh' must be prepared after 'Image' as meshes rely on the morph target image being ready
            .add_plugins(RenderAssetPlugin::<Mesh, Image>::default());

        let Ok(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };

        render_app.init_resource::<MeshVertexBufferLayouts>();
    }
}

/// Describes the layout of the mesh vertices in GPU memory.
///
/// At most one copy of a mesh vertex buffer layout ever exists in GPU memory at
/// once. Therefore, comparing these for equality requires only a single pointer
/// comparison.
#[derive(Clone, Debug)]
pub struct MeshVertexBufferLayoutRef(pub Arc<MeshVertexBufferLayout>);

/// Stores the single copy of each mesh vertex buffer layout.
#[derive(Clone, Default, Resource)]
pub struct MeshVertexBufferLayouts(HashSet<MeshVertexBufferLayoutRef>);

impl MeshVertexBufferLayouts {
    /// Inserts a new mesh vertex buffer layout in the store and returns a
    /// reference to it, reusing the existing reference if this mesh vertex
    /// buffer layout was already in the store.
    pub(crate) fn insert(&mut self, layout: MeshVertexBufferLayout) -> MeshVertexBufferLayoutRef {
        // Because we're using `get_or_insert_with` on the layout itself, not
        // the `MeshVertexBufferLayoutRef`, this compares the mesh vertex buffer
        // layout structurally, not by pointer.
        self.0
            .get_or_insert_with(&layout, |layout| {
                MeshVertexBufferLayoutRef(Arc::new(layout.clone()))
            })
            .clone()
    }
}

impl Borrow<MeshVertexBufferLayout> for MeshVertexBufferLayoutRef {
    fn borrow(&self) -> &MeshVertexBufferLayout {
        &self.0
    }
}

impl PartialEq for MeshVertexBufferLayoutRef {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }
}

impl Eq for MeshVertexBufferLayoutRef {}

impl Hash for MeshVertexBufferLayoutRef {
    fn hash<H: Hasher>(&self, state: &mut H) {
        (&*self.0 as &MeshVertexBufferLayout as *const MeshVertexBufferLayout as usize).hash(state);
    }
}
