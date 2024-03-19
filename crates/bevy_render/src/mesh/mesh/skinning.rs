use bevy_asset::{Asset, Assets, Handle};
use bevy_ecs::{
    component::Component,
    entity::{Entity, EntityMapper, MapEntities},
    prelude::ReflectComponent,
    query::Without,
    reflect::ReflectMapEntities,
    system::{Query, Res},
};
use bevy_math::Mat4;
use bevy_reflect::{Reflect, TypePath};
use bevy_transform::components::GlobalTransform;
use std::ops::Deref;

use crate::prelude::ViewVisibility;

/// Maximum number of joints supported for skinned meshes.
pub const MAX_JOINTS: usize = 256;

#[derive(Component, Debug, Default, Clone, Reflect)]
#[reflect(Component, MapEntities)]
pub struct SkinnedMesh {
    pub inverse_bindposes: Handle<SkinnedMeshInverseBindposes>,
    pub joints: Vec<Entity>,
}

impl MapEntities for SkinnedMesh {
    fn map_entities<M: EntityMapper>(&mut self, entity_mapper: &mut M) {
        for joint in &mut self.joints {
            *joint = entity_mapper.map_entity(*joint);
        }
    }
}

#[derive(Asset, TypePath, Debug)]
pub struct SkinnedMeshInverseBindposes(Box<[Mat4]>);

impl From<Vec<Mat4>> for SkinnedMeshInverseBindposes {
    fn from(value: Vec<Mat4>) -> Self {
        Self(value.into_boxed_slice())
    }
}

impl Deref for SkinnedMeshInverseBindposes {
    type Target = [Mat4];
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

/// Stores the resolved joint matrices.
///
/// These are computed for all visible skinned meshes just before rendering
/// starts.
#[derive(Component, Debug, Default, Clone, Reflect)]
#[reflect(Component)]
pub struct ComputedPose {
    /// Global transforms, stored in the same order as `Mat4`.
    pub joints: Vec<Mat4>,
}

pub fn compute_poses(
    mut meshes: Query<(&SkinnedMesh, &ViewVisibility, &mut ComputedPose)>,
    joints: Query<&GlobalTransform, Without<ComputedPose>>,
    inverse_bindposes: Res<Assets<SkinnedMeshInverseBindposes>>,
) {
    meshes
        .par_iter_mut()
        .for_each(|(skinned_mesh, view_visibility, mut computed_pose)| {
            // Skip invisible meshes.
            if !view_visibility.get() {
                return;
            }

            // Skip meshes with no inverse bindposes.
            let Some(inverse_bindposes) = inverse_bindposes.get(&skinned_mesh.inverse_bindposes)
            else {
                return;
            };

            computed_pose.joints.clear();
            computed_pose.joints.extend(
                joints
                    .iter_many(&skinned_mesh.joints)
                    .zip(inverse_bindposes.iter())
                    .take(MAX_JOINTS)
                    .map(|(joint, bindpose)| joint.affine() * *bindpose),
            );

            // `iter_many`` will skip any failed fetches. This will cause it to
            // assign the wrong bones, so just bail by truncating.
            if computed_pose.joints.len() != inverse_bindposes.len().min(MAX_JOINTS) {
                computed_pose.joints.clear();
            }
        });
}
