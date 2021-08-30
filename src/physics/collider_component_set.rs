use super::{IntoEntity, IntoHandle};
use bevy::prelude::*;
use rapier::data::{ComponentSet, ComponentSetMut, ComponentSetOption, Index};
use rapier::prelude::*;

impl IntoHandle<ColliderHandle> for Entity {
    #[inline]
    fn handle(self) -> ColliderHandle {
        ColliderHandle::from_raw_parts(self.id(), self.generation())
    }
}

impl IntoEntity for ColliderHandle {
    #[inline]
    fn entity(self) -> Entity {
        self.0.entity()
    }
}

pub type QueryPipelineColliderComponentsQuery<'world, 'state, 'a> = Query<
    'world,
    'state,
    (
        Entity,
        &'a ColliderPosition,
        &'a ColliderShape,
        &'a ColliderFlags,
    ),
>;

pub struct QueryPipelineColliderComponentsSet<'world, 'state, 'a, 'c>(
    pub &'c QueryPipelineColliderComponentsQuery<'world, 'state, 'a>,
);

impl_component_set!(
    QueryPipelineColliderComponentsSet,
    ColliderPosition,
    |data| data.1
);
impl_component_set!(QueryPipelineColliderComponentsSet, ColliderShape, |data| {
    data.2
});
impl_component_set!(QueryPipelineColliderComponentsSet, ColliderFlags, |data| {
    data.3
});

pub struct ColliderComponentsSet<'world, 'state, 'a>(
    pub Query<'world, 'state, ColliderComponentsQueryPayload<'a>>,
);

pub type ColliderComponentsQueryPayload<'a> = (
    Entity,
        &'a mut ColliderChanges,
        &'a mut ColliderPosition,
        &'a mut ColliderBroadPhaseData,
        &'a mut ColliderShape,
        &'a mut ColliderType,
        &'a mut ColliderMaterial,
        &'a mut ColliderFlags,
        Option<&'a ColliderParent>,
);

pub type ColliderChangesQueryPayload<'a> = (
    Entity,
    &'a mut ColliderChanges,
    Or<(Changed<ColliderPosition>, Added<ColliderPosition>)>,
    Or<(Changed<ColliderFlags>, Added<ColliderFlags>)>,
    Or<(Changed<ColliderShape>, Added<ColliderShape>)>,
    Or<(Changed<ColliderType>, Added<ColliderType>)>,
    Option<Or<(Changed<ColliderParent>, Added<ColliderParent>)>>,
);

pub type ColliderChangesQueryFilter = (
    Or<(Changed<ColliderPosition>, Added<ColliderPosition>)>,
    Or<(Changed<ColliderFlags>, Added<ColliderFlags>)>,
    Or<(Changed<ColliderShape>, Added<ColliderShape>)>,
    Or<(Changed<ColliderType>, Added<ColliderType>)>,
    Or<(Changed<ColliderParent>, Added<ColliderParent>)>,
);

pub type ColliderComponentsQuerySet<'world, 'state, 'a> = QuerySet<
    'world,
    'state,
    (
        // Components query
        QueryState<ColliderComponentsQueryPayload<'a>>,
        // Changes query
        QueryState<ColliderChangesQueryPayload<'a>, ColliderChangesQueryFilter>,
    ),
>;

impl_component_set_mut!(ColliderComponentsSet, ColliderChanges, |data| &*data.1);
impl_component_set_mut!(ColliderComponentsSet, ColliderPosition, |data| &*data.2);
impl_component_set_mut!(ColliderComponentsSet, ColliderBroadPhaseData, |d| &*d.3);
impl_component_set_mut!(ColliderComponentsSet, ColliderShape, |data| &*data.4);
impl_component_set_mut!(ColliderComponentsSet, ColliderType, |data| &*data.5);
impl_component_set_mut!(ColliderComponentsSet, ColliderMaterial, |data| &*data.6);
impl_component_set_mut!(ColliderComponentsSet, ColliderFlags, |data| &*data.7);
impl_component_set_option!(ColliderComponentsSet, ColliderParent);

#[derive(Bundle)]
pub struct ColliderBundle {
    pub collider_type: ColliderType,
    pub shape: ColliderShape,
    pub position: ColliderPosition,
    pub material: ColliderMaterial,
    pub flags: ColliderFlags,
    pub mass_properties: ColliderMassProps,
    pub changes: ColliderChanges,
    pub bf_data: ColliderBroadPhaseData,
}

impl Default for ColliderBundle {
    fn default() -> Self {
        Self {
            collider_type: ColliderType::Solid,
            shape: ColliderShape::ball(0.5),
            position: ColliderPosition::default(),
            material: ColliderMaterial::default(),
            flags: ColliderFlags::default(),
            mass_properties: ColliderMassProps::default(),
            changes: ColliderChanges::default(),
            bf_data: ColliderBroadPhaseData::default(),
        }
    }
}
