use core::{any::TypeId, mem};

use bevy_ecs::{
    component::Component,
    entity::Entity,
    lifecycle::RemovedComponents,
    prelude::ReflectComponent,
    query::Changed,
    system::{Query, SystemParam},
};
use bevy_reflect::{prelude::ReflectDefault, Reflect};
use bevy_utils::TypeIdMap;

use crate::{
    sync_world::{MainEntity, MainEntityHashMap, RenderEntity},
    Extract,
};

mod range;
use bevy_camera::visibility::*;
pub use range::*;

/// Collection of entities visible from the current view.
///
/// This component is extracted from [`VisibleEntities`].
#[derive(Clone, Component, Default, Debug, Reflect)]
#[reflect(Component, Default, Debug, Clone)]
pub struct RenderVisibleEntities {
    #[reflect(ignore, clone)]
    pub entities: TypeIdMap<RenderVisibleMeshEntities>,
}

/// Stores a list of all entities that are visible from this view, as well as
/// the change lists.
///
/// Note that all lists in this component are guaranteed to be sorted. Thus you
/// can test for the presence of an entity in these lists via binary search.
///
/// Note also that, for 3D meshes, the render-world [`Entity`] values will
/// always be [`Entity::PLACEHOLDER`]. The render-world entities are kept for
/// legacy passes that still need to process visibility of render-world
/// entities.
#[derive(Component, Clone, Debug, Default, Reflect)]
#[reflect(Component, Debug, Default, Clone)]
pub struct RenderVisibleMeshEntities {
    /// A sorted list of all entities that are visible from this view.
    #[reflect(ignore, clone)]
    pub entities_cpu_culling: Vec<(Entity, MainEntity)>,
    pub entities_gpu_culling: MainEntityHashMap<Entity>,
    /// A sorted list of all entities that were invisible last frame (including
    /// ones that didn't exist at all last frame) and became visible this frame.
    pub added_entities: Vec<(Entity, MainEntity)>,
    /// A sorted list of all entities that were visible last frame and became
    /// invisible this frame, including those that were despawned this frame.
    pub removed_entities: Vec<(Entity, MainEntity)>,
}

impl RenderVisibleEntities {
    pub fn get<QF>(&self) -> Option<&RenderVisibleMeshEntities>
    where
        QF: 'static,
    {
        self.entities.get(&TypeId::of::<QF>())
    }
}

impl RenderVisibleMeshEntities {
    /// Processes a list of visible entities for a new frame, computing the set
    /// of newly-added and newly-removed entities as it goes.
    pub fn update_from(
        &mut self,
        visibility_extraction_system_param: &PreparedVisibilityExtractionSystemParam,
        visible_mesh_entities: &[Entity],
    ) {
        let PreparedVisibilityExtractionSystemParam {
            mapper,
            no_cpu_culling_added_entities,
            no_cpu_culling_removed_entities,
        } = visibility_extraction_system_param;

        let old_entities_cpu_culling = mem::take(&mut self.entities_cpu_culling);
        self.added_entities.clear();
        self.removed_entities.clear();

        // March over the old and new visible entity lists in lockstep, diffing
        // as we go to determine the added and removed entities. The lists must
        // be sorted.
        let mut old_entity_cpu_culling_iter = old_entities_cpu_culling.iter().peekable();
        for &visible_main_entity in visible_mesh_entities {
            let visible_main_entity = MainEntity::from(visible_main_entity);

            // Mark entities as removed until we see the one we're looking at.
            while old_entity_cpu_culling_iter
                .peek()
                .is_some_and(|(_, main_entity)| *main_entity < visible_main_entity)
            {
                self.removed_entities
                    .push(*old_entity_cpu_culling_iter.next().unwrap());
            }

            // Add the visible entity to the list.
            let render_entity = mapper
                .get(*visible_main_entity)
                .cloned()
                .unwrap_or(RenderEntity::from(Entity::PLACEHOLDER));
            self.entities_cpu_culling
                .push((*render_entity, visible_main_entity));

            // If the next entity in the old list isn't equal to the entity we
            // just marked visible, then our entity is newly visible this frame.
            if old_entity_cpu_culling_iter
                .peek()
                .is_some_and(|&&(_, main_entity)| main_entity == visible_main_entity)
            {
                old_entity_cpu_culling_iter.next();
            } else {
                self.added_entities
                    .push((*render_entity, visible_main_entity));
            }
        }

        // Any entities we didn't see yet are removed, so drain them.
        self.removed_entities
            .extend(old_entity_cpu_culling_iter.copied());

        // Now work on the GPU entities.
        // FIXME: check visibility class
        for (visible_main_entity, visibility_class) in no_cpu_culling_added_entities.iter() {
            let visible_main_entity = MainEntity::from(visible_main_entity);
            let render_entity = mapper
                .get(*visible_main_entity)
                .cloned()
                .unwrap_or(RenderEntity::from(Entity::PLACEHOLDER));
            self.added_entities
                .push((*render_entity, visible_main_entity));
            self.entities_gpu_culling
                .insert(visible_main_entity, *render_entity);
        }
        self.added_entities
            .sort_unstable_by_key(|(_, main_entity)| *main_entity);

        for removed_main_entity in no_cpu_culling_removed_entities {
            let removed_main_entity = MainEntity::from(*removed_main_entity);
            // Standard "added and removed same frame" check
            if self
                .entities_cpu_culling
                .binary_search_by_key(&removed_main_entity, |(_, main_entity)| *main_entity)
                .is_err()
                && self
                    .added_entities
                    .binary_search_by_key(&removed_main_entity, |(_, main_entity)| *main_entity)
                    .is_err()
            {
                self.removed_entities
                    .push((Entity::PLACEHOLDER, removed_main_entity));
                self.entities_gpu_culling.remove(&removed_main_entity);
            }
        }
    }

    pub fn entity_pair_is_visible(&self, entity: Entity, main_entity: MainEntity) -> bool {
        self.entities_cpu_culling
            .binary_search(&(entity, main_entity))
            .is_ok()
            || self
                .entities_gpu_culling
                .get(&main_entity)
                .is_some_and(|that_entity| *that_entity == entity)
    }

    pub fn iter_visible<'a>(&'a self) -> impl Iterator<Item = (&'a Entity, &'a MainEntity)> {
        self.entities_cpu_culling
            .iter()
            .map(|(entity, main_entity)| (entity, main_entity))
            .chain(
                self.entities_gpu_culling
                    .iter()
                    .map(|(main_entity, entity)| (entity, main_entity)),
            )
    }
}

#[derive(SystemParam)]
pub struct VisibilityExtractionSystemParam<'w, 's> {
    pub mapper: Extract<'w, 's, Query<'static, 'static, &'static RenderEntity>>,
    pub no_cpu_culling_added_entities: Extract<
        'w,
        's,
        Query<'static, 'static, (Entity, Option<&'static VisibilityClass>), Changed<NoCpuCulling>>,
    >,
    // FIXME: should be `VisibilityClass` too
    pub no_cpu_culling_removed_entities:
        Extract<'w, 's, RemovedComponents<'static, 'static, NoCpuCulling>>,
}

pub struct PreparedVisibilityExtractionSystemParam<'w, 's> {
    pub mapper: Extract<'w, 's, Query<'static, 'static, &'static RenderEntity>>,
    pub no_cpu_culling_added_entities: Extract<
        'w,
        's,
        Query<'static, 'static, (Entity, Option<&'static VisibilityClass>), Changed<NoCpuCulling>>,
    >,
    pub no_cpu_culling_removed_entities: Vec<Entity>,
}

impl<'w, 's> VisibilityExtractionSystemParam<'w, 's> {
    pub fn prepare(mut self) -> PreparedVisibilityExtractionSystemParam<'w, 's> {
        let no_cpu_culling_removed_entities = self.no_cpu_culling_removed_entities.read().collect();
        PreparedVisibilityExtractionSystemParam {
            mapper: self.mapper,
            no_cpu_culling_added_entities: self.no_cpu_culling_added_entities,
            no_cpu_culling_removed_entities,
        }
    }
}
