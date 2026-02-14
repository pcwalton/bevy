use core::{any::TypeId, mem};

use bevy_ecs::{component::Component, entity::Entity, prelude::ReflectComponent, system::Query};
use bevy_reflect::{prelude::ReflectDefault, Reflect};
use bevy_utils::TypeIdMap;

use crate::{
    sync_world::{MainEntity, RenderEntity},
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

#[derive(Component, Clone, Debug, Default, Reflect)]
#[reflect(Component, Debug, Default, Clone)]
pub struct RenderVisibleMeshEntities {
    #[reflect(ignore, clone)]
    pub entities: Vec<(Entity, MainEntity)>,
    pub added_entities: Vec<(Entity, MainEntity)>,
    pub removed_entities: Vec<(Entity, MainEntity)>,
}

impl RenderVisibleEntities {
    pub fn get<QF>(&self) -> Option<&RenderVisibleMeshEntities>
    where
        QF: 'static,
    {
        self.entities.get(&TypeId::of::<QF>())
    }

    pub fn len<QF>(&self) -> usize
    where
        QF: 'static,
    {
        match self.get::<QF>() {
            Some(entities) => entities.entities.len(),
            None => 0,
        }
    }

    pub fn is_empty<QF>(&self) -> bool
    where
        QF: 'static,
    {
        match self.get::<QF>() {
            Some(entities) => entities.entities.is_empty(),
            None => true,
        }
    }
}

impl RenderVisibleMeshEntities {
    pub fn update_from(
        &mut self,
        mapper: &Extract<Query<RenderEntity>>,
        visible_mesh_entities: &[Entity],
    ) {
        let old_entities = mem::take(&mut self.entities);
        self.added_entities.clear();
        self.removed_entities.clear();

        // March over the old and new visible entity lists in lockstep, diffing
        // as we go to determine the added and removed entities. The lists must
        // be sorted.
        let mut old_entity_iter = old_entities.iter().peekable();
        for &visible_main_entity in visible_mesh_entities {
            let visible_main_entity = MainEntity::from(visible_main_entity);

            // Mark entities as removed until we see the one we're looking at.
            while old_entity_iter
                .peek()
                .is_some_and(|(_, main_entity)| *main_entity < visible_main_entity)
            {
                self.removed_entities.push(*old_entity_iter.next().unwrap());
            }

            // Add the visible entity to the list.
            let render_entity = mapper
                .get(*visible_main_entity)
                .unwrap_or(Entity::PLACEHOLDER);
            self.entities.push((render_entity, visible_main_entity));

            if old_entity_iter
                .peek()
                .is_some_and(|&&(_, main_entity)| main_entity == visible_main_entity)
            {
                old_entity_iter.next();
            } else {
                self.added_entities
                    .push((render_entity, visible_main_entity));
            }
        }

        // Any entities we didn't see yet are removed, so drain them.
        self.removed_entities.extend(old_entity_iter.copied());
    }
}
