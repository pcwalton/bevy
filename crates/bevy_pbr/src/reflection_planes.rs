use bevy_ecs::{prelude::Component, reflect::ReflectComponent};
use bevy_reflect::Reflect;

#[derive(Component, Reflect, Debug, Default, Clone, Copy)]
#[reflect(Component)]
pub struct ReflectionPlane {
    pub thickness: f32,
}
