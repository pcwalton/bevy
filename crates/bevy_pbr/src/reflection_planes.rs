use bevy_ecs::{prelude::Component, reflect::ReflectComponent};
use bevy_reflect::Reflect;

#[derive(Component, Reflect, Debug, Default)]
#[reflect(Component)]
pub struct ReflectionPlane;
