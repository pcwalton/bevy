//! GPU two-phase occlusion culling.

use bevy_ecs::{component::Component, reflect::ReflectComponent};
use bevy_reflect::Reflect;
use bevy_render_macros::ExtractComponent;

#[derive(Clone, Copy, Component, Reflect, ExtractComponent)]
#[reflect(Component)]
pub struct OcclusionCulling;
