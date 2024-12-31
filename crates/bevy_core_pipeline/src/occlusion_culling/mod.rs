//! GPU occlusion culling.

use bevy_app::{App, Plugin};
use bevy_ecs::{component::Component, prelude::ReflectComponent};
use bevy_reflect::{prelude::ReflectDefault, Reflect};
use bevy_render::{sync_component::SyncComponentPlugin, RenderApp};

pub struct OcclusionCullingPlugin;

impl Plugin for OcclusionCullingPlugin {
    fn build(&self, app: &mut App) {
        app.register_type::<OcclusionCulling>()
            .add_plugins(SyncComponentPlugin::<OcclusionCulling>::default());

        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };

        // TODO: prepare bind groups etc.
    }
}

#[derive(Component, Default, Reflect)]
#[reflect(Component, Default)]
pub struct OcclusionCulling;
