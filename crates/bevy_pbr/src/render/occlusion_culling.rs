//! Two-phase occlusion culling.

use bevy_core_pipeline::core_3d::Camera3d;
use bevy_ecs::{query::With, system::Query};
use bevy_render::{camera::OcclusionCulling, render_resource::TextureUsages};

pub fn configure_occlusion_culling_view_targets(
    mut view_targets: Query<&mut Camera3d, With<OcclusionCulling>>,
) {
    for mut camera_3d in view_targets.iter_mut() {
        let mut depth_texture_usages = TextureUsages::from(camera_3d.depth_texture_usages);
        depth_texture_usages |= TextureUsages::TEXTURE_BINDING;
        camera_3d.depth_texture_usages = depth_texture_usages.into();
    }
}
