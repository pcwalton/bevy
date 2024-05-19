//! Displays an example model with sheen.

use bevy::{math::vec3, prelude::*};

fn main() {
    App::new()
        .add_plugins(DefaultPlugins)
        .add_systems(Startup, setup)
        .run();
}

fn setup(mut commands: Commands, asset_server: Res<AssetServer>) {
    commands
        .spawn(Camera3dBundle {
            transform: Transform::from_xyz(-0.8, 0.8, 0.8)
                .looking_at(vec3(0.0, 0.25, 0.0), Vec3::Y),
            ..default()
        })
        .insert(EnvironmentMapLight {
            diffuse_map: asset_server.load("environment_maps/pisa_diffuse_rgb9e5_zstd.ktx2"),
            specular_map: asset_server.load("environment_maps/pisa_specular_rgb9e5_zstd.ktx2"),
            intensity: 250.0,
        });

    commands.spawn(DirectionalLightBundle::default());

    commands.spawn(SceneBundle {
        scene: asset_server.load("models/SheenChair/SheenChair.gltf#Scene0"),
        ..default()
    });
}
