//! Demonstrates fog volumes.

use bevy::{
    math::vec3,
    pbr::{FogVolume, VolumetricFogSettings, VolumetricLight},
    prelude::*,
};

fn main() {
    App::new()
        .add_plugins(DefaultPlugins)
        .add_systems(Startup, setup)
        .run();
}

fn setup(mut commands: Commands) {
    commands
        .spawn(SpatialBundle {
            visibility: Visibility::Visible,
            transform: Transform::from_xyz(0.0, 0.5, 0.0),
            ..default()
        })
        .insert(FogVolume::default());

    commands
        .spawn(DirectionalLightBundle {
            transform: Transform::from_xyz(1.0, 1.0, -0.3).looking_at(vec3(0.0, 0.5, 0.0), Vec3::Y),
            ..default()
        })
        .insert(VolumetricLight);

    commands
        .spawn(Camera3dBundle {
            transform: Transform::from_xyz(-2.5, 4.5, 9.0).looking_at(vec3(0.0, 0.5, 0.0), Vec3::Y),
            ..default()
        })
        .insert(VolumetricFogSettings::default());
}
