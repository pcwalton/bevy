//! Occlusion culling.

use std::f32::consts::PI;

use bevy::{
    color::palettes::css::{SILVER, WHITE},
    core_pipeline::prepass::DepthPrepass,
    prelude::*,
};
use bevy_render::occlusion_culling::OcclusionCulling;

const OUTER_RADIUS: f32 = 3.0;
const OUTER_SUBDIVISION_COUNT: u32 = 5;
const ROTATION_SPEED: f32 = 0.01;
const SMALL_CUBE_SIZE: f32 = 0.1;
const LARGE_CUBE_SIZE: f32 = 2.0;

#[derive(Default, Component)]
struct SphereParent;

#[derive(Default, Component)]
struct LargeCube;

fn main() {
    App::new()
        .add_plugins(DefaultPlugins.set(WindowPlugin {
            primary_window: Some(Window {
                title: "Bevy Occlusion Culling Example".into(),
                ..default()
            }),
            ..default()
        }))
        .add_systems(Startup, setup)
        .add_systems(Update, spin_small_cubes)
        .add_systems(Update, spin_large_cubes)
        .run();
}

fn setup(
    mut commands: Commands,
    asset_server: Res<AssetServer>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    let small_cube = meshes.add(Cuboid::new(
        SMALL_CUBE_SIZE,
        SMALL_CUBE_SIZE,
        SMALL_CUBE_SIZE,
    ));
    let small_cube_material = materials.add(StandardMaterial {
        base_color: SILVER.into(),
        ..default()
    });

    let sphere_parent = commands
        .spawn(Transform::from_translation(Vec3::ZERO))
        .insert(Visibility::default())
        .insert(SphereParent)
        .id();

    let sphere = Sphere::new(OUTER_RADIUS)
        .mesh()
        .ico(OUTER_SUBDIVISION_COUNT)
        .unwrap();
    let sphere_positions = sphere.attribute(Mesh::ATTRIBUTE_POSITION).unwrap();
    for sphere_position in sphere_positions.as_float3().unwrap() {
        let sphere_position = Vec3::from_slice(sphere_position);
        let small_cube = commands
            .spawn(Mesh3d(small_cube.clone()))
            .insert(MeshMaterial3d(small_cube_material.clone()))
            .insert(Transform::from_translation(sphere_position))
            .id();
        commands.entity(sphere_parent).add_child(small_cube);
    }

    commands
        .spawn(Mesh3d(meshes.add(Cuboid::new(
            LARGE_CUBE_SIZE,
            LARGE_CUBE_SIZE,
            LARGE_CUBE_SIZE,
        ))))
        .insert(MeshMaterial3d(materials.add(StandardMaterial {
            base_color: WHITE.into(),
            base_color_texture: Some(asset_server.load("branding/icon.png")),
            ..default()
        })))
        .insert(Transform::IDENTITY)
        .insert(LargeCube);

    commands
        .spawn(DirectionalLight::default())
        .insert(Transform::from_rotation(Quat::from_euler(
            EulerRot::ZYX,
            0.0,
            PI * -0.15,
            PI * -0.15,
        )));

    commands
        .spawn(Camera3d::default())
        .insert(Transform::from_xyz(0.0, 0.0, 9.0).looking_at(Vec3::ZERO, Vec3::Y))
        .insert(DepthPrepass)
        .insert(OcclusionCulling)
        .insert(Msaa::Off);
}

fn spin_small_cubes(mut sphere_parents: Query<&mut Transform, With<SphereParent>>) {
    for mut sphere_parent_transform in &mut sphere_parents {
        sphere_parent_transform.rotate_y(ROTATION_SPEED);
    }
}

fn spin_large_cubes(mut large_cubes: Query<&mut Transform, With<LargeCube>>) {
    for mut transform in &mut large_cubes {
        transform.rotate(Quat::from_euler(
            EulerRot::XYZ,
            0.13 * ROTATION_SPEED,
            0.29 * ROTATION_SPEED,
            0.35 * ROTATION_SPEED,
        ));
    }
}
