use std::f32::consts::FRAC_PI_4;

use bevy::prelude::*;
use bevy_internal::{
    asset::AssetLoader,
    core_pipeline::prepass::DepthPrepass,
    math::vec3,
    pbr::ReflectionPlane,
    prelude::shape::{Cube, Plane},
    render::texture::{ImageFormatSetting, ImageLoaderSettings, ImageSampler},
};

const CAMERA_ROTATION_SPEED: f32 = 0.3;
const CAMERA_DISTANCE: f32 = 3.0;
const CAMERA_HEIGHT: f32 = 2.0;

fn main() {
    App::new()
        .insert_resource(AmbientLight {
            color: Color::WHITE,
            brightness: 0.2,
        })
        .add_plugins(DefaultPlugins)
        .add_systems(Startup, setup)
        .add_systems(Update, rotate)
        .run();
}

fn setup(
    mut commands: Commands,
    asset_server: Res<AssetServer>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    // Add mesh assets.
    let plane = meshes.add(
        Mesh::from(Plane::default())
            .with_generated_tangents()
            .unwrap(),
    );
    let cube = meshes.add(Cube::default().into());

    // Add normal map asset.
    let normal_map: Handle<Image> = asset_server.load_with_settings(
        "textures/NormalMap.png",
        |settings: &mut ImageLoaderSettings| {
            settings.is_srgb = false;
        },
    );

    let reflective_material = materials.add(StandardMaterial {
        base_color: Color::rgba(0.1, 0.1, 0.1, 1.0),
        perceptual_roughness: 0.0,
        metallic: 1.0,
        cull_mode: None, // FIXME
        normal_map_texture: Some(normal_map),
        ..StandardMaterial::default()
    });
    let cube_material = materials.add(StandardMaterial {
        base_color: Color::rgba(0.7, 0.0, 0.0, 1.0),
        metallic: 0.0,
        cull_mode: None, // FIXME
        ..StandardMaterial::default()
    });

    commands.spawn(PbrBundle {
        mesh: cube,
        material: cube_material,
        transform: Transform::from_xyz(0.0, 0.5, 0.0),
        ..PbrBundle::default()
    });

    // Spawn mirror.
    commands.spawn(PbrBundle {
        mesh: plane,
        material: reflective_material.clone(),
        transform: Transform::from_scale(Vec3::splat(10.0)),
        ..PbrBundle::default()
    });

    // Spawn reflection plane.
    commands
        .spawn(SpatialBundle {
            transform: Transform::from_scale(Vec3::splat(10.0))
                .with_translation(vec3(0.0, -0.00001, 0.0)),
            ..SpatialBundle::default()
        })
        .insert(ReflectionPlane { thickness: 0.2 });

    commands.spawn(DirectionalLightBundle {
        directional_light: DirectionalLight {
            shadows_enabled: true,
            ..DirectionalLight::default()
        },
        transform: Transform::from_rotation(Quat::from_euler(EulerRot::ZYX, 0.0, 0.0, -FRAC_PI_4)),
        ..DirectionalLightBundle::default()
    });

    commands.spawn((
        Camera3dBundle {
            camera: Camera {
                hdr: true,
                ..default()
            },
            ..default()
        },
        DepthPrepass,
    ));
}

fn rotate(mut cameras: Query<&mut Transform, With<Camera>>, time: Res<Time>) {
    let theta = time.elapsed().as_secs_f32() * CAMERA_ROTATION_SPEED;
    for mut transform in &mut cameras {
        *transform = Transform::from_xyz(
            f32::cos(theta) * CAMERA_DISTANCE,
            CAMERA_HEIGHT,
            f32::sin(theta) * CAMERA_DISTANCE,
        )
        .looking_at(vec3(0.0, 0.5, 0.0), Vec3::Y);
    }
}
