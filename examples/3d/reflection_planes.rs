use std::f32::consts::{FRAC_PI_4, PI};

use bevy::prelude::*;
use bevy_internal::{
    core_pipeline::prepass::DepthPrepass,
    math::vec3,
    pbr::{ExtendedMaterial, MaterialExtension, ReflectionPlane},
    prelude::shape::{Cube, Plane},
    render::{
        render_resource::{AsBindGroup, ShaderRef},
        texture::{ImageAddressMode, ImageLoaderSettings, ImageSampler, ImageSamplerDescriptor},
    },
};

const CAMERA_ROTATION_SPEED: f32 = 0.3;
const CAMERA_DISTANCE: f32 = 3.0;
const CAMERA_HEIGHT: f32 = 2.0;

const PLANE_SCALE: f32 = 100.0;

const THICKNESS: f32 = 0.3;

#[derive(Asset, AsBindGroup, TypePath, Debug, Clone)]
struct WaterRipplesExtension {
    #[texture(100, dimension = "3d")]
    #[sampler(101)]
    noise: Handle<Image>,
}

fn main() {
    App::new()
        .insert_resource(AmbientLight {
            color: Color::WHITE,
            brightness: 0.0,
        })
        .add_plugins(DefaultPlugins)
        .add_plugins(MaterialPlugin::<
            ExtendedMaterial<StandardMaterial, WaterRipplesExtension>,
        >::default())
        .add_systems(Startup, setup)
        .add_systems(Update, rotate)
        .run();
}

fn setup(
    mut commands: Commands,
    asset_server: Res<AssetServer>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut standard_materials: ResMut<Assets<StandardMaterial>>,
    mut extended_materials: ResMut<
        Assets<ExtendedMaterial<StandardMaterial, WaterRipplesExtension>>,
    >,
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
        |settings: &mut ImageLoaderSettings| settings.is_srgb = false,
    );

    // Add noise asset.
    let noise: Handle<Image> = asset_server.load_with_settings(
        "textures/Test3DTexture.ktx2",
        |settings: &mut ImageLoaderSettings| {
            settings.is_srgb = false;
            settings.sampler = ImageSampler::Descriptor(ImageSamplerDescriptor {
                address_mode_u: ImageAddressMode::Repeat,
                address_mode_v: ImageAddressMode::Repeat,
                address_mode_w: ImageAddressMode::Repeat,
                ..ImageSamplerDescriptor::linear()
            });
        },
    );

    let reflective_material = extended_materials.add(ExtendedMaterial {
        base: StandardMaterial {
            base_color: Color::rgba(1.0, 1.0, 1.0, 1.0),
            perceptual_roughness: 0.0,
            metallic: 1.0,
            cull_mode: None, // FIXME
            ..StandardMaterial::default()
        },
        extension: WaterRipplesExtension { noise },
    });

    let cube_material = standard_materials.add(StandardMaterial {
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
    commands.spawn(MaterialMeshBundle {
        mesh: plane,
        material: reflective_material.clone(),
        transform: Transform::from_scale(Vec3::splat(PLANE_SCALE)),
        ..MaterialMeshBundle::default()
    });

    // Spawn reflection plane.
    commands
        .spawn(SpatialBundle {
            transform: Transform::from_scale(Vec3::splat(PLANE_SCALE))
                .with_translation(vec3(0.0, -0.00001, 0.0)),
            ..SpatialBundle::default()
        })
        .insert(ReflectionPlane { thickness: THICKNESS });

    commands.spawn(DirectionalLightBundle {
        directional_light: DirectionalLight {
            shadows_enabled: true,
            ..DirectionalLight::default()
        },
        transform: Transform::from_rotation(Quat::from_euler(EulerRot::ZYX, 0.0, 0.0, -FRAC_PI_4)),
        ..DirectionalLightBundle::default()
    });

    commands.spawn(DirectionalLightBundle {
        transform: Transform::from_rotation(Quat::from_euler(
            EulerRot::ZYX,
            0.4,
            0.0,
            PI + FRAC_PI_4 * 1.3,
        )),
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

impl MaterialExtension for WaterRipplesExtension {
    fn fragment_shader() -> ShaderRef {
        "shaders/water_ripples.wgsl".into()
    }
}
