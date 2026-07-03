//! A shader and a material that uses it.

use std::time::Duration;

use argh::FromArgs;
use bevy::{
    ecs::{query::QueryItem, system::lifetimeless::Read},
    prelude::*,
    reflect::TypePath,
    render::{
        gpu_component_array_buffer::{
            GpuComponentArray, GpuComponentArrayBuffer, GpuComponentArrayBufferPlugin,
        },
        render_resource::{AsBindGroup, ShaderType},
        storage::ShaderBuffer,
    },
    shader::ShaderRef,
    time::common_conditions::on_timer,
};
use bytemuck::{Pod, Zeroable};
use chacha20::ChaCha8Rng;
use rand::{seq::IndexedRandom, RngExt as _, SeedableRng as _};

/// This example uses a shader source file from the assets subdirectory
const SHADER_ASSET_PATH: &str = "shaders/gpu_component_array_buffer.wgsl";
const BINDLESS_SHADER_ASSET_PATH: &str = "shaders/gpu_component_array_buffer_bindless.wgsl";

#[derive(FromArgs, Resource)]
/// Demonstrates use of `GpuComponentArrayBuffer` to store custom
/// per-mesh-instance data
pub struct Args {
    /// enable bindless
    #[argh(switch)]
    bindless: bool,
}

#[derive(Resource)]
struct AppData {
    mesh: Handle<Mesh>,
    materials: AppMaterials,
    rng: ChaCha8Rng,
}

enum AppMaterials {
    NonBindless {
        material_light: Handle<CustomMaterial>,
        material_dark: Handle<CustomMaterial>,
    },
    Bindless {
        material_light: Handle<CustomBindlessMaterial>,
        material_dark: Handle<CustomBindlessMaterial>,
    },
}

fn main() {
    let args: Args = argh::from_env();

    App::new()
        .add_plugins((
            DefaultPlugins,
            MaterialPlugin::<CustomMaterial>::default(),
            MaterialPlugin::<CustomBindlessMaterial>::default(),
            GpuComponentArrayBufferPlugin::<CustomMaterialData>::default(),
        ))
        .insert_resource(args)
        .add_systems(Startup, setup)
        .add_systems(
            Update,
            (
                add_cube.run_if(on_timer(Duration::from_millis(300))),
                remove_cube.run_if(on_timer(Duration::from_millis(1000))),
            ),
        )
        .run();
}

/// set up a simple 3D scene
fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<CustomMaterial>>,
    mut bindless_materials: ResMut<Assets<CustomBindlessMaterial>>,
    mut shader_buffers: ResMut<Assets<ShaderBuffer>>,
    asset_server: Res<AssetServer>,
    args: Res<Args>,
) {
    let component_array = GpuComponentArray::<CustomMaterialData>::new(&mut shader_buffers);
    let buffer = component_array.buffer.clone();
    commands.insert_resource(component_array);

    let mesh = meshes.add(Cuboid::default());

    let (texture_dark, texture_light) = (
        asset_server.load("branding/bevy_bird_dark.png"),
        asset_server.load("branding/icon.png"),
    );

    let materials = if args.bindless {
        AppMaterials::Bindless {
            material_light: bindless_materials.add(CustomBindlessMaterial {
                data: buffer.clone(),
                color_texture: texture_light,
            }),
            material_dark: bindless_materials.add(CustomBindlessMaterial {
                data: buffer.clone(),
                color_texture: texture_dark,
            }),
        }
    } else {
        AppMaterials::NonBindless {
            material_light: materials.add(CustomMaterial {
                data: buffer.clone(),
                color_texture: texture_light,
            }),
            material_dark: materials.add(CustomMaterial {
                data: buffer.clone(),
                color_texture: texture_dark,
            }),
        }
    };

    commands.insert_resource(AppData {
        mesh,
        materials,
        rng: ChaCha8Rng::seed_from_u64(12345),
    });

    // camera
    commands.spawn((
        Camera3d::default(),
        Transform::from_xyz(-2.0, 1.25, 2.5).looking_at(Vec3::ZERO, Vec3::Y),
    ));
}

fn add_cube(mut commands: Commands, mut app_data: ResMut<AppData>) {
    let xz_offset = vec2(
        app_data.rng.random_range((-1.0)..1.0),
        app_data.rng.random_range((-1.0)..1.0),
    );
    let color = vec3(
        app_data.rng.random_range((0.0)..1.0),
        app_data.rng.random_range((0.0)..1.0),
        app_data.rng.random_range((0.0)..1.0),
    );
    let use_light_material = app_data.rng.random_bool(0.5);

    let mut entity_commands = commands.spawn((
        Mesh3d(app_data.mesh.clone()),
        Transform::from_xyz(xz_offset.x, 0.5, xz_offset.y).with_scale(Vec3::splat(0.1)),
        CustomMaterialData { color },
    ));

    match (&app_data.materials, use_light_material) {
        (
            &AppMaterials::Bindless {
                material_light: ref material,
                ..
            },
            true,
        )
        | (
            &AppMaterials::Bindless {
                material_dark: ref material,
                ..
            },
            false,
        ) => {
            entity_commands.insert(MeshMaterial3d(material.clone()));
        }
        (
            &AppMaterials::NonBindless {
                material_light: ref material,
                ..
            },
            true,
        )
        | (
            &AppMaterials::NonBindless {
                material_dark: ref material,
                ..
            },
            false,
        ) => {
            entity_commands.insert(MeshMaterial3d(material.clone()));
        }
    }

    println!("spawned cube");
}

fn remove_cube(
    mut commands: Commands,
    mut app_data: ResMut<AppData>,
    cubes: Query<Entity, With<CustomMaterialData>>,
) {
    let all_cubes: Vec<Entity> = cubes.iter().collect();
    if let Some(&cube_to_despawn) = all_cubes.choose(&mut app_data.rng) {
        commands.entity(cube_to_despawn).despawn();
    }
}

#[derive(Asset, TypePath, AsBindGroup, Debug, Clone)]
struct CustomMaterial {
    #[storage(0, read_only)]
    data: Handle<ShaderBuffer>,
    #[texture(1)]
    #[sampler(2)]
    color_texture: Handle<Image>,
}

#[derive(Asset, TypePath, AsBindGroup, Debug, Clone)]
#[bindless(index_table(range(0..4)))]
struct CustomBindlessMaterial {
    #[storage(1, read_only, binding_array(4))]
    data: Handle<ShaderBuffer>,
    #[texture(2)]
    #[sampler(3)]
    color_texture: Handle<Image>,
}

#[derive(Clone, Copy, Component, Debug)]
struct CustomMaterialData {
    color: Vec3,
}

#[derive(Clone, Copy, Default, ShaderType, Pod, Zeroable)]
#[repr(C)]
struct GpuCustomMaterialData {
    color: Vec3,
    pad: u32,
}

impl GpuComponentArrayBuffer for CustomMaterialData {
    type QueryData = Read<CustomMaterialData>;
    type QueryFilter = Changed<CustomMaterialData>;
    type Out = GpuCustomMaterialData;

    fn extract_component(data: QueryItem<'_, '_, Self::QueryData>) -> Option<Self::Out> {
        Some(GpuCustomMaterialData {
            color: data.color,
            pad: 0,
        })
    }
}

impl Material for CustomMaterial {
    fn fragment_shader() -> ShaderRef {
        SHADER_ASSET_PATH.into()
    }
}

impl Material for CustomBindlessMaterial {
    fn fragment_shader() -> ShaderRef {
        BINDLESS_SHADER_ASSET_PATH.into()
    }
}
