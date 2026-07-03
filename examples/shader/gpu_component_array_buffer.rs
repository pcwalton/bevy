//! A shader and a material that uses it.

use std::time::Duration;

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

#[derive(Resource)]
struct AppData {
    mesh: Handle<Mesh>,
    material_light: Handle<CustomMaterial>,
    material_dark: Handle<CustomMaterial>,
    rng: ChaCha8Rng,
}

fn main() {
    App::new()
        .add_plugins((
            DefaultPlugins,
            MaterialPlugin::<CustomMaterial>::default(),
            GpuComponentArrayBufferPlugin::<CustomMaterialData>::default(),
        ))
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
    mut shader_buffers: ResMut<Assets<ShaderBuffer>>,
    asset_server: Res<AssetServer>,
) {
    let component_array = GpuComponentArray::<CustomMaterialData>::new(&mut shader_buffers);
    let buffer = component_array.buffer.clone();
    commands.insert_resource(component_array);

    let mesh = meshes.add(Cuboid::default());
    let material_dark = materials.add(CustomMaterial {
        data: buffer.clone(),
        color_texture: asset_server.load("branding/bevy_bird_dark.png"),
    });
    let material_light = materials.add(CustomMaterial {
        data: buffer,
        color_texture: asset_server.load("branding/icon.png"),
    });

    commands.insert_resource(AppData {
        mesh,
        material_dark,
        material_light,
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
    let material = if app_data.rng.random_bool(0.5) {
        app_data.material_light.clone()
    } else {
        app_data.material_dark.clone()
    };

    commands.spawn((
        Mesh3d(app_data.mesh.clone()),
        MeshMaterial3d(material),
        Transform::from_xyz(xz_offset.x, 0.5, xz_offset.y).with_scale(Vec3::splat(0.1)),
        CustomMaterialData { color },
    ));
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

// This struct defines the data that will be passed to your shader
#[derive(Asset, TypePath, AsBindGroup, Debug, Clone)]
#[bindless(index_table(range(0..4)))]
struct CustomMaterial {
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

// The Material trait is very configurable, but comes with sensible defaults
// for all methods. You only need to implement functions for features that
// need non-default behavior. See the Material api docs for details!
impl Material for CustomMaterial {
    fn fragment_shader() -> ShaderRef {
        SHADER_ASSET_PATH.into()
    }
}
