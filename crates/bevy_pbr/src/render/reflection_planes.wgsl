#define_import_path bevy_pbr::reflection_planes

struct ReflectionPlane {
    transform: mat4x4<f32>,
    index: u32,
};

struct ReflectionPlanes {
    data: array<ReflectionPlane, 256u>,
};

fn reflection_planes_light() -> vec3<f32> {
    // TODO(pcwalton): Fill this in.
    return vec3(1.0, 0.0, 0.0);
}
