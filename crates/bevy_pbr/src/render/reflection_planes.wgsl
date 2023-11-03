#define_import_path bevy_pbr::reflection_planes

#import bevy_pbr::view_transformations
#import bevy_pbr::mesh_view_bindings::{reflection_planes_texture, reflection_planes_sampler}

fn reflection_planes_light(frag_coord: vec2<f32>) -> vec3<f32> {
    // TODO(pcwalton): Raytrace.
    let uv = view_transformations::frag_coord_to_uv(frag_coord);
    return textureSample(reflection_planes_texture, reflection_planes_sampler, uv, 0u).rgb;
}
