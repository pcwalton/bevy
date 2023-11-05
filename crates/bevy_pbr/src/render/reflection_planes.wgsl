#define_import_path bevy_pbr::reflection_planes

#import bevy_pbr::view_transformations
#import bevy_pbr::mesh_view_bindings
#import bevy_pbr::mesh_view_bindings::{
    reflection_planes,
    reflection_planes_texture,
    reflection_planes_sampler,
}

fn intersect_ray_with_plane(
        plane_normal: vec3<f32>,
        plane_origin: vec3<f32>,
        ray_origin: vec3<f32>,
        ray_direction: vec3<f32>)
        -> f32 {
    let denom = dot(plane_normal, ray_direction);
    let offset = plane_origin - ray_origin;
    return dot(offset, plane_normal) / denom;
}

fn reflection_planes_light(frag_coord: vec2<f32>, view_space_normal: vec3<f32>) -> vec3<f32> {
    let near =
        mesh_view_bindings::view.inverse_projection * vec4(
            view_transformations::uv_to_ndc(view_transformations::frag_coord_to_uv(frag_coord)),
            1.0,
            1.0);
    let ray_direction = normalize(near.xyz / near.w);

    let reflection_plane = reflection_planes.data[0];

    let plane_hit = ray_direction * intersect_ray_with_plane(
        reflection_plane.view_space_normal, 
        reflection_plane.view_space_origin,
        vec3(0.0),
        ray_direction);
    let reflection = reflect(ray_direction, view_space_normal);

    let virtual_plane_hit = plane_hit - reflection * intersect_ray_with_plane(
        reflection_plane.view_space_normal, 
        reflection_plane.view_space_origin + reflection_plane.view_space_normal *
            reflection_plane.thickness,
        plane_hit,
        reflection);

    let mirror_virtual_plane_hit =
        (reflection_plane.view_space_reflection_matrix * vec4(virtual_plane_hit, 1.0)).xyz;
    let mirror_direction = normalize(mirror_virtual_plane_hit);

    let projected_mirror_direction = mesh_view_bindings::view.projection *
        vec4(mirror_direction, 1.0);
    let uv = view_transformations::ndc_to_uv(
        projected_mirror_direction.xy / projected_mirror_direction.w);

    return textureSample(reflection_planes_texture, reflection_planes_sampler, uv, 0u).rgb;
}
