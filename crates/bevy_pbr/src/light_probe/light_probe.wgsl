#define_import_path bevy_pbr::light_probe

#import bevy_pbr::mesh_view_types::LightProbe

struct LightProbeQueryResult {
    texture_index: i32,
    intensity: f32,
    inverse_transform: mat4x4<f32>,
};

fn transpose_affine_matrix(matrix: mat3x4<f32>) -> mat4x4<f32> {
    let matrix4x4 = mat4x4<f32>(
        matrix[0],
        matrix[1],
        matrix[2],
        vec4<f32>(0.0, 0.0, 0.0, 1.0));
    return transpose(matrix4x4);
}
// Searches for a light probe that contains the fragment.
//
// TODO: Interpolate between multiple light probes.
fn query_light_probe(
    in_light_probes: array<LightProbe, 8u>,
    light_probe_count: i32,
    world_position: vec3<f32>,
) -> LightProbeQueryResult {
    // This is needed to index into the array with a non-constant expression.
    var light_probes = in_light_probes;

    var result: LightProbeQueryResult;
    result.texture_index = -1;

    for (var light_probe_index: i32 = 0;
            light_probe_index < light_probe_count;
            light_probe_index += 1) {
        let light_probe = light_probes[light_probe_index];

        // Unpack the inverse transform.
        let inverse_transform =
            transpose_affine_matrix(light_probe.inverse_transpose_transform);

        // Check to see if the transformed point is inside the unit cube
        // centered at the origin.
        let probe_space_pos = (inverse_transform * vec4<f32>(world_position, 1.0f)).xyz;
        if (all(abs(probe_space_pos) <= vec3(0.5f))) {
            result.texture_index = light_probe.cubemap_index;
            result.intensity = light_probe.intensity;
            result.inverse_transform = inverse_transform;
            break;
        }
    }

    return result;
}
