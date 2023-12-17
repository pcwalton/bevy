#define_import_path bevy_pbr::environment_map

#import bevy_pbr::mesh_view_bindings as bindings
#import bevy_pbr::mesh_view_bindings::light_probes

struct EnvironmentMapLight {
    diffuse: vec3<f32>,
    specular: vec3<f32>,
};

fn environment_map_light(
    perceptual_roughness: f32,
    roughness: f32,
    diffuse_color: vec3<f32>,
    NdotV: f32,
    f_ab: vec2<f32>,
    N: vec3<f32>,
    R: vec3<f32>,
    F0: vec3<f32>,
    world_position: vec3<f32>,
) -> EnvironmentMapLight {
    var out: EnvironmentMapLight;

    // Split-sum approximation for image based lighting: https://cdn2.unrealengine.com/Resources/files/2013SiggraphPresentationsNotes-26915738.pdf
    // Technically we could use textureNumLevels(environment_map_specular) - 1 here, but we use a uniform
    // because textureNumLevels() does not work on WebGL2
    let radiance_level = perceptual_roughness * f32(bindings::lights.environment_map_smallest_specular_mip_level);

#ifdef ENVIRONMENT_MAP_LIGHT_PROBES
    // Search for a reflection probe that contains the fragment.
    //
    // TODO: Interpolate between multiple reflection probes.
    var cubemap_index: i32 = -1;
    for (var reflection_probe_index: i32 = 0;
            reflection_probe_index < light_probes.reflection_probe_count;
            reflection_probe_index += 1) {
        let reflection_probe = light_probes.reflection_probes[reflection_probe_index];

        // Transpose the inverse transpose transform to recover the inverse
        // transform.
        let inverse_transpose_transform = mat4x4<f32>(
            reflection_probe.inverse_transpose_transform[0],
            reflection_probe.inverse_transpose_transform[1],
            reflection_probe.inverse_transpose_transform[2],
            vec4<f32>(0.0, 0.0, 0.0, 1.0));
        let inverse_transform = transpose(inverse_transpose_transform);

        // Check to see if the transformed point is inside the unit cube
        // centered at the origin.
        let probe_space_pos = (inverse_transform * vec4<f32>(world_position, 1.0)).xyz;
        if (all(abs(probe_space_pos) <= vec3(0.5))) {
            cubemap_index = reflection_probe.cubemap_index;
            break;
        }
    }

    // If we didn't find a reflection probe, use the view environment map if applicable.
    if (cubemap_index < 0) {
        cubemap_index = light_probes.cubemap_index;
    }

    // If there's no cubemap, bail out.
    if (cubemap_index < 0) {
        out.diffuse = vec3(0.0);
        out.specular = vec3(0.0);
        return out;
    }

    let irradiance = textureSampleLevel(
        bindings::diffuse_environment_maps[cubemap_index],
        bindings::environment_map_sampler,
        vec3(N.xy, -N.z),
        0.0).rgb;

    let radiance = textureSampleLevel(
        bindings::specular_environment_maps[cubemap_index],
        bindings::environment_map_sampler,
        vec3(R.xy, -R.z),
        radiance_level).rgb;
#else   // ENVIRONMENT_MAP_LIGHT_PROBES
    if (light_probes.cubemap_index < 0) {
        out.diffuse = vec3(0.0);
        out.specular = vec3(0.0);
        return out;
    }

    let irradiance = textureSampleLevel(
        bindings::diffuse_environment_map,
        bindings::environment_map_sampler,
        vec3(N.xy, -N.z),
        0.0).rgb;

    let radiance = textureSampleLevel(
        bindings::specular_environment_map,
        bindings::environment_map_sampler,
        vec3(R.xy, -R.z),
        radiance_level).rgb;
#endif  // ENVIRONMENT_MAP_LIGHT_PROBES

    // No real world material has specular values under 0.02, so we use this range as a
    // "pre-baked specular occlusion" that extinguishes the fresnel term, for artistic control.
    // See: https://google.github.io/filament/Filament.html#specularocclusion
    let specular_occlusion = saturate(dot(F0, vec3(50.0 * 0.33)));

    // Multiscattering approximation: https://www.jcgt.org/published/0008/01/03/paper.pdf
    // Useful reference: https://bruop.github.io/ibl
    let Fr = max(vec3(1.0 - roughness), F0) - F0;
    let kS = F0 + Fr * pow(1.0 - NdotV, 5.0);
    let Ess = f_ab.x + f_ab.y;
    let FssEss = kS * Ess * specular_occlusion;
    let Ems = 1.0 - Ess;
    let Favg = F0 + (1.0 - F0) / 21.0;
    let Fms = FssEss * Favg / (1.0 - Ems * Favg);
    let FmsEms = Fms * Ems;
    let Edss = 1.0 - (FssEss + FmsEms);
    let kD = diffuse_color * Edss;

    out.diffuse = (FmsEms + kD) * irradiance;
    out.specular = FssEss * radiance;
    return out;
}