// A postprocessing shader that implements volumetric fog via raymarching.
//
// The overall approach is well-described in:
// <https://www.scratchapixel.com/lessons/3d-basic-rendering/volume-rendering-for-developers/intro-volume-rendering.html>

#import bevy_core_pipeline::fullscreen_vertex_shader::FullscreenVertexOutput
#import bevy_pbr::mesh_view_bindings::{lights, view}
#import bevy_pbr::mesh_view_types::DIRECTIONAL_LIGHT_FLAGS_VOLUMETRIC_BIT
#import bevy_pbr::shadow_sampling::sample_shadow_map_hardware
#import bevy_pbr::shadows::{get_cascade_index, world_to_directional_light_local}
#import bevy_pbr::view_transformations::{
    frag_coord_to_ndc,
    position_ndc_to_view,
    position_ndc_to_world
}

struct VolumetricFog {
    fog_color: vec3<f32>,
    light_tint: vec3<f32>,
    ambient_color: vec3<f32>,
    ambient_intensity: f32,
    step_count: i32,
    max_depth: f32,
    absorption: f32,
    scattering: f32,
    density: f32,
    scattering_asymmetry: f32,
    light_intensity: f32,
}

@group(1) @binding(0) var<uniform> volumetric_fog: VolumetricFog;
@group(1) @binding(1) var color_texture: texture_2d<f32>;
@group(1) @binding(2) var color_sampler: sampler;

#ifdef MULTISAMPLED
@group(1) @binding(3) var depth_texture: texture_depth_multisampled_2d;
#else
@group(1) @binding(3) var depth_texture: texture_depth_2d;
#endif

// 1 / (4π)
const FRAC_4_PI: f32 = 0.07957747154594767;

// https://www.scratchapixel.com/lessons/3d-basic-rendering/volume-rendering-for-developers/ray-marching-get-it-right.html
fn henyey_greenstein(LdotV: f32) -> f32 {
    let g = volumetric_fog.scattering_asymmetry;
    let denom = 1.0 + g * g - 2.0 * g * LdotV;
    return FRAC_4_PI * (1.0 - g * g) / (denom * sqrt(denom));
}

@fragment
fn fragment(in: FullscreenVertexOutput) -> @location(0) vec4<f32> {
    let fog_color = volumetric_fog.fog_color;
    let ambient_color = volumetric_fog.ambient_color;
    let ambient_intensity = volumetric_fog.ambient_intensity;
    let step_count = volumetric_fog.step_count;
    let max_depth = volumetric_fog.max_depth;
    let absorption = volumetric_fog.absorption;
    let scattering = volumetric_fog.scattering;
    let density = volumetric_fog.density;
    let light_tint = volumetric_fog.light_tint;
    let light_intensity = volumetric_fog.light_intensity;

    let exposure = view.exposure;

    let frag_coord = in.position;
    let depth = textureLoad(depth_texture, vec2<i32>(frag_coord.xy), 0);

    let directional_light_count = lights.n_directional_lights;

    let Rd_ndc = vec3(frag_coord_to_ndc(in.position).xy, 1.0);
    let Rd_view = normalize(position_ndc_to_view(Rd_ndc));
    let Ro_world = view.world_position;
    let Rd_world = normalize(position_ndc_to_world(Rd_ndc) - Ro_world);

    let end_depth = min(
        volumetric_fog.max_depth,
        -position_ndc_to_view(frag_coord_to_ndc(vec4(in.position.xy, depth, 1.0))).z
    );
    let step_size = end_depth / f32(step_count);

    // Assume the light is `max_depth` away.
    let light_attenuation = exp(-density * max_depth * (absorption + scattering));
    let light_factor_per_step = light_attenuation * scattering * density * step_size *
        light_intensity * exposure;

    // TODO: initialize this with analytically evaluated integral of ambient light.
    //var fog_color = vec3(0.0);
    var accumulated_color = exp(-end_depth * (absorption + scattering)) * ambient_color *
        ambient_intensity;

    var background_alpha = 1.0;

    for (var light_index = 0u; light_index < directional_light_count; light_index += 1u) {
        // Volumetric lights are all sorted first, so this is fine.
        let light = &lights.directional_lights[light_index];
        if (((*light).flags & DIRECTIONAL_LIGHT_FLAGS_VOLUMETRIC_BIT) == 0) {
            break;
        }

        let depth_offset = (*light).shadow_depth_bias * (*light).direction_to_light.xyz;
        // Assume that it's infinitely far.
        //let LdotV = dot(normalize((*light).direction_to_light.xyz), -Rd_world);
        let LdotV = dot(normalize((*light).direction_to_light.xyz), Rd_world);
        let phase = henyey_greenstein(LdotV);
        let light_color_per_step = fog_color * (*light).color.rgb * light_tint * phase *
            light_factor_per_step;

        background_alpha = 1.0;

        for (var step = 0; step < step_count; step += 1) {
            // As an optimization, break if we've gotten too dark.
            if (background_alpha < 0.001) {
                break;
            }

            let P_world = Ro_world + Rd_world * f32(step) * step_size;
            let P_view = Rd_view * f32(step) * step_size;

            let cascade_index = get_cascade_index(light_index, P_view.z);
            let light_local = world_to_directional_light_local(
                light_index,
                cascade_index,
                vec4(P_world + depth_offset, 1.0)
            );

            let sample_attenuation = exp(-step_size * density * (absorption + scattering));
            background_alpha *= sample_attenuation;

            // Compute in-scattering.

            var local_light_attenuation = f32(light_local.w != 0.0);
            if (local_light_attenuation != 0.0) {
                let cascade = &(*light).cascades[cascade_index];
                let array_index = i32((*light).depth_texture_base_index + cascade_index);
                local_light_attenuation *=
                    sample_shadow_map_hardware(light_local.xy, light_local.z, array_index);
            }

            if (local_light_attenuation != 0.0) {
                accumulated_color += light_color_per_step * local_light_attenuation *
                    background_alpha;
            }
        }
    }

    let source = textureSample(color_texture, color_sampler, in.uv);
    return vec4(source.rgb * background_alpha + accumulated_color, source.a);
}
