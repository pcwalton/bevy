#import bevy_pbr::{
    pbr_fragment::pbr_input_from_standard_material,
    pbr_functions::alpha_discard,
    mesh_view_bindings,
}

#ifdef PREPASS_PIPELINE
#import bevy_pbr::{
    prepass_io::{VertexOutput, FragmentOutput},
    pbr_deferred_functions::deferred_output,
}
#else
#import bevy_pbr::{
    forward_io::{VertexOutput, FragmentOutput},
    pbr_functions::{apply_pbr_lighting, main_pass_post_lighting_processing},
}
#endif

@group(1) @binding(100) var noise_texture: texture_3d<f32>;
@group(1) @binding(101) var noise_sampler: sampler;

const OCTAVES: i32 = 8;
const UV_SCALE: f32 = 10.0;
const ANIMATION_SPEED: f32 = 0.1;

fn octave(uv: vec2<f32>) -> vec2<f32> {
    let offset = vec2(1.0) / vec2<f32>(textureDimensions(noise_texture).xy);

    let z = mesh_view_bindings::globals.time * ANIMATION_SPEED;

    let ddx =
        textureSample(noise_texture, noise_sampler, vec3(uv + vec2(offset.x, 0.0), z)).r -
        textureSample(noise_texture, noise_sampler, vec3(uv - vec2(offset.x, 0.0), z)).r;

    let ddy =
        textureSample(noise_texture, noise_sampler, vec3(uv + vec2(0.0, offset.y), z)).r -
        textureSample(noise_texture, noise_sampler, vec3(uv - vec2(0.0, offset.y), z)).r;

    return vec2(ddx, ddy);
}

fn bump(world_tangent: vec4<f32>, N: vec3<f32>, uv: vec2<f32>) -> vec3<f32> {
    var grad = vec2(0.0);
    for (var i = 0; i < OCTAVES; i += 1) {
        grad += octave(pow(2.1, f32(i)) * uv + 0.12345 * f32(i)) / pow(1.958, f32(i));
    }

    var T: vec3<f32> = world_tangent.xyz;
    var B: vec3<f32> = world_tangent.w * cross(N, T);

    let Nt = normalize(vec3(grad, 1.0));
    return normalize(Nt.x * T + Nt.y * B + Nt.z * N);
}

@fragment
fn fragment(
    in: VertexOutput,
    @builtin(front_facing) is_front: bool,
) -> FragmentOutput {
    // generate a PbrInput struct from the StandardMaterial bindings
    var pbr_input = pbr_input_from_standard_material(in, is_front);

    let uv = in.uv * UV_SCALE;
    pbr_input.N = bump(in.world_tangent, pbr_input.world_normal, uv);

#ifdef PREPASS_PIPELINE
    // in deferred mode we can't modify anything after that, as lighting is run in a separate fullscreen shader.
    let out = deferred_output(in, pbr_input);
#else
    var out: FragmentOutput;
    // apply lighting
    out.color = apply_pbr_lighting(pbr_input);

    // apply in-shader post processing (fog, alpha-premultiply, and also tonemapping, debanding if the camera is non-hdr)
    // note this does not include fullscreen postprocessing effects like bloom.
    out.color = main_pass_post_lighting_processing(pbr_input, out.color);
#endif

    return out;
}

