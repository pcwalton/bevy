// Bloom works by creating an intermediate texture with a bunch of mip levels, each half the size of the previous.
// You then downsample each mip (starting with the original texture) to the lower resolution mip under it, going in order.
// You then upsample each mip (starting from the smallest mip) and blend with the higher resolution mip above it (ending on the original texture).
//
// References:
// * [COD] - Next Generation Post Processing in Call of Duty - http://www.iryoku.com/next-generation-post-processing-in-call-of-duty-advanced-warfare
// * [PBB] - Physically Based Bloom - https://learnopengl.com/Guest-Articles/2022/Phys.-Based-Bloom

struct BloomUniforms {
    threshold_precomputations: vec4<f32>,
    viewport: vec4<f32>,
    scale: vec2<f32>,
    aspect: f32,
};

#ifdef BLOOM_COMPUTE
struct BloomImmediates {
    input_mip: u32,
    output_mip: u32,
    src_blend_factor: f32,
    dest_blend_factor: f32,
};
#endif  // BLOOM_COMPUTE

@group(0) @binding(0) var<uniform> uniforms: BloomUniforms;

#ifdef BLOOM_COMPUTE

@group(1) @binding(0) var textures: binding_array<texture_storage_2d<rg11b10ufloat, read_write>>;

var<immediate> immediates: BloomImmediates;

#else   // BLOOM_COMPUTE

@group(0) @binding(1) var s: sampler;
@group(0) @binding(2) var input_texture: texture_2d<f32>;

#endif  // BLOOM_COMPUTE

#ifdef FIRST_DOWNSAMPLE
// https://catlikecoding.com/unity/tutorials/advanced-rendering/bloom/#3.4
fn soft_threshold(color: vec3<f32>) -> vec3<f32> {
    let brightness = max(color.r, max(color.g, color.b));
    var softness = brightness - uniforms.threshold_precomputations.y;
    softness = clamp(softness, 0.0, uniforms.threshold_precomputations.z);
    softness = softness * softness * uniforms.threshold_precomputations.w;
    var contribution = max(brightness - uniforms.threshold_precomputations.x, softness);
    contribution /= max(brightness, 0.00001); // Prevent division by 0
    return color * contribution;
}
#endif

// luminance coefficients from Rec. 709.
// https://en.wikipedia.org/wiki/Rec._709
fn tonemapping_luminance(v: vec3<f32>) -> f32 {
    return dot(v, vec3<f32>(0.2126, 0.7152, 0.0722));
}

// http://graphicrants.blogspot.com/2013/12/tone-mapping.html
fn karis_average(color: vec3<f32>) -> f32 {
    // Luminance calculated based on Rec. 709 color primaries.
    // This must be done in *linear* color space.
    let luma = tonemapping_luminance(color) / 4.0;
    return 1.0 / (1.0 + luma);
}

fn texture_sample_offset(uv: vec2<f32>, offset: vec2<i32>) -> vec3<f32> {
#ifdef BLOOM_COMPUTE

    let dimensions = vec2<f32>(textureDimensions(textures[immediates.input_mip]));
    let st = uv * dimensions + vec2<f32>(offset) - vec2<f32>(0.5);

    var st0 = vec2<i32>(floor(st));
    var st1 = st0 + 1;
    let st_frac = fract(st);

    st0 = clamp(st0, vec2(0), vec2<i32>(dimensions) - 1);
    st1 = clamp(st1, vec2(0), vec2<i32>(dimensions) - 1);

    let color_00 = textureLoad(textures[immediates.input_mip], st0).rgb;
    let color_10 = textureLoad(textures[immediates.input_mip], vec2(st1.x, st0.y)).rgb;
    let color_01 = textureLoad(textures[immediates.input_mip], vec2(st0.x, st1.y)).rgb;
    let color_11 = textureLoad(textures[immediates.input_mip], st1).rgb;

    let color_0 = mix(color_00, color_10, st_frac.x);
    let color_1 = mix(color_01, color_11, st_frac.x);
    return mix(color_0, color_1, st_frac.y);

#else   // BLOOM_COMPUTE
#ifdef UNIFORM_SCALE

    // This is the fast path. When the bloom scale is uniform, the 13 tap sampling kernel can be
    // expressed with constant offsets.
    //
    // It's possible that this isn't meaningfully faster than the "slow" path. However, because it
    // is hard to test performance on all platforms, and uniform bloom is the most common case, this
    // path was retained when adding non-uniform (anamorphic) bloom. This adds a small, but nonzero,
    // cost to maintainability, but it does help me sleep at night.
    let uv_offset = vec2<f32>(offset) / vec2<f32>(textureDimensions(input_texture));
    return textureSample(input_texture, s, uv + uv_offset).rgb;

#else   // UNIFORM_SCALE

    // This is the flexible, but potentially slower, path for non-uniform sampling. Because the
    // sample is not a constant, and it can fall outside of the limits imposed on constant sample
    // offsets (-8..8), we have to compute the pixel offset in uv coordinates using the size of the
    // texture.
    //
    // It isn't clear if this is meaningfully slower than using the offset syntax, the spec doesn't
    // mention it anywhere: https://www.w3.org/TR/WGSL/#texturesample, but the fact that the offset
    // syntax uses a const-expr implies that it allows some compiler optimizations - maybe more
    // impactful on mobile?
    let uv_offset = uniforms.scale * vec2<f32>(offset) /
        vec2<f32>(textureDimensions(input_texture));
    return textureSample(input_texture, s, uv + uv_offset).rgb;

#endif  // UNIFORM_SCALE
#endif  // BLOOM_COMPUTE
}

fn texture_sample(uv: vec2<f32>) -> vec3<f32> {
    return texture_sample_offset(uv, vec2<i32>(0));
}

// [COD] slide 153
fn sample_input_13_tap(uv: vec2<f32>) -> vec3<f32> {
    let a = texture_sample_offset(uv, vec2<i32>(-2, 2)).rgb;
    let b = texture_sample_offset(uv, vec2<i32>(0, 2)).rgb;
    let c = texture_sample_offset(uv, vec2<i32>(2, 2)).rgb;
    let d = texture_sample_offset(uv, vec2<i32>(-2, 0)).rgb;
    let e = texture_sample(uv).rgb;
    let f = texture_sample_offset(uv, vec2<i32>(2, 0)).rgb;
    let g = texture_sample_offset(uv, vec2<i32>(-2, -2)).rgb;
    let h = texture_sample_offset(uv, vec2<i32>(0, -2)).rgb;
    let i = texture_sample_offset(uv, vec2<i32>(2, -2)).rgb;
    let j = texture_sample_offset(uv, vec2<i32>(-1, 1)).rgb;
    let k = texture_sample_offset(uv, vec2<i32>(1, 1)).rgb;
    let l = texture_sample_offset(uv, vec2<i32>(-1, -1)).rgb;
    let m = texture_sample_offset(uv, vec2<i32>(1, -1)).rgb;

#ifdef FIRST_DOWNSAMPLE
    // [COD] slide 168
    //
    // The first downsample pass reads from the rendered frame which may exhibit
    // 'fireflies' (individual very bright pixels) that should not cause the bloom effect.
    //
    // The first downsample uses a firefly-reduction method proposed by Brian Karis
    // which takes a weighted-average of the samples to limit their luma range to [0, 1].
    // This implementation matches the LearnOpenGL article [PBB].
    var group0 = (a + b + d + e) * (0.125f / 4.0f);
    var group1 = (b + c + e + f) * (0.125f / 4.0f);
    var group2 = (d + e + g + h) * (0.125f / 4.0f);
    var group3 = (e + f + h + i) * (0.125f / 4.0f);
    var group4 = (j + k + l + m) * (0.5f / 4.0f);
    group0 *= karis_average(group0);
    group1 *= karis_average(group1);
    group2 *= karis_average(group2);
    group3 *= karis_average(group3);
    group4 *= karis_average(group4);
    return group0 + group1 + group2 + group3 + group4;
#else
    var sample = (a + c + g + i) * 0.03125;
    sample += (b + d + f + h) * 0.0625;
    sample += (e + j + k + l + m) * 0.125;
    return sample;
#endif
}

// [COD] slide 162
fn sample_input_3x3_tent(uv: vec2<f32>) -> vec3<f32> {
    // While this is probably technically incorrect, it makes nonuniform bloom smoother, without
    // having any impact on uniform bloom, which simply evaluates to 1.0 here.
#ifdef BLOOM_COMPUTE
    let frag_size = uniforms.scale / vec2<f32>(textureDimensions(textures[immediates.input_mip]));
#else   // BLOOM_COMPUTE
    let frag_size = uniforms.scale / vec2<f32>(textureDimensions(input_texture));
#endif  // BLOOM_COMPUTE
    let x = frag_size.x;
    let y = frag_size.y;

    let a = texture_sample(vec2<f32>(uv.x - x, uv.y + y)).rgb;
    let b = texture_sample(vec2<f32>(uv.x, uv.y + y)).rgb;
    let c = texture_sample(vec2<f32>(uv.x + x, uv.y + y)).rgb;

    let d = texture_sample(vec2<f32>(uv.x - x, uv.y)).rgb;
    let e = texture_sample(vec2<f32>(uv.x, uv.y)).rgb;
    let f = texture_sample(vec2<f32>(uv.x + x, uv.y)).rgb;

    let g = texture_sample(vec2<f32>(uv.x - x, uv.y - y)).rgb;
    let h = texture_sample(vec2<f32>(uv.x, uv.y - y)).rgb;
    let i = texture_sample(vec2<f32>(uv.x + x, uv.y - y)).rgb;

    var sample = e * 0.25;
    sample += (b + d + f + h) * 0.125;
    sample += (a + c + g + i) * 0.0625;

    return sample;
}

#ifdef FIRST_DOWNSAMPLE
fn do_downsample_first(output_uv: vec2<f32>) -> vec3<f32> {
    let sample_uv = uniforms.viewport.xy + output_uv * uniforms.viewport.zw;
    var sample = sample_input_13_tap(sample_uv);
    // Lower bound of 0.0001 is to avoid propagating multiplying by 0.0 through the
    // downscaling and upscaling which would result in black boxes.
    // The upper bound is to prevent NaNs.
    // with f32::MAX (E+38) Chrome fails with ":value 340282346999999984391321947108527833088.0 cannot be represented as 'f32'"
    sample = clamp(sample, vec3<f32>(0.0001), vec3<f32>(3.40282347E+37));

#ifdef USE_THRESHOLD
    sample = soft_threshold(sample);
#endif

    return sample;
}
#endif  // FIRST_DOWNSAMPLE

#ifdef BLOOM_COMPUTE

@compute
@workgroup_size(16, 16, 1)
fn downsample(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let output_size = textureDimensions(textures[immediates.output_mip]).xy;
    if (global_id.x >= output_size.x || global_id.y >= output_size.y) {
        return;
    }
    let output_uv = (vec2<f32>(global_id.xy) + 0.5) / vec2<f32>(output_size);
    let sample = vec4<f32>(sample_input_13_tap(output_uv), 1.0);
    textureStore(textures[immediates.output_mip], global_id.xy, sample);
}

@compute
@workgroup_size(16, 16, 1)
fn upsample(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let output_size = textureDimensions(textures[immediates.output_mip]).xy;
    if (global_id.x >= output_size.x || global_id.y >= output_size.y) {
        return;
    }
    let output_uv = (vec2<f32>(global_id.xy) + 0.5) / vec2<f32>(output_size);
    let src = sample_input_3x3_tent(output_uv);

    let dest = textureLoad(textures[immediates.output_mip], global_id.xy).rgb;
    let sample = vec4(src * immediates.src_blend_factor + dest * immediates.dest_blend_factor, 1.0);
    textureStore(textures[immediates.output_mip], global_id.xy, sample);
}

#else   // BLOOM_COMPUTE

#ifdef FIRST_DOWNSAMPLE
@fragment
fn downsample_first(@location(0) output_uv: vec2<f32>) -> @location(0) vec4<f32> {
    return vec4<f32>(do_downsample_first(output_uv), 1.0);
}
#endif  // FIRST_DOWNSAMPLE

@fragment
fn downsample(@location(0) uv: vec2<f32>) -> @location(0) vec4<f32> {
    return vec4<f32>(sample_input_13_tap(uv), 1.0);
}

@fragment
fn upsample(@location(0) uv: vec2<f32>) -> @location(0) vec4<f32> {
    return vec4<f32>(sample_input_3x3_tent(uv), 1.0);
}

#endif  // BLOOM_COMPUTE
