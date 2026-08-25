import bevy_render::view::View;
import bevy_render::globals::Globals;
import package::ui::{
    select_corner_radius
};

const PI: f32 = 3.14159265358979323846;
const SAMPLES: i32 = constants::SHADOW_SAMPLES;

struct BoxShadowInstance {
    world_from_local: vec4<f32>,
    color: vec4<f32>,
    radius_x: vec4<f32>,
    radius_y: vec4<f32>,
    translation: vec2<f32>,
    size: vec2<f32>,
    bounds: vec2<f32>,
    blur: f32,
    pad: f32,
}

@group(0) @binding(0) var<uniform> view: View;

@if(RETAINED_INSTANCES)
@group(1) @binding(0) var<storage, read> instances: array<BoxShadowInstance>;

struct BoxShadowVertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) point: vec2<f32>,
    @location(1) color: vec4<f32>,
    @location(2) @interpolate(flat) size: vec2<f32>,
    @location(3) @interpolate(flat) radius_x: vec4<f32>,
    @location(4) @interpolate(flat) radius_y: vec4<f32>,
    @location(5) @interpolate(flat) blur: f32,
}

fn gaussian(x: f32, sigma: f32) -> f32 {
    return exp(-(x * x) / (2. * sigma * sigma)) / (sqrt(2. * PI) * sigma);
}

// Approximates the Gauss error function: https://en.wikipedia.org/wiki/Error_function
fn erf(p: vec2<f32>) -> vec2<f32> {
    let s = sign(p);
    let a = abs(p);
    // fourth degree polynomial approximation for erf
    var result = 1.0 + (0.278393 + (0.230389 + 0.078108 * (a * a)) * a) * a;
    result = result * result;
    return s - s / (result * result);
}

fn horizontalRoundedBoxShadow(x: f32, y: f32, blur: f32, radius: vec2<f32>, half_size: vec2<f32>) -> f32 {    
    var c = half_size.x;
    if 0.0 < min(radius.x, radius.y) {
        let d = min(half_size.y - radius.y - abs(y), 0.);
        c = half_size.x - radius.x + radius.x * sqrt(max(0., 1. - d * d / (radius.y * radius.y)));
    }
    let integral = 0.5 + 0.5 * erf((x + vec2(-c, c)) * (sqrt(0.5) / blur));
    return integral.y - integral.x;
}

fn roundedBoxShadow(
    lower: vec2<f32>,
    upper: vec2<f32>,
    point: vec2<f32>,
    blur: f32,
    corners_x: vec4<f32>,
    corners_y: vec4<f32>,
) -> f32 {
    let center = (lower + upper) * 0.5;
    let half_size = (upper - lower) * 0.5;
    let p = point - center;
    let low = p.y - half_size.y;
    let high = p.y + half_size.y;
    let start = clamp(-3. * blur, low, high);
    let end = clamp(3. * blur, low, high);
    let step = (end - start) / f32(SAMPLES);
    var y = start + step * 0.5;
    var value: f32 = 0.0;
    for (var i = 0; i < SAMPLES; i++) {
        let corner = select_corner_radius(p, corners_x, corners_y);
        value += horizontalRoundedBoxShadow(p.x, p.y - y, blur, corner, half_size) * gaussian(y, blur) * step;
        y += step;
    }
    return value;
}

// The vertex shader logic, shared between retained and non-retained modes.
fn compute_vertex(vertex_position: vec2<f32>, instance: BoxShadowInstance) -> BoxShadowVertexOutput {
    var out: BoxShadowVertexOutput;
    let world_position = mat2x2<f32>(instance.world_from_local.xy, instance.world_from_local.zw) *
        vertex_position + instance.translation;
    out.position = view.clip_from_world * vec4(world_position, 0.0, 1.0);
    out.point = vertex_position * instance.bounds;
    out.color = instance.color;
    out.size = instance.size;
    out.radius_x = instance.radius_x;
    out.radius_y = instance.radius_y;
    out.blur = instance.blur;
    return out;
}

// In retained mode, we take in the instance index and look up the retained
// instance data in a storage buffer. In non-retained mode, we take the instance
// data directly (as non-retained mode is for platforms that don't support
// storage buffers).
@if(RETAINED_INSTANCES)
@vertex
fn vertex(
    @location(0) vertex_position: vec2<f32>,
    @location(1) instance_index: u32
) -> BoxShadowVertexOutput {
    return compute_vertex(vertex_position, instances[instance_index]);
}
@else
@vertex
fn vertex(
    @location(0) vertex_position: vec2<f32>,
    @location(1) @interpolate(flat) world_from_local: vec4<f32>,
    @location(2) @interpolate(flat) vertex_color: vec4<f32>,
    @location(3) @interpolate(flat) radius_x: vec4<f32>,
    @location(4) @interpolate(flat) radius_y: vec4<f32>,
    @location(5) @interpolate(flat) translation: vec2<f32>,
    @location(6) @interpolate(flat) size: vec2<f32>,
    @location(7) @interpolate(flat) bounds: vec2<f32>,
    @location(8) @interpolate(flat) blur: f32,
) -> BoxShadowVertexOutput {
    var instance: BoxShadowInstance;
    instance.world_from_local = world_from_local;
    instance.color = vertex_color;
    instance.radius_x = radius_x;
    instance.radius_y = radius_y;
    instance.translation = translation;
    instance.size = size;
    instance.bounds = bounds;
    instance.blur = blur;
    return compute_vertex(vertex_position, instance);
}

@fragment
fn fragment(
    in: BoxShadowVertexOutput,
) -> @location(0) vec4<f32> {
    let g = in.color.a * roundedBoxShadow(-0.5 * in.size, 0.5 * in.size, in.point, max(in.blur, 0.01), in.radius_x, in.radius_y);
    return vec4(in.color.rgb, g);
}


