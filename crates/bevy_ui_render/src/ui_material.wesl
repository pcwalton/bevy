import bevy_render::{
    view::View,
    globals::Globals,
};
import package::ui_vertex_output::UiVertexOutput;

@group(0) @binding(0)
var<uniform> view: View;
@group(0) @binding(1)
var<uniform> globals: Globals;

@vertex
fn vertex(
    @location(0) vertex_position: vec2<f32>,
    @location(1) @interpolate(flat) world_from_local: vec4<f32>,
    @location(2) @interpolate(flat) border_widths: vec4<f32>,
    @location(3) @interpolate(flat) border_radius_x: vec4<f32>,
    @location(4) @interpolate(flat) border_radius_y: vec4<f32>,
    @location(5) @interpolate(flat) size: vec2<f32>,
    @location(6) @interpolate(flat) translation: vec2<f32>,
    @location(7) @interpolate(flat) uv_scale: vec2<f32>,
    @location(8) @interpolate(flat) uv_offset: vec2<f32>,
) -> UiVertexOutput {
    var out: UiVertexOutput;
    let world_position = mat2x2<f32>(world_from_local.xy, world_from_local.zw) * vertex_position +
        translation;
    out.uv = uv_offset + (vertex_position + vec2(0.5)) * uv_scale;
    out.position = view.clip_from_world * vec4<f32>(world_position, 0.0, 1.0);
    out.size = size;
    out.border_widths = border_widths;
    out.border_radius_x = border_radius_x;
    out.border_radius_y = border_radius_y;
    return out;
}

@fragment
fn fragment(in: UiVertexOutput) -> @location(0) vec4<f32> {
    return vec4<f32>(1.0);
}
