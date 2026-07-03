#import bevy_pbr::{forward_io::VertexOutput, mesh_bindings::mesh}

struct CustomMaterialData {
    color: vec3<f32>,
    pad: u32,
}

@group(#{MATERIAL_BIND_GROUP}) @binding(0) var<storage> material_data: array<CustomMaterialBindings>;
@group(#{MATERIAL_BIND_GROUP}) @binding(1) var color_texture: texture_2d<f32>;
@group(#{MATERIAL_BIND_GROUP}) @binding(2) var color_texture_sampler: sampler;

@fragment
fn fragment(in: VertexOutput) -> @location(0) vec4<f32> {
    let tag = mesh[in.instance_index].tag;
    let color = material_data[tag].color;

    let texture_color = textureSample(color_texture, color_texture_sampler, in.uv).rgb;
    return vec4<f32>(color * texture_color, 1.0);
}
