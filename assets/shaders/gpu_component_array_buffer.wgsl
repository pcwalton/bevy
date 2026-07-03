#import bevy_pbr::{forward_io::VertexOutput, mesh_bindings::mesh}
#import bevy_render::bindless::{bindless_samplers_filtering, bindless_textures_2d}

struct CustomMaterialData {
    color: vec3<f32>,
    pad: u32,
}

struct CustomMaterialDataArray {
    data_array: array<CustomMaterialData>,
}

struct CustomMaterialBindings {
    material: u32,              // 0
    data: u32,                  // 1
    color_texture: u32,         // 2
    color_texture_sampler: u32, // 3
}

@group(#{MATERIAL_BIND_GROUP}) @binding(0) var<storage> material_indices: array<CustomMaterialBindings>;
@group(#{MATERIAL_BIND_GROUP}) @binding(4) var<storage> material_data: binding_array<CustomMaterialDataArray>;

@fragment
fn fragment(in: VertexOutput) -> @location(0) vec4<f32> {
    let tag = mesh[in.instance_index].tag;
    let slot = mesh[in.instance_index].material_and_lightmap_bind_group_slot & 0xffffu;

    let data_index = material_indices[slot].data;
    let color_texture_index = material_indices[slot].color_texture;
    let color_texture_sampler_index = material_indices[slot].color_texture_sampler;

    let data = material_data[data_index].data_array[tag];
    let color = data.color;
    let texture_color = textureSample(
        bindless_textures_2d[color_texture_index],
        bindless_samplers_filtering[color_texture_sampler_index],
        in.uv
    ).rgb;

    return vec4<f32>(color * texture_color, 1.0);
}
