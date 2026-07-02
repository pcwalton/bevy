#import bevy_pbr::{forward_io::VertexOutput, mesh_bindings::mesh}

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

#ifdef BINDLESS

@group(#{MATERIAL_BIND_GROUP}) @binding(0) var<storage> material_indices: array<CustomMaterialBindings>;
@group(#{MATERIAL_BIND_GROUP}) @binding(4) var<storage> material_data: binding_array<CustomMaterialDataArray>;

#else   // BINDLESS

@group(#{MATERIAL_BIND_GROUP}) @binding(1) var<storage> material_data: array<CustomMaterialData>;
@group(#{MATERIAL_BIND_GROUP}) @binding(2) var material_color_texture: texture_2d<f32>;
@group(#{MATERIAL_BIND_GROUP}) @binding(3) var material_color_sampler: sampler;

#endif  // BINDLESS

@fragment
fn fragment(in: VertexOutput) -> @location(0) vec4<f32> {
    let tag = mesh[in.instance_index].tag;
    let slot = mesh[in.instance_index].material_and_lightmap_bind_group_slot & 0xffffu;
    let data_index = material_indices[slot].data;
    let data = material_data[data_index].data_array[tag];
    //let color = vec3<f32>(1.0, 0.0, 0.0);
    let color = data.color;
    //let texture_color = textureSample(material_color_texture, material_color_sampler, in.uv).rgb;
    return vec4<f32>(color /* * texture_color */, 1.0);
}
