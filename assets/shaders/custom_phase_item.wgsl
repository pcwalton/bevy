// `custom_phase_item.wgsl`

struct Vertex {
    @location(0) position: vec3<f32>,
    @location(1) color: vec3<f32>,
};

struct Varyings {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) color: vec3<f32>,
};

@vertex
fn vertex(vertex: Vertex) -> Varyings {
    var varyings: Varyings;
    varyings.clip_position = vec4(vertex.position.xyz, 1.0);
    varyings.color = vertex.color;
    return varyings;
}

@fragment
fn fragment(varyings: Varyings) -> @location(0) vec4<f32> {
    return vec4(varyings.color, 1.0);
}
