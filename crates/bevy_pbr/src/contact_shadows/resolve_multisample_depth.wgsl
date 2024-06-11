#import bevy_core_pipeline::fullscreen_vertex_shader::FullscreenVertexOutput

@group(0) @binding(0) var source_depth_buffer: texture_depth_multisampled_2d;

@fragment
fn fragment(in: FullscreenVertexOutput) -> @builtin(frag_depth) f32 {
    let texture_size = textureDimensions(source_depth_buffer);
    let sample_count = i32(textureNumSamples(source_depth_buffer));

    let st = vec2<u32>(floor(in.position.xy));

    var sum = 0.0;
    for (var sample_index = 0; sample_index < sample_count; sample_index += 1) {
        sum += textureLoad(source_depth_buffer, st, sample_index);
    }

    return sum / f32(sample_count);
}
