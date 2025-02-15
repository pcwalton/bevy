#define_import_path bevy_render::bindless

#ifdef BINDLESS

@group(2) @binding(1) var<storage> bindless_buffers: binding_array<array<u32>>;
@group(2) @binding(2) var<storage, read_write> bindless_buffers_rw: binding_array<array<u32>>;
@group(2) @binding(3) var bindless_samplers_filtering: binding_array<sampler>;
@group(2) @binding(4) var bindless_samplers_non_filtering: binding_array<sampler>;
@group(2) @binding(5) var bindless_samplers_comparison: binding_array<sampler>;
@group(2) @binding(6) var bindless_textures_1d: binding_array<texture_1d<f32>>;
@group(2) @binding(7) var bindless_textures_2d: binding_array<texture_2d<f32>>;
@group(2) @binding(8) var bindless_textures_2d_array: binding_array<texture_2d_array<f32>>;
@group(2) @binding(9) var bindless_textures_3d: binding_array<texture_3d<f32>>;
@group(2) @binding(10) var bindless_textures_cube: binding_array<texture_cube<f32>>;
@group(2) @binding(11) var bindless_textures_cube_array: binding_array<texture_cube_array<f32>>;

#endif  // BINDLESS
