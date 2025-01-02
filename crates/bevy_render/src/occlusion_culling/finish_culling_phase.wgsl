// Messes with the indirect parameters.

#import bevy_pbr::mesh_preprocess_types::IndirectParameters

@group(0) @binding(0) var<storage, read_write> indirect_parameters: array<IndirectParameters>;

@compute
@workgroup_size(64)
fn finish_early_culling_phase(@builtin(global_invocation_id) global_invocation_id: vec3<u32>) {
    // TODO
}

@compute
@workgroup_size(64)
fn finish_main_culling_phase(@builtin(global_invocation_id) global_invocation_id: vec3<u32>) {
    // TODO
}
