// Messes with the indirect parameters.

#import bevy_pbr::mesh_preprocess_types::IndirectParameters

@group(0) @binding(0) var<storage, read_write> indirect_parameters: array<IndirectParameters>;

fn finish_early_culling_phase() {}

fn finish_main_culling_phase() {}
