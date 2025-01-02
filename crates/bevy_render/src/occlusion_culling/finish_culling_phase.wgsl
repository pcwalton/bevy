// Messes with the indirect parameters.

#import bevy_pbr::mesh_preprocess_types::IndirectParameters

@group(0) @binding(0) var<storage, read_write> indirect_parameters: array<IndirectParameters>;
@group(0) @binding(1) var<storage, read_write> original_indirect_parameter_first_instances:
    array<u32>;

@compute
@workgroup_size(64)
fn finish_early_culling_phase(@builtin(global_invocation_id) global_invocation_id: vec3<u32>) {
    let instance_index = global_invocation_id.x;
    if (instance_index >= arrayLength(&indirect_parameters)) {
        return;
    }

    let instance_count = atomicLoad(&indirect_parameters[instance_index].instance_count);

    if (indirect_parameters[instance_index].first_instance == 0xffffffffu) {
        // Non-indexed mesh.
        original_indirect_parameter_first_instances[instance_index] =
            indirect_parameters[instance_index].base_vertex_or_first_instance;
        indirect_parameters[instance_index].base_vertex_or_first_instance += instance_count;
    } else {
        // Indexed mesh.
        original_indirect_parameter_first_instances[instance_index] =
            indirect_parameters[instance_index].first_instance;
        indirect_parameters[instance_index].first_instance += instance_count;
    }
}

@compute
@workgroup_size(64)
fn finish_main_culling_phase(@builtin(global_invocation_id) global_invocation_id: vec3<u32>) {
    let instance_index = global_invocation_id.x;
    if (instance_index >= arrayLength(&indirect_parameters)) {
        return;
    }

    let original_indirect_parameter_first_instance =
        original_indirect_parameter_first_instances[instance_index];

    if (indirect_parameters[instance_index].first_instance == 0xffffffffu) {
        // Non-indexed mesh.
        indirect_parameters[instance_index].base_vertex_or_first_instance =
            original_indirect_parameter_first_instance;
    } else {
        // Indexed mesh.
        indirect_parameters[instance_index].first_instance =
            original_indirect_parameter_first_instance;
    }
}
