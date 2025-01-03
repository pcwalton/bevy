// Building indirect parameters

#import bevy_pbr::mesh_preprocess_types::{IndirectParameters, IndirectParametersMetadata, MeshInput}

@group(0) @binding(0) var<storage> current_input: array<MeshInput>;
@group(0) @binding(1) var<storage> indirect_parameters_metadata: array<IndirectParametersMetadata>;
@group(0) @binding(2) var<storage, read_write> indirect_parameters: array<IndirectParameters>;

@compute
@workgroup_size(64)
fn main(@builtin(global_invocation_id) global_invocation_id: vec3<u32>) {
    // Figure out our instance index. If this thread doesn't correspond to any
    // index, bail.
    let instance_index = global_invocation_id.x;
    if (instance_index >= arrayLength(&indirect_parameters_metadata)) {
        return;
    }

    let mesh_index = indirect_parameters_metadata[instance_index].mesh_index;
    let base_output_index = indirect_parameters_metadata[instance_index].base_output_index;
    let instance_count = atomicLoad(&indirect_parameters_metadata[instance_index].instance_count);

    indirect_parameters[instance_index].instance_count = instance_count;

    //if ((indirect_parameters[instance_index].flags & MESH_FLAGS_INDEXED_BIT) != 0) {
        indirect_parameters[instance_index].vertex_count_or_index_count =
            current_input[mesh_index].index_count;
        indirect_parameters[instance_index].first_vertex_or_first_index =
            current_input[mesh_index].first_index_index;
        indirect_parameters[instance_index].base_vertex_or_first_instance =
            current_input[mesh_index].first_vertex_index;
        indirect_parameters[instance_index].first_instance = base_output_index;
    /*} else {
        indirect_parameters[instance_index].vertex_count_or_index_count =
            current_input[mesh_index].vertex_count;
        indirect_parameters[instance_index].first_vertex_or_first_index =
            current_input[mesh_index].first_vertex_index;
        indirect_parameters[instance_index].base_vertex_or_first_instance = base_output_index;
        indirect_parameters[instance_index].first_instance = 0xffffffffu;
    }*/
}