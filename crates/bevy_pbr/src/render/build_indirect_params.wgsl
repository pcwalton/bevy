// Building indirect parameters

#import bevy_pbr::mesh_preprocess_types::{
    IndirectBatchSet,
    IndirectParametersIndexed,
    IndirectParametersNonIndexed,
    IndirectParametersMetadata,
    MeshInput
}

@group(0) @binding(0) var<storage> current_input: array<MeshInput>;
@group(0) @binding(1) var<storage> indirect_parameters_metadata: array<IndirectParametersMetadata>;
@group(0) @binding(2) var<storage, read_write> indirect_batch_sets: array<IndirectBatchSet>;

#ifdef INDEXED
@group(0) @binding(3) var<storage, read_write> indirect_parameters:
    array<IndirectParametersIndexed>;
#else   // INDEXED
@group(0) @binding(3) var<storage, read_write> indirect_parameters:
    array<IndirectParametersNonIndexed>;
#endif  // INDEXED

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
    let batch_set_index = indirect_parameters_metadata[instance_index].batch_set_index;

#ifdef EARLY
    let instance_count = atomicLoad(&indirect_parameters_metadata[instance_index].early_instance_count);
#else   // EARLY
    let instance_count = atomicLoad(&indirect_parameters_metadata[instance_index].late_instance_count);
#endif  // EARLY

    var indirect_parameters_index = instance_index;
#ifdef MULTI_DRAW_INDIRECT_COUNT_SUPPORTED
    if (instance_count == 0u) {
        return;
    }

    if (batch_set_index != 0xffffffffu) {
        let indirect_parameters_base =
            indirect_batch_sets[batch_set_index].indirect_parameters_base;
        let indirect_parameters_offset =
            atomicAdd(&indirect_batch_sets[batch_set_index].indirect_parameters_count, 1u);

        indirect_parameters_index = indirect_parameters_base + indirect_parameters_offset;
    }
#endif  // MULTI_DRAW_INDIRECT_COUNT_SUPPORTED

#ifdef OCCLUSION_CULLING
#ifdef EARLY
    indirect_parameters[indirect_parameters_index].instance_count = instance_count;
    indirect_parameters[indirect_parameters_index].first_instance =
        base_output_index + instance_count;
#else   // EARLY
    indirect_parameters[indirect_parameters_index].instance_count += instance_count;
    indirect_parameters[indirect_parameters_index].first_instance = base_output_index;
#endif  // EARLY
#else   // OCCLUSION_CULLING
    indirect_parameters[indirect_parameters_index].first_instance = base_output_index;
#endif  // OCCLUSION_CULLING

    indirect_parameters[indirect_parameters_index].base_vertex =
        current_input[mesh_index].first_vertex_index;

#ifdef INDEXED
    indirect_parameters[indirect_parameters_index].index_count =
        current_input[mesh_index].index_count;
    indirect_parameters[indirect_parameters_index].first_index =
        current_input[mesh_index].first_index_index;
#else   // INDEXED
    indirect_parameters[indirect_parameters_index].vertex_count =
        current_input[mesh_index].index_count;
#endif  // INDEXED
}