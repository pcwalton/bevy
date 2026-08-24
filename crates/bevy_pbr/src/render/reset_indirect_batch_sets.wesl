// Resets the indirect draw counts to zero.
//
// This shader is needed because we reuse the same indirect batch set count
// buffer (i.e. the buffer that gets passed to `multi_draw_indirect_count` to
// determine how many objects to draw) between phases (early, late, and main).
// Before launching `build_indirect_params.wesl`, we need to reinitialize the
// value to 0.

import bevy_render::occlusion_culling::mesh_preprocess_types::IndirectBatchSet;
import bevy_render::utils::get_extended_instance_index;

@group(0) @binding(0) var<storage, read_write> indirect_batch_sets: array<IndirectBatchSet>;

@compute
@workgroup_size(64)
fn main(
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
    @builtin(num_workgroups) num_workgroups: vec3<u32>,
    @builtin(local_invocation_id) local_invocation_id: vec3<u32>,
) {
    // Figure out our instance index. If this thread doesn't correspond to any
    // index, bail.
    let instance_index =
        get_extended_instance_index(workgroup_id, num_workgroups, local_invocation_id, 64u);
    if (instance_index >= arrayLength(&indirect_batch_sets)) {
        return;
    }

    // Reset the number of batch sets to 0.
    atomicStore(&indirect_batch_sets[instance_index].indirect_parameters_count, 0u);
}
