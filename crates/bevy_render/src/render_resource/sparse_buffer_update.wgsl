struct SparseBufferUpdateMetadata {
    element_size: u32,
    element_stride: u32,
    element_update_count: u32,
};

@group(0) @binding(0) var<storage, read_write> dest_buffer: array<u32>;
@group(0) @binding(1) var<storage> src_buffer: array<u32>;
@group(0) @binding(2) var<storage> indices: array<u32>;
@group(0) @binding(3) var<uniform> metadata: SparseBufferUpdateMetadata;

@workgroup_size(256, 1, 1)
@compute
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let invocation_index = global_id.x;
    let update_index = invocation_index / metadata.element_size;
    if (update_index >= metadata.element_update_count) {
        return;
    }
    let word_index = invocation_index % metadata.element_size;
    let element_index = indices[update_index];
    let dest_index = element_index * metadata.element_stride + word_index;
    let src_index = update_index * metadata.element_stride + word_index;
    dest_buffer[dest_index] = src_buffer[src_index];
}
