// GPU mesh uniform building and culling.
//
// This is a compute shader that expands each `MeshInputUniform` out to a full
// `MeshUniform` for each view before rendering. (Thus `MeshInputUniform`
// and `MeshUniform` are in a 1:N relationship.) It runs in parallel for all
// meshes for all views. As part of this process, the shader gathers each
// mesh's transform on the previous frame and writes it into the `MeshUniform`
// so that TAA works.

#import bevy_pbr::mesh_types::Mesh
#import bevy_pbr::view_transformations
#import bevy_render::maths
#import bevy_render::view::View

// Per-frame data that the CPU supplies to the GPU.
struct MeshInput {
    // The model transform.
    world_from_local: mat3x4<f32>,
    // The lightmap UV rect, packed into 64 bits.
    lightmap_uv_rect: vec2<u32>,
    // Various flags.
    flags: u32,
    // The index of this mesh's `MeshInput` in the `previous_input` array, if
    // applicable. If not present, this is `u32::MAX`.
    previous_input_index: u32,
}

// Information about each mesh instance needed to cull it on GPU.
//
// At the moment, this just consists of its axis-aligned bounding box (AABB).
struct MeshCullingData {
    // The 3D center of the AABB in model space, padded with an extra unused
    // float value.
    aabb_center: vec4<f32>,
    // The 3D extents of the AABB in model space, divided by two, padded with
    // an extra unused float value.
    aabb_half_extents: vec4<f32>,
}

// One invocation of this compute shader: i.e. one mesh instance in a view.
struct PreprocessWorkItem {
    // The index of the `MeshInput` in the `current_input` buffer that we read
    // from.
    input_index: u32,
    // In direct mode, the index of the `Mesh` in `output` that we write to. In
    // indirect mode, the index of the `IndirectParameters` in
    // `indirect_parameters` that we write to.
    output_index: u32,
}

// The `wgpu` indirect parameters structure. This is a union of two structures.
// For more information, see the corresponding comment in
// `gpu_preprocessing.rs`.
struct IndirectParameters {
    // `vertex_count` or `index_count`.
    data0: u32,
    // `instance_count` in both structures.
    instance_count: atomic<u32>,
    // `first_vertex` in both structures.
    first_vertex: u32,
    // `first_instance` or `base_vertex`.
    data1: u32,
    // A read-only copy of `instance_index`.
    instance_index: u32,
}

// The current frame's `MeshInput`.
@group(0) @binding(0) var<storage> current_input: array<MeshInput>;
// The `MeshInput` values from the previous frame.
@group(0) @binding(1) var<storage> previous_input: array<MeshInput>;
// Indices into the `MeshInput` buffer.
//
// There may be many indices that map to the same `MeshInput`.
@group(0) @binding(2) var<storage> work_items: array<PreprocessWorkItem>;
// The output array of `Mesh`es.
@group(0) @binding(3) var<storage, read_write> main_pass_output: array<Mesh>;

#ifdef INDIRECT
// The array of indirect parameters for drawcalls.
@group(0) @binding(4) var<storage, read_write> main_pass_indirect_parameters:
    array<IndirectParameters>;
#endif

#ifdef FRUSTUM_CULLING

// Data needed to cull the meshes.
//
// At the moment, this consists only of AABBs.
@group(0) @binding(5) var<storage> mesh_culling_data: array<MeshCullingData>;

// The view data, including the view matrix.
@group(0) @binding(6) var<uniform> view: View;

#ifdef OCCLUSION_CULLING

// The output array of `Mesh`es for the next prepass phase (early or late).
@group(0) @binding(7) var<storage, read_write> prepass_output: array<Mesh>;
@group(0) @binding(8) var<storage, read_write> prepass_indirect_parameters:
    array<IndirectParameters>;
@group(0) @binding(9) var<storage, read_write> visibility: array<atomic<u32>>;
@group(0) @binding(10) var depth_pyramid: texture_2d<f32>;

#endif  // OCCLUSION_CULLING

// Returns true if the view frustum intersects an oriented bounding box (OBB).
//
// `aabb_center.w` should be 1.0.
fn view_frustum_intersects_obb(
    world_from_local: mat4x4<f32>,
    aabb_center: vec4<f32>,
    aabb_half_extents: vec3<f32>,
) -> bool {

    for (var i = 0; i < 5; i += 1) {
        // Calculate relative radius of the sphere associated with this plane.
        let plane_normal = view.frustum[i];
        let relative_radius = dot(
            abs(
                vec3(
                    dot(plane_normal, world_from_local[0]),
                    dot(plane_normal, world_from_local[1]),
                    dot(plane_normal, world_from_local[2]),
                )
            ),
            aabb_half_extents
        );

        // Check the frustum plane.
        if (!maths::sphere_intersects_plane_half_space(
                plane_normal, aabb_center, relative_radius)) {
            return false;
        }
    }

    return true;
}

#ifdef OCCLUSION_CULLING

fn object_is_unoccluded(
    world_from_local: mat4x4<f32>,
    aabb_center: vec4<f32>,
    aabb_half_extents: vec3<f32>,
    output_depth: ptr<function, vec2<u32>>,
) -> bool {
    var aabb_min = vec3(0.0);
    var aabb_max = vec3(0.0);
    for (var i = 0; i < 6; i++) {
        let signs = vec3(
            select(-1.0, 1.0, (i & 1) != 0),
            select(-1.0, 1.0, (i & 2) != 0),
            select(-1.0, 1.0, (i & 4) != 0),
        );
        var local_pos = aabb_center.xyz + signs * aabb_half_extents;
        let clip_pos = view.clip_from_world * (world_from_local * vec4(local_pos, 1.0));
        let ndc_pos = clip_pos.xyz / clip_pos.w;
        let uv_pos = vec3(view_transformations::ndc_to_uv(ndc_pos.xy), ndc_pos.z);
        if (i == 0) {
            aabb_min = uv_pos;
            aabb_max = uv_pos;
        } else {
            aabb_min = min(aabb_min, uv_pos);
            aabb_max = max(aabb_max, uv_pos);
        }
    }

    // Halve the view-space AABB size as the depth pyramid is half the view size
    let depth_pyramid_size_mip_0 = vec2<f32>(textureDimensions(depth_pyramid, 0)) * 0.5;
    let aabb_size = (aabb_max.xy - aabb_min.xy) * depth_pyramid_size_mip_0;
    let depth_level = max(0, i32(ceil(log2(max(aabb_size.x, aabb_size.y)))));
    let depth_mip_size = vec2<f32>(textureDimensions(depth_pyramid, depth_level));
    let aabb_top_left = vec2<u32>(aabb_min.xy * depth_mip_size);

    var depth_quad = vec4<f32>(0.0);
    depth_quad.x = textureLoad(depth_pyramid, aabb_top_left, depth_level).x;
    depth_quad.y = textureLoad(depth_pyramid, aabb_top_left + vec2(1u, 0u), depth_level).x;
    depth_quad.z = textureLoad(depth_pyramid, aabb_top_left + vec2(0u, 1u), depth_level).x;
    depth_quad.w = textureLoad(depth_pyramid, aabb_top_left + vec2(1u, 1u), depth_level).x;
    let occluder_depth = min(min(depth_quad.x, depth_quad.y), min(depth_quad.z, depth_quad.w));
    //let test = textureLoad(depth_pyramid, vec2(242, 131), 1);
    //*output_depth = vec2<u32>(u32(occluder_depth * 1000000.0), u32(aabb_max.z * 1000000.0));

    return aabb_max.z >= occluder_depth;
}

#endif  // OCCLUSION_CULLING

#endif  // FRUSTUM_CULLING

@compute
@workgroup_size(64)
fn main(@builtin(global_invocation_id) global_invocation_id: vec3<u32>) {
    // Figure out our instance index. If this thread doesn't correspond to any
    // index, bail.
    let instance_index = global_invocation_id.x;
    if (instance_index >= arrayLength(&work_items)) {
        return;
    }

    // Unpack indices.
    let input_index = work_items[instance_index].input_index;
    let output_index = work_items[instance_index].output_index;

    // If we're already known visible, then skip processing.
    // This ensures that, during the late prepass, we don't needlessly redraw
    // objects that were already enqueued in the early prepasss.
#ifdef OCCLUSION_CULLING
    let visible_index = instance_index / 32u;
    let visible_bitmask = 1u << (instance_index % 32u);
    if ((atomicLoad(&visibility[visible_index]) & visible_bitmask) != 0u) {
        return;
    }
#endif  // OCCLUSION_CULLING

    // Unpack transform.
    let world_from_local_affine_transpose = current_input[input_index].world_from_local;
    let world_from_local = maths::affine3_to_square(world_from_local_affine_transpose);

    // Cull if necessary.
    var output_depth: vec2<u32> = vec2(0u, 1u);
#ifdef FRUSTUM_CULLING
    let aabb_center = mesh_culling_data[input_index].aabb_center.xyz;
    let aabb_half_extents = mesh_culling_data[input_index].aabb_half_extents.xyz;

    // Do an OBB-based frustum cull.
    let model_center = world_from_local * vec4(aabb_center, 1.0);
    if (!view_frustum_intersects_obb(world_from_local, model_center, aabb_half_extents)) {
        return;
    }

#ifdef OCCLUSION_CULLING
    if (!object_is_unoccluded(world_from_local, model_center, aabb_half_extents, &output_depth)) {
        return;
    }

    // We now know that the object was visible. Go ahead and mark it as such.
    atomicOr(&visibility[visible_index], visible_bitmask);
#endif  // OCCLUSION_CULLING

#endif  // FRUSTUM_CULLING

    // Calculate inverse transpose.
    let local_from_world_transpose = transpose(maths::inverse_affine3(transpose(
        world_from_local_affine_transpose)));

    // Pack inverse transpose.
    let local_from_world_transpose_a = mat2x4<f32>(
        vec4<f32>(local_from_world_transpose[0].xyz, local_from_world_transpose[1].x),
        vec4<f32>(local_from_world_transpose[1].yz, local_from_world_transpose[2].xy));
    let local_from_world_transpose_b = local_from_world_transpose[2].z;

    // Look up the previous model matrix.
    let previous_input_index = current_input[input_index].previous_input_index;
    var previous_world_from_local: mat3x4<f32>;
    if (previous_input_index == 0xffffffff) {
        previous_world_from_local = world_from_local_affine_transpose;
    } else {
        previous_world_from_local = previous_input[previous_input_index].world_from_local;
    }

    var output_mesh: Mesh;
    output_mesh.world_from_local = world_from_local_affine_transpose;
    output_mesh.previous_world_from_local = previous_world_from_local;
    output_mesh.local_from_world_transpose_a = local_from_world_transpose_a;
    output_mesh.local_from_world_transpose_b = local_from_world_transpose_b;
    output_mesh.flags = current_input[input_index].flags;
    output_mesh.lightmap_uv_rect = current_input[input_index].lightmap_uv_rect;

    // Figure out the output index. In indirect mode, this involves bumping the
    // instance index in the indirect parameters structure. Otherwise, this
    // index was directly supplied to us.
#ifdef INDIRECT
    let main_pass_output_index = main_pass_indirect_parameters[output_index].instance_index +
        atomicAdd(&main_pass_indirect_parameters[output_index].instance_count, 1u);
#else   // INDIRECT
    let main_pass_output_index = output_index;
#endif  // INDIRECT

    // Write the output.
    main_pass_output[main_pass_output_index] = output_mesh;

#ifdef OCCLUSION_CULLING
    let prepass_output_index = prepass_indirect_parameters[output_index].instance_index +
        atomicAdd(&prepass_indirect_parameters[output_index].instance_count, 1u);
    prepass_output[prepass_output_index] = output_mesh;
#endif  // OCCLUSION_CULLING
}
