// GPU mesh uniform building.
//
// This is a compute shader that expands each `MeshInputUniform` out to a full
// `MeshUniform` for each view before rendering. (Thus `MeshInputUniform`
// and `MeshUniform` are in a 1:N relationship.) It runs in parallel for all
// meshes for all views. As part of this process, the shader gathers each
// mesh's transform on the previous frame and writes it into the `MeshUniform`
// so that TAA works.

#import bevy_pbr::mesh_types::{Mesh, MESH_FLAGS_NO_FRUSTUM_CULLING_BIT}
#import bevy_pbr::mesh_preprocess_types::{IndirectParametersMetadata, MeshInput}
#import bevy_pbr::occlusion_culling
#import bevy_pbr::view_transformations::ndc_to_uv
#import bevy_render::maths
#import bevy_render::view::View

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
    // The index of the `Mesh` in `output` that we write to.
    output_index: u32,
    // The index of the `IndirectParameters` in `indirect_parameters` that we
    // write to.
    indirect_parameters_index: u32,
}

struct ViewVisibility {
    visibility: u32,
    debug_max_depth_view: f32,
    debug_max_depth_ndc: f32,
    already_drawn: u32,
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
@group(0) @binding(3) var<storage, read_write> output: array<Mesh>;

#ifdef INDIRECT
// The array of indirect parameters for drawcalls.
@group(0) @binding(4) var<storage, read_write> indirect_parameters_metadata:
    array<IndirectParametersMetadata>;
#endif

#ifdef FRUSTUM_CULLING
// Data needed to cull the meshes.
//
// At the moment, this consists only of AABBs.
@group(0) @binding(5) var<storage> mesh_culling_data: array<MeshCullingData>;

// The view data, including the view matrix.
@group(0) @binding(6) var<uniform> view: View;
#endif  // FRUSTUM_CULLING

#ifdef OCCLUSION_CULLING
// TODO: Make this a bitfield? Would have to use atomics then.
// Meshlets makes this a bitfield.
@group(0) @binding(7) var<storage, read_write> view_visibility: array<ViewVisibility>;

#ifdef EARLY
@group(0) @binding(8) var<storage, read> previous_frame_view_visibility: array<ViewVisibility>;
#else   // EARLY
@group(0) @binding(8) var depth_pyramid: texture_2d<f32>;
#endif  // EARLY
#endif  // OCCLUSION_CULLING

#ifdef FRUSTUM_CULLING
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
#endif

@compute
@workgroup_size(64)
fn main(@builtin(global_invocation_id) global_invocation_id: vec3<u32>) {
    // Figure out our instance index. If this thread doesn't correspond to any
    // index, bail.
    let instance_index = global_invocation_id.x;
    if (instance_index >= arrayLength(&work_items)) {
        return;
    }

    // Unpack the work item.
    let input_index = work_items[instance_index].input_index;
    let output_index = work_items[instance_index].output_index;
    let indirect_parameters_index = work_items[instance_index].indirect_parameters_index;

#ifdef OCCLUSION_CULLING
#ifdef EARLY
    // If this is phase 1 of the occlusion culling pass, only draw the object if
    // it was visible the previous frame.
    if (previous_frame_view_visibility[input_index].visibility != 2u) {
        return;
    }
#endif  // EARLY
#endif  // OCCLUSION_CULLING

    // Unpack the input matrix.
    let world_from_local_affine_transpose = current_input[input_index].world_from_local;
    let world_from_local = maths::affine3_to_square(world_from_local_affine_transpose);

    // Frustum cull if necessary.
#ifdef FRUSTUM_CULLING
    if ((current_input[input_index].flags & MESH_FLAGS_NO_FRUSTUM_CULLING_BIT) == 0u) {
        let aabb_center = mesh_culling_data[input_index].aabb_center.xyz;
        let aabb_half_extents = mesh_culling_data[input_index].aabb_half_extents.xyz;

        // Do an OBB-based frustum cull.
        let model_center = world_from_local * vec4(aabb_center, 1.0);
        if (!view_frustum_intersects_obb(world_from_local, model_center, aabb_half_extents)) {
            return;
        }
    }
#endif

    // Occlusion cull if necessary.
#ifdef OCCLUSION_CULLING
#ifdef EARLY
    view_visibility[input_index].visibility = 1u;
#else   // EARLY

    let aabb_center = mesh_culling_data[input_index].aabb_center.xyz;
    let aabb_half_extents = mesh_culling_data[input_index].aabb_half_extents.xyz;

    var aabb = vec4(0.0);
    var max_depth_view = 0.0;

    for (var i = 0u; i < 8u; i += 1u) {
        let local_pos = aabb_center + select(
            vec3(-1.0),
            vec3(1.0),
            vec3((i & 1) != 0, (i & 2) != 0, (i & 4) != 0)
        ) * aabb_half_extents;

        let world_pos = (world_from_local * vec4(local_pos, 1.0)).xyz;
        let view_pos = position_world_to_view(world_pos);
        let ndc_pos = position_world_to_ndc(world_pos);
        let uv_pos = ndc_to_uv(ndc_pos.xy);

        if (i == 0u) {
            aabb = vec4(uv_pos, uv_pos);
            max_depth_view = view_pos.z;
        } else {
            aabb = vec4(min(aabb.xy, uv_pos), max(aabb.zw, uv_pos));
            max_depth_view = max(max_depth_view, view_pos.z);
        }
    }

    // Clip to near plane to avoid NDC depth becoming negative.
    max_depth_view = min(-view.clip_from_view[3][2], max_depth_view);

    let aabb_pixel_size = occlusion_culling::get_aabb_size_in_pixels(aabb, depth_pyramid);
    let occluder_depth_ndc =
        occlusion_culling::get_occluder_depth(aabb, aabb_pixel_size, depth_pyramid);

    let max_depth_ndc = view_z_to_depth_ndc(max_depth_view);

    view_visibility[input_index].debug_max_depth_view = max_depth_view;
    view_visibility[input_index].debug_max_depth_ndc = max_depth_ndc;

    if (max_depth_ndc < occluder_depth_ndc) {
        return;
    }

    let early_view_visibility = view_visibility[input_index].visibility;
    view_visibility[input_index].visibility = 2u;

    // Now if this is phase 2 of the occlusion culling pass, and we've already
    // drawn the object, don't draw it again.
    view_visibility[input_index].already_drawn = select(0u, 1u, early_view_visibility != 0u);
    if (early_view_visibility != 0u) {
        return;
    }
#endif  // EARLY
#endif  // OCCLUSION_CULLING

    // Calculate inverse transpose.
    let local_from_world_transpose = transpose(maths::inverse_affine3(transpose(
        world_from_local_affine_transpose)));

    // Pack inverse transpose.
    let local_from_world_transpose_a = mat2x4<f32>(
        vec4<f32>(local_from_world_transpose[0].xyz, local_from_world_transpose[1].x),
        vec4<f32>(local_from_world_transpose[1].yz, local_from_world_transpose[2].xy));
    let local_from_world_transpose_b = local_from_world_transpose[2].z;

    // Look up the previous model matrix.
    let previous_world_from_local = previous_input[input_index].world_from_local;

    // Figure out the output index. In indirect mode, this involves bumping the
    // instance index in the indirect parameters structure. Otherwise, this
    // index was directly supplied to us.
#ifdef INDIRECT
#ifdef LATE
    let batch_output_index =
        atomicLoad(&indirect_parameters_metadata[indirect_parameters_index].early_instance_count) +
        atomicAdd(&indirect_parameters_metadata[indirect_parameters_index].late_instance_count, 1u);
#else   // LATE
    let batch_output_index = atomicAdd(
        &indirect_parameters_metadata[indirect_parameters_index].early_instance_count,
        1u
    );
#endif  // LATE

    let mesh_output_index =
        indirect_parameters_metadata[indirect_parameters_index].base_output_index +
        batch_output_index;

#else   // INDIRECT
    let mesh_output_index = output_index;
#endif  // INDIRECT

    // Write the output.
    output[mesh_output_index].world_from_local = world_from_local_affine_transpose;
    output[mesh_output_index].previous_world_from_local = previous_world_from_local;
    output[mesh_output_index].local_from_world_transpose_a = local_from_world_transpose_a;
    output[mesh_output_index].local_from_world_transpose_b = local_from_world_transpose_b;
    output[mesh_output_index].flags = current_input[input_index].flags;
    output[mesh_output_index].lightmap_uv_rect = current_input[input_index].lightmap_uv_rect;
    output[mesh_output_index].first_vertex_index = current_input[input_index].first_vertex_index;
    output[mesh_output_index].current_skin_index = current_input[input_index].current_skin_index;
    output[mesh_output_index].previous_skin_index = current_input[input_index].previous_skin_index;
    output[mesh_output_index].material_and_lightmap_bind_group_slot =
        current_input[input_index].material_and_lightmap_bind_group_slot;
}

#ifdef OCCLUSION_CULLING
// https://zeux.io/2023/01/12/approximate-projected-bounds
fn project_view_space_sphere_to_screen_space_aabb(cp: vec3<f32>, r: f32) -> vec4<f32> {
    let inv_width = view.clip_from_view[0][0] * 0.5;
    let inv_height = view.clip_from_view[1][1] * 0.5;
    if view.clip_from_view[3][3] == 1.0 {
        // Orthographic
        let min_x = cp.x - r;
        let max_x = cp.x + r;

        let min_y = cp.y - r;
        let max_y = cp.y + r;

        return vec4(min_x * inv_width, 1.0 - max_y * inv_height, max_x * inv_width, 1.0 - min_y * inv_height);
    } else {
        // Perspective
        let c = vec3(cp.xy, -cp.z);
        let cr = c * r;
        let czr2 = c.z * c.z - r * r;

        let vx = sqrt(c.x * c.x + czr2);
        let min_x = (vx * c.x - cr.z) / (vx * c.z + cr.x);
        let max_x = (vx * c.x + cr.z) / (vx * c.z - cr.x);

        let vy = sqrt(c.y * c.y + czr2);
        let min_y = (vy * c.y - cr.z) / (vy * c.z + cr.y);
        let max_y = (vy * c.y + cr.z) / (vy * c.z - cr.y);

        return vec4(min_x * inv_width, -max_y * inv_height, max_x * inv_width, -min_y * inv_height) + vec4(0.5);
    }
}

/// Convert a world space position to view space
fn position_world_to_view(world_pos: vec3<f32>) -> vec3<f32> {
    let view_pos = view.view_from_world * vec4(world_pos, 1.0);
    return view_pos.xyz;
}

/// Convert a world space position to ndc space
fn position_world_to_ndc(world_pos: vec3<f32>) -> vec3<f32> {
    let ndc_pos = view.clip_from_world * vec4(world_pos, 1.0);
    return ndc_pos.xyz / ndc_pos.w;
}

/// Convert linear view z to ndc.
fn view_z_to_depth_ndc(view_z: f32) -> f32 {
    if (view.clip_from_view[3][3] != 1.0) {
        // Perspective
        return -view.clip_from_view[3][2] / view_z;
    }

    // Orthographic
    return view.clip_from_view[3][2] + view_z * view.clip_from_view[2][2];
}
#endif  // OCCLUSION_CULLING