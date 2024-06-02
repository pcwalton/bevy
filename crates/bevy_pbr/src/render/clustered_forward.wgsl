#define_import_path bevy_pbr::clustered_forward

#import bevy_pbr::{
    mesh_view_bindings as bindings,
    utils::rand_f,
}

#import bevy_render::{
   color_operations::hsv_to_rgb,
   maths::PI_2,
}

struct ClusterableIndices {
    first_point_light_index: u32,
    first_spot_light_index: u32,
    last_clusterable_index: u32,
}

// NOTE: Keep in sync with bevy_pbr/src/light.rs
fn view_z_to_z_slice(view_z: f32, is_orthographic: bool) -> u32 {
    var z_slice: u32 = 0u;
    if is_orthographic {
        // NOTE: view_z is correct in the orthographic case
        z_slice = u32(floor((view_z - bindings::lights.cluster_factors.z) * bindings::lights.cluster_factors.w));
    } else {
        // NOTE: had to use -view_z to make it positive else log(negative) is nan
        z_slice = u32(log(-view_z) * bindings::lights.cluster_factors.z - bindings::lights.cluster_factors.w + 1.0);
    }
    // NOTE: We use min as we may limit the far z plane used for clustering to be closer than
    // the furthest thing being drawn. This means that we need to limit to the maximum cluster.
    return min(z_slice, bindings::lights.cluster_dimensions.z - 1u);
}

fn fragment_cluster_index(frag_coord: vec2<f32>, view_z: f32, is_orthographic: bool) -> u32 {
    let xy = vec2<u32>(floor((frag_coord - bindings::view.viewport.xy) * bindings::lights.cluster_factors.xy));
    let z_slice = view_z_to_z_slice(view_z, is_orthographic);
    // NOTE: Restricting cluster index to avoid undefined behavior when accessing uniform buffer
    // arrays based on the cluster index.
    return min(
        (xy.y * bindings::lights.cluster_dimensions.x + xy.x) * bindings::lights.cluster_dimensions.z + z_slice,
        bindings::lights.cluster_dimensions.w - 1u
    );
}

// this must match CLUSTER_COUNT_SIZE in light.rs
const CLUSTER_COUNT_SIZE = 9u;
fn unpack_clusterable_indices(cluster_index: u32) -> ClusterableIndices {
#if AVAILABLE_STORAGE_BUFFER_BINDINGS >= 3
    let packed = bindings::cluster_offsets_and_counts.data[cluster_index];
    let first_point_light_index = packed.x;
    let first_spot_light_index = first_point_light_index + (packed.y & 0xffffu);
    let last_clusterable_index = first_point_light_index + (packed.y >> 16u);
    return ClusterableIndices(
        first_point_light_index,
        first_spot_light_index,
        last_clusterable_index,
    );
#else
    let offset_and_counts = bindings::cluster_offsets_and_counts.data[cluster_index >> 2u][cluster_index & ((1u << 2u) - 1u)];
    //  [ 31     ..     18 | 17        ..         9 | 8         ..         0 ]
    //  [      offset      | first spot light index | last clusterable index ]
    let first_point_light_index = (offset_and_counts >> (CLUSTER_COUNT_SIZE * 2u)) &
        ((1u << (32u - (CLUSTER_COUNT_SIZE * 2u))) - 1u);
    let first_spot_light_index = first_point_light_index +
        ((offset_and_counts >> CLUSTER_COUNT_SIZE) & ((1u << CLUSTER_COUNT_SIZE) - 1u));
    let last_clusterable_index = first_point_light_index +
        (offset_and_counts & ((1u << CLUSTER_COUNT_SIZE) - 1u));
    return ClusterableIndices(
        first_point_light_index,
        first_spot_light_index,
        last_clusterable_index,
    );
#endif
}

fn get_light_id(index: u32) -> u32 {
#if AVAILABLE_STORAGE_BUFFER_BINDINGS >= 3
    return bindings::cluster_index_lists.data[index];
#else
    // The index is correct but in cluster_light_index_lists we pack 4 u8s into a u32
    // This means the index into cluster_light_index_lists is index / 4
    let indices = bindings::cluster_index_lists.data[index >> 4u][(index >> 2u) & ((1u << 2u) - 1u)];
    // And index % 4 gives the sub-index of the u8 within the u32 so we shift by 8 * sub-index
    return (indices >> (8u * (index & ((1u << 2u) - 1u)))) & ((1u << 8u) - 1u);
#endif
}

fn cluster_debug_visualization(
    input_color: vec4<f32>,
    view_z: f32,
    is_orthographic: bool,
    clusterable_indices: ptr<function, ClusterableIndices>,
    cluster_index: u32,
) -> vec4<f32> {
    var output_color = input_color;

    // Cluster allocation debug (using 'over' alpha blending)
#ifdef CLUSTERED_FORWARD_DEBUG_Z_SLICES
    // NOTE: This debug mode visualises the z-slices
    let cluster_overlay_alpha = 0.1;
    var z_slice: u32 = view_z_to_z_slice(view_z, is_orthographic);
    // A hack to make the colors alternate a bit more
    if (z_slice & 1u) == 1u {
        z_slice = z_slice + bindings::lights.cluster_dimensions.z / 2u;
    }
    let slice_color = hsv_to_rgb(
        f32(z_slice) / f32(bindings::lights.cluster_dimensions.z + 1u) * PI_2,
        1.0,
        0.5
    );
    output_color = vec4<f32>(
        (1.0 - cluster_overlay_alpha) * output_color.rgb + cluster_overlay_alpha * slice_color,
        output_color.a
    );
#endif // CLUSTERED_FORWARD_DEBUG_Z_SLICES

#ifdef CLUSTERED_FORWARD_DEBUG_CLUSTER_LIGHT_COMPLEXITY
    // NOTE: This debug mode visualises the number of lights within the cluster that contains
    // the fragment. It shows a sort of lighting complexity measure.
    let cluster_overlay_alpha = 0.1;
    let max_light_complexity_per_cluster = 64.0;
    let light_complexity = f32(clusterable_indices.last_clusterable_index -
        clusterable_indices.first_point_light_index);
    output_color.r = (1.0 - cluster_overlay_alpha) * output_color.r +
        cluster_overlay_alpha * smoothStep(
            0.0,
            max_light_complexity_per_cluster,
            light_complexity,
        );
    output_color.g = (1.0 - cluster_overlay_alpha) * output_color.g +
        cluster_overlay_alpha * (1.0 - smoothStep(
            0.0,
            max_light_complexity_per_cluster,
            light_complexity));
#endif // CLUSTERED_FORWARD_DEBUG_CLUSTER_LIGHT_COMPLEXITY

#ifdef CLUSTERED_FORWARD_DEBUG_CLUSTER_COHERENCY
    // NOTE: Visualizes the cluster to which the fragment belongs
    let cluster_overlay_alpha = 0.1;
    var rng = cluster_index;
    let cluster_color = hsv_to_rgb(rand_f(&rng) * PI_2, 1.0, 0.5);
    output_color = vec4<f32>(
        (1.0 - cluster_overlay_alpha) * output_color.rgb + cluster_overlay_alpha * cluster_color,
        output_color.a
    );
#endif // CLUSTERED_FORWARD_DEBUG_CLUSTER_COHERENCY

    return output_color;
}
