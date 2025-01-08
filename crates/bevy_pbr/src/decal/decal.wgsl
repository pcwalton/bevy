// Support code for clustered decal projectors.

#define_import_path bevy_pbr::decal

#import bevy_pbr::clustered_forward
#import bevy_pbr::clustered_forward::ClusterableObjectIndexRanges
#import bevy_pbr::mesh_view_bindings
#import bevy_render::maths

// TODO: make this an iterator
struct DecalQueryResult {
    texture_index: i32,
    uv: vec2<f32>,
}

struct DecalIterator {
    start_offset: i32,
    end_offset: i32,
    decal_index_offset: i32,
    world_position: vec3<f32>,
    texture_index: i32,
    uv: vec2<f32>,
}

#ifdef DECALS_ARE_USABLE

fn decal_iterator_new(
    world_position: vec3<f32>,
    clusterable_object_index_ranges: ptr<function, ClusterableObjectIndexRanges>
) -> DecalIterator {
    return DecalIterator(
        i32((*clusterable_object_index_ranges).first_decal_offset),
        i32((*clusterable_object_index_ranges).last_clusterable_object_index_offset),
        -1,
        world_position,
        -1,
        vec2(0.0),
    );
}

fn decal_iterator_next(iterator: ptr<function, DecalIterator>) -> bool {
    if ((*iterator).decal_index_offset == (*iterator).end_offset) {
        return false;
    }

    (*iterator).decal_index_offset += 1;

    while ((*iterator).decal_index_offset < (*iterator).end_offset) {
        let decal_index = i32(clustered_forward::get_clusterable_object_id(
            u32((*iterator).decal_index_offset)
        ));
        let decal_space_vector = (mesh_view_bindings::decals.decals[decal_index].local_from_world *
            vec4((*iterator).world_position, 1.0)).xyz;
        
        if (all(decal_space_vector >= vec3(-0.5)) && all(decal_space_vector <= vec3(0.5))) {
            (*iterator).texture_index =
                i32(mesh_view_bindings::decals.decals[decal_index].image_index);
            (*iterator).uv = decal_space_vector.xy * vec2(1.0, -1.0) + vec2(0.5);
            return true;
        }

        (*iterator).decal_index_offset += 1;
    }

    return false;
}

#endif  // DECALS_ARE_USABLE

// Modifies the base color at the given position to account for decals.
//
// Returns the new base color with decals taken into account. If no such base color
fn apply_decal_base_color(
    world_position: vec3<f32>,
    frag_coord: vec2<f32>,
    initial_base_color: vec4<f32>,
) -> vec4<f32> {
    var base_color = initial_base_color;

#ifdef DECALS_ARE_USABLE
    let view_z = dot(vec4<f32>(
        mesh_view_bindings::view.view_from_world[0].z,
        mesh_view_bindings::view.view_from_world[1].z,
        mesh_view_bindings::view.view_from_world[2].z,
        mesh_view_bindings::view.view_from_world[3].z
    ), vec4(world_position, 1.0));
    let is_orthographic = mesh_view_bindings::view.clip_from_view[3].w == 1.0;

    let cluster_index =
        clustered_forward::fragment_cluster_index(frag_coord, view_z, is_orthographic);
    var clusterable_object_index_ranges =
        clustered_forward::unpack_clusterable_object_index_ranges(cluster_index);

    var iterator = decal_iterator_new(world_position, &clusterable_object_index_ranges);
    while (decal_iterator_next(&iterator)) {
        let decal_base_color = textureSampleLevel(
            mesh_view_bindings::decal_textures[iterator.texture_index],
            mesh_view_bindings::decal_sampler,
            iterator.uv,
            0.0
        );
        base_color = vec4(
            mix(base_color.rgb, decal_base_color.rgb, decal_base_color.a),
            base_color.a + decal_base_color.a
        );
    }
#endif  // DECALS_ARE_USABLE

    return base_color;
}
