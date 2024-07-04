#import bevy_pbr::{
    pbr_prepass_functions,
    pbr_bindings,
    pbr_bindings::material,
    pbr_types,
    pbr_functions,
    pbr_functions::SampleBias,
    prepass_io,
    mesh_bindings::mesh,
    mesh_view_bindings::view,
}
#import bevy_render::maths

#ifdef MESHLET_MESH_MATERIAL_PASS
#import bevy_pbr::meshlet_visibility_buffer_resolve::resolve_vertex_output
#endif

#ifdef PREPASS_FRAGMENT
@fragment
fn fragment(
#ifdef MESHLET_MESH_MATERIAL_PASS
    @builtin(position) frag_coord: vec4<f32>,
#else
    in: prepass_io::VertexOutput,
    @builtin(front_facing) is_front: bool,
#endif
) -> prepass_io::FragmentOutput {
#ifdef MESHLET_MESH_MATERIAL_PASS
    let in = resolve_vertex_output(frag_coord);
    let is_front = true;
#else
    pbr_prepass_functions::prepass_alpha_discard(in);
#endif

    let material_id = mesh[in.instance_index].material_id;
#ifdef BINDLESS_TEXTURES
    var bindless_indices = pbr_bindings::bindless_indices[material_id];
#endif  // BINDLESS_TEXTURES

    var out: prepass_io::FragmentOutput;

#ifdef DEPTH_CLAMP_ORTHO
    out.frag_depth = in.clip_position_unclamped.z;
#endif // DEPTH_CLAMP_ORTHO

#ifdef NORMAL_PREPASS
    // NOTE: Unlit bit not set means == 0 is true, so the true case is if lit
    if (pbr_bindings::materials[material_id].flags &
            pbr_types::STANDARD_MATERIAL_FLAGS_UNLIT_BIT) == 0u {
        let double_sided = (pbr_bindings::materials[material_id].flags &
            pbr_types::STANDARD_MATERIAL_FLAGS_DOUBLE_SIDED_BIT) != 0u;

        let world_normal = pbr_functions::prepare_world_normal(
            in.world_normal,
            double_sided,
            is_front,
        );

        var normal = world_normal;

#ifdef VERTEX_UVS
#ifdef VERTEX_TANGENTS
#ifdef STANDARD_MATERIAL_NORMAL_MAP

        if (true
#ifdef BINDLESS_TEXTURES
                && bindless_indices.normal_map_texture < 0xffffffffu
                && bindless_indices.normal_map_sampler < 0xffffffffu
#endif  // BINDLESS_TEXTURES
        ) {

            let uv_transform_mat4 = pbr_bindings::materials[material_id].uv_transform;
            let uv_transform = maths::mat4x4_to_mat3x3(uv_transform_mat4);

#ifdef STANDARD_MATERIAL_NORMAL_MAP_UV_B
            let uv = (uv_transform * vec3(in.uv_b, 1.0)).xy;
#else
            let uv = (uv_transform * vec3(in.uv, 1.0)).xy;
#endif

            // Fill in the sample bias so we can sample from textures.
            var bias: SampleBias;
#ifdef MESHLET_MESH_MATERIAL_PASS
            bias.ddx_uv = in.ddx_uv;
            bias.ddy_uv = in.ddy_uv;
#else   // MESHLET_MESH_MATERIAL_PASS
            bias.mip_bias = view.mip_bias;
#endif  // MESHLET_MESH_MATERIAL_PASS

            let Nt =
#ifdef MESHLET_MESH_MATERIAL_PASS
                textureSampleGrad(
#else   // MESHLET_MESH_MATERIAL_PASS
                textureSampleBias(
#endif  // MESHLET_MESH_MATERIAL_PASS
#ifdef BINDLESS_TEXTURES
                    pbr_bindings::normal_map_texture[bindless_indices.normal_map_texture],
                    pbr_bindings::normal_map_sampler[bindless_indices.normal_map_sampler],
#else   // BINDLESS_TEXTURES
                    pbr_bindings::normal_map_texture,
                    pbr_bindings::normal_map_sampler,
#endif  // BINDLESS_TEXTURES
                    uv,
#ifdef MESHLET_MESH_MATERIAL_PASS
                    bias.ddx_uv,
                    bias.ddy_uv,
#else   // MESHLET_MESH_MATERIAL_PASS
                    bias.mip_bias,
#endif  // MESHLET_MESH_MATERIAL_PASS
                ).rgb;

            let TBN = pbr_functions::calculate_tbn_mikktspace(normal, in.world_tangent);

            normal = pbr_functions::apply_normal_mapping(
                pbr_bindings::materials[material_id].flags,
                TBN,
                double_sided,
                is_front,
                Nt,
            );

        }

#endif  // STANDARD_MATERIAL_NORMAL_MAP
#endif  // VERTEX_TANGENTS
#endif  // VERTEX_UVS

        out.normal = vec4(normal * 0.5 + vec3(0.5), 1.0);
    } else {
        out.normal = vec4(in.world_normal * 0.5 + vec3(0.5), 1.0);
    }
#endif // NORMAL_PREPASS

#ifdef MOTION_VECTOR_PREPASS
#ifdef MESHLET_MESH_MATERIAL_PASS
    out.motion_vector = in.motion_vector;
#else
    out.motion_vector = pbr_prepass_functions::calculate_motion_vector(in.world_position, in.previous_world_position);
#endif
#endif

    return out;
}
#else
@fragment
fn fragment(in: prepass_io::VertexOutput) {
    pbr_prepass_functions::prepass_alpha_discard(in);
}
#endif // PREPASS_FRAGMENT
