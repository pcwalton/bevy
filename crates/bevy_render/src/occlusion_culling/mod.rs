//! GPU occlusion culling.

use bevy_app::{App, Plugin};
use bevy_asset::{load_internal_asset, Handle};
use bevy_ecs::{component::Component, prelude::ReflectComponent};
use bevy_math::UVec3;
use bevy_reflect::{prelude::ReflectDefault, Reflect};

use crate::{
    extract_component::ExtractComponent,
    render_resource::{Shader, ShaderType},
};

/// The handle to the `mesh_preprocess_types.wgsl` compute shader.
pub const MESH_PREPROCESS_TYPES_SHADER_HANDLE: Handle<Shader> =
    Handle::weak_from_u128(2720440370122465935);

pub struct OcclusionCullingPlugin;

impl Plugin for OcclusionCullingPlugin {
    fn build(&self, app: &mut App) {
        load_internal_asset!(
            app,
            MESH_PREPROCESS_TYPES_SHADER_HANDLE,
            "mesh_preprocess_types.wgsl",
            Shader::from_wgsl
        );
    }
}

#[derive(Component, ExtractComponent, Clone, Copy, Default, Reflect)]
#[reflect(Component, Default)]
pub struct OcclusionCulling;

#[derive(Clone, Copy, Default, ShaderType)]
pub struct OcclusionCullingIndirectCounts {
    pub workgroup_counts: UVec3,
    pub invocation_count: u32,
}
