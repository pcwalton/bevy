pub mod copy_lighting_id;
pub mod node;

use std::ops::Range;

use bevy_asset::AssetId;
use bevy_ecs::prelude::*;
use bevy_render::{
    mesh::Mesh,
    render_phase::{
        BinnedPhaseItem, CachedRenderPipelinePhaseItem, DrawFunctionId, PhaseItem, SortedPhaseItem,
    },
    render_resource::{CachedRenderPipelineId, TextureFormat},
};
use nonmax::NonMaxU32;

use crate::prepass::Opaque3dPrepassBinKey;

pub const DEFERRED_PREPASS_FORMAT: TextureFormat = TextureFormat::Rgba32Uint;
pub const DEFERRED_LIGHTING_PASS_ID_FORMAT: TextureFormat = TextureFormat::R8Uint;
pub const DEFERRED_LIGHTING_PASS_ID_DEPTH_FORMAT: TextureFormat = TextureFormat::Depth16Unorm;

/// Opaque phase of the 3D Deferred pass.
///
/// Sorted by pipeline, then by mesh to improve batching.
///
/// Used to render all 3D meshes with materials that have no transparency.
#[derive(PartialEq, Eq, Hash)]
pub struct Opaque3dDeferred {
    pub key: Opaque3dPrepassBinKey,
    pub representative_entity: Entity,
    pub batch_range: Range<u32>,
    pub dynamic_offset: Option<NonMaxU32>,
}

impl PhaseItem for Opaque3dDeferred {
    #[inline]
    fn entity(&self) -> Entity {
        self.representative_entity
    }

    #[inline]
    fn draw_function(&self) -> DrawFunctionId {
        self.key.draw_function
    }

    #[inline]
    fn batch_range(&self) -> &Range<u32> {
        &self.batch_range
    }

    #[inline]
    fn batch_range_mut(&mut self) -> &mut Range<u32> {
        &mut self.batch_range
    }

    #[inline]
    fn dynamic_offset(&self) -> Option<NonMaxU32> {
        self.dynamic_offset
    }

    #[inline]
    fn dynamic_offset_mut(&mut self) -> &mut Option<NonMaxU32> {
        &mut self.dynamic_offset
    }
}

impl BinnedPhaseItem for Opaque3dDeferred {
    type BinKey = Opaque3dPrepassBinKey;

    fn new(
        key: Self::BinKey,
        representative_entity: Entity,
        batch_range: Range<u32>,
        dynamic_offset: Option<NonMaxU32>,
    ) -> Self {
        Opaque3dDeferred {
            key,
            representative_entity,
            batch_range,
            dynamic_offset,
        }
    }
}

impl CachedRenderPipelinePhaseItem for Opaque3dDeferred {
    #[inline]
    fn cached_pipeline(&self) -> CachedRenderPipelineId {
        self.key.pipeline
    }
}

/// Alpha mask phase of the 3D Deferred pass.
///
/// Sorted by pipeline, then by mesh to improve batching.
///
/// Used to render all meshes with a material with an alpha mask.
pub struct AlphaMask3dDeferred {
    pub asset_id: AssetId<Mesh>,
    pub entity: Entity,
    pub pipeline_id: CachedRenderPipelineId,
    pub draw_function: DrawFunctionId,
    pub batch_range: Range<u32>,
    pub dynamic_offset: Option<NonMaxU32>,
}

impl PhaseItem for AlphaMask3dDeferred {
    #[inline]
    fn entity(&self) -> Entity {
        self.entity
    }

    #[inline]
    fn draw_function(&self) -> DrawFunctionId {
        self.draw_function
    }

    #[inline]
    fn batch_range(&self) -> &Range<u32> {
        &self.batch_range
    }

    #[inline]
    fn batch_range_mut(&mut self) -> &mut Range<u32> {
        &mut self.batch_range
    }

    #[inline]
    fn dynamic_offset(&self) -> Option<NonMaxU32> {
        self.dynamic_offset
    }

    #[inline]
    fn dynamic_offset_mut(&mut self) -> &mut Option<NonMaxU32> {
        &mut self.dynamic_offset
    }
}

impl SortedPhaseItem for AlphaMask3dDeferred {
    type SortKey = (usize, AssetId<Mesh>);

    #[inline]
    fn sort_key(&self) -> Self::SortKey {
        // Sort by pipeline, then by mesh to massively decrease drawcall counts in real scenes.
        (self.pipeline_id.id(), self.asset_id)
    }

    #[inline]
    fn sort(items: &mut [Self]) {
        items.sort_unstable_by_key(Self::sort_key);
    }
}

impl CachedRenderPipelinePhaseItem for AlphaMask3dDeferred {
    #[inline]
    fn cached_pipeline(&self) -> CachedRenderPipelineId {
        self.pipeline_id
    }
}
