use core::ops::Range;
use std::marker::PhantomData;

use super::{UiBatch, UiMeta, UiTexturedBindGroups, UiViewTarget};

use crate::{ExtractedUiNode, UiCameraView, UiInstances, UiMetaDrawArgs, UiRenderObject};
use bevy_ecs::{
    entity::EntityHash,
    prelude::*,
    system::{lifetimeless::*, SystemParamItem},
};
use bevy_math::FloatOrd;
use bevy_render::{
    batching::gpu_preprocessing::IndirectParametersIndexed,
    camera::ExtractedCamera,
    diagnostic::RecordDiagnostics,
    render_phase::*,
    render_resource::{CachedRenderPipelineId, RenderPassDescriptor},
    renderer::{RenderContext, ViewQuery},
    sync_world::MainEntity,
    view::*,
};
use indexmap::IndexMap;
use tracing::error;

pub fn ui_pass(
    world: &World,
    view: ViewQuery<&UiCameraView>,
    ui_view_query: Query<(&ExtractedView, &UiViewTarget)>,
    ui_view_target_query: Query<(&ViewTarget, &ExtractedCamera)>,
    transparent_render_phases: Res<ViewSortedRenderPhases<TransparentUi>>,
    mut ctx: RenderContext,
) {
    let ui_camera_view = view.into_inner();
    let ui_view_entity = ui_camera_view.0;

    let Ok((extracted_view, ui_view_target)) = ui_view_query.get(ui_view_entity) else {
        return;
    };

    let Ok((target, camera)) = ui_view_target_query.get(ui_view_target.0) else {
        return;
    };

    let Some(transparent_phase) =
        transparent_render_phases.get(&extracted_view.retained_view_entity)
    else {
        return;
    };

    if transparent_phase.items.is_empty() {
        return;
    }

    let diagnostics = ctx.diagnostic_recorder();
    let diagnostics = diagnostics.as_deref();

    let mut render_pass = ctx.begin_tracked_render_pass(RenderPassDescriptor {
        label: Some("ui"),
        color_attachments: &[Some(target.get_unsampled_color_attachment())],
        depth_stencil_attachment: None,
        timestamp_writes: None,
        occlusion_query_set: None,
        multiview_mask: None,
    });
    let pass_span = diagnostics.pass_span(&mut render_pass, "ui");

    if let Some(viewport) = camera.viewport.as_ref() {
        render_pass.set_camera_viewport(viewport);
    }

    if let Err(err) = transparent_phase.render(&mut render_pass, world, ui_view_entity) {
        error!("Error encountered while rendering the ui phase {err:?}");
    }

    pass_span.end(&mut render_pass);
}

#[derive(Debug)]
pub struct TransparentUi {
    pub sort_key: FloatOrd,
    pub entity: (Entity, MainEntity),
    pub pipeline: CachedRenderPipelineId,
    pub draw_function: DrawFunctionId,
    pub batch_range: Range<u32>,
    pub extra_index: PhaseItemExtraIndex,
    pub indexed: bool,
}

impl PhaseItem for TransparentUi {
    #[inline]
    fn entity(&self) -> Entity {
        self.entity.0
    }

    fn main_entity(&self) -> MainEntity {
        self.entity.1
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
    fn extra_index(&self) -> PhaseItemExtraIndex {
        self.extra_index.clone()
    }

    #[inline]
    fn batch_range_and_extra_index_mut(&mut self) -> (&mut Range<u32>, &mut PhaseItemExtraIndex) {
        (&mut self.batch_range, &mut self.extra_index)
    }
}

impl SortedPhaseItem for TransparentUi {
    type SortKey = FloatOrd;

    #[inline]
    fn sort_key(&self) -> Self::SortKey {
        self.sort_key
    }

    #[inline]
    fn sort(items: &mut IndexMap<(Entity, MainEntity), TransparentUi, EntityHash>) {
        items.sort_by_key(|_, value| value.sort_key());
    }

    fn recalculate_sort_keys(
        _: &mut IndexMap<(Entity, MainEntity), Self, EntityHash>,
        _: &ExtractedView,
    ) {
        // Sort keys are precalculated for UI phase items.
    }

    #[inline]
    fn indexed(&self) -> bool {
        self.indexed
    }
}

impl CachedRenderPipelinePhaseItem for TransparentUi {
    #[inline]
    fn cached_pipeline(&self) -> CachedRenderPipelineId {
        self.pipeline
    }
}

/// The render command used to draw a normal UI element (node or glyph).
pub type DrawUi = (
    SetItemPipeline,
    SetUiViewBindGroup<ExtractedUiNode, 0>,
    SetUiTextureBindGroup<ExtractedUiNode, 1>,
    DrawUiRenderObject<ExtractedUiNode, 2>,
);

/// The render command that sets the bind group corresponding to the view
/// uniform for a UI render object.
///
/// The `I` type parameter specifies the index of the bind group.
pub struct SetUiViewBindGroup<E, const I: usize>(PhantomData<E>)
where
    E: UiRenderObject;
impl<E, P: PhaseItem, const I: usize> RenderCommand<P> for SetUiViewBindGroup<E, I>
where
    E: UiRenderObject,
{
    type Param = SRes<UiMeta<E>>;
    type ViewQuery = Read<ViewUniformOffset>;
    type ItemQuery = ();

    fn render<'w>(
        _item: &P,
        view_uniform: &'w ViewUniformOffset,
        _entity: Option<()>,
        ui_meta: SystemParamItem<'w, '_, Self::Param>,
        pass: &mut TrackedRenderPass<'w>,
    ) -> RenderCommandResult {
        let Some(view_bind_group) = ui_meta.into_inner().view_bind_group.as_ref() else {
            return RenderCommandResult::Failure("view_bind_group not available");
        };
        pass.set_bind_group(I, view_bind_group, &[view_uniform.offset]);
        RenderCommandResult::Success
    }
}
/// The render command that sets the bind group corresponding to the texture
/// uniform for a UI render object.
///
/// The `I` type parameter specifies the index of the bind group.
pub struct SetUiTextureBindGroup<E, const I: usize>(PhantomData<E>)
where
    E: UiRenderObject;
impl<E, P: PhaseItem, const I: usize> RenderCommand<P> for SetUiTextureBindGroup<E, I>
where
    E: UiRenderObject,
{
    type Param = SRes<UiTexturedBindGroups<E::TexturedGpuAsset>>;
    type ViewQuery = ();
    type ItemQuery = Read<UiBatch<E>>;

    #[inline]
    fn render<'w>(
        _item: &P,
        _view: (),
        batch: Option<&'w UiBatch<E>>,
        textured_bind_groups: SystemParamItem<'w, '_, Self::Param>,
        pass: &mut TrackedRenderPass<'w>,
    ) -> RenderCommandResult {
        let Some(batch) = batch else {
            return RenderCommandResult::Skip;
        };

        let Some(slab) = textured_bind_groups
            .into_inner()
            .allocator
            .get(batch.textured_bind_group_index)
        else {
            return RenderCommandResult::Skip;
        };
        let Some(bind_group) = slab.bind_group() else {
            return RenderCommandResult::Skip;
        };

        pass.set_bind_group(I, bind_group, &[]);
        RenderCommandResult::Success
    }
}

/// The render command that issues the draw command to render a UI render
/// object.
///
/// If UI instances are in retained mode, then the instance buffer is bound as
/// well. The supplied constant parameter specifies the bind group of the
/// instance buffer, when retained mode is enabled.
pub struct DrawUiRenderObject<E, const INSTANCES_BIND_GROUP: usize>(PhantomData<E>)
where
    E: UiRenderObject;
impl<E, P: PhaseItem, const INSTANCES_BIND_GROUP: usize> RenderCommand<P>
    for DrawUiRenderObject<E, INSTANCES_BIND_GROUP>
where
    E: UiRenderObject,
{
    type Param = SRes<UiMeta<E>>;
    type ViewQuery = ();
    type ItemQuery = Read<UiBatch<E>>;

    #[inline]
    fn render<'w>(
        _item: &P,
        _view: (),
        batch: Option<&'w UiBatch<E>>,
        ui_meta: SystemParamItem<'w, '_, Self::Param>,
        pass: &mut TrackedRenderPass<'w>,
    ) -> RenderCommandResult {
        let Some(batch) = batch else {
            return RenderCommandResult::Skip;
        };
        let Some(ref params_range) = batch.params_range else {
            return RenderCommandResult::Skip;
        };
        let ui_meta = ui_meta.into_inner();
        let Some(vertices) = ui_meta.vertices.buffer() else {
            return RenderCommandResult::Failure("missing vertices to draw ui");
        };
        let Some(indices) = ui_meta.indices.buffer() else {
            return RenderCommandResult::Failure("missing indices to draw ui");
        };

        // Store the vertices
        pass.set_vertex_buffer(0, vertices.slice(..));

        // Attach the instance index buffer (if in retained mode) and instance
        // buffer.
        match ui_meta.instances {
            UiInstances::Retained {
                ref instance_index_buffer,
                ref bind_group,
                ..
            } => {
                let Some(instance_index_buffer) = instance_index_buffer.buffer() else {
                    return RenderCommandResult::Failure(
                        "missing instance index buffer to draw ui",
                    );
                };
                let Some(bind_group) = bind_group.as_ref() else {
                    return RenderCommandResult::Failure(
                        "missing retained instance bind group to draw ui",
                    );
                };
                pass.set_vertex_buffer(1, instance_index_buffer.slice(..));
                pass.set_bind_group(INSTANCES_BIND_GROUP, bind_group, &[]);
            }
            UiInstances::Immediate { ref instances } => {
                let Some(instances) = instances.buffer() else {
                    return RenderCommandResult::Failure("missing instances to draw ui");
                };
                pass.set_vertex_buffer(1, instances.slice(..));
            }
        }

        // Define how to "connect" the vertices
        pass.set_index_buffer(
            indices.slice(..),
            bevy_render::render_resource::IndexFormat::Uint32,
        );
        match ui_meta.draw_args {
            UiMetaDrawArgs::Indirect(ref indirect_draw_args) => {
                let Some(indirect_draw_args_buffer) = indirect_draw_args.buffer() else {
                    return RenderCommandResult::Failure(
                        "missing indirect draw arguments buffer to draw ui",
                    );
                };
                pass.multi_draw_indexed_indirect(
                    indirect_draw_args_buffer,
                    params_range.start as u64 * size_of::<IndirectParametersIndexed>() as u64,
                    params_range.end - params_range.start,
                );
            }
            UiMetaDrawArgs::Direct(ref draw_params) => {
                for params in &draw_params[params_range.start as usize..params_range.end as usize] {
                    pass.draw_indexed(
                        params.first_index..(params.first_index + params.index_count),
                        params.base_vertex as i32,
                        params.first_instance..(params.first_instance + params.instance_count),
                    );
                }
            }
        }
        RenderCommandResult::Success
    }
}
