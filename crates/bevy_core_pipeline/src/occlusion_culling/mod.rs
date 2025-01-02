//! Occlusion culling render graph nodes.

use bevy_app::{App, Plugin};
use bevy_ecs::{query::With, schedule::IntoSystemConfigs as _, system::Query};
use bevy_render::{
    occlusion_culling::{
        FinishEarlyCullingPhaseNode, FinishLateCullingPhaseNode, OcclusionCulling,
    },
    render_graph::RenderGraphApp as _,
    render_resource::TextureUsages,
    view::prepare_view_targets,
    Render, RenderApp, RenderSet,
};

use crate::core_3d::{
    graph::{Core3d, Node3d},
    Camera3d,
};

pub struct OcclusionCullingCorePipelinePlugin;

impl Plugin for OcclusionCullingCorePipelinePlugin {
    fn build(&self, app: &mut App) {
        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };

        render_app
            .add_systems(
                Render,
                configure_occlusion_culling_view_targets
                    .after(prepare_view_targets)
                    .in_set(RenderSet::ManageViews),
            )
            .add_render_graph_node::<FinishEarlyCullingPhaseNode>(
                Core3d,
                Node3d::FinishEarlyCullingPhase,
            )
            .add_render_graph_node::<FinishLateCullingPhaseNode>(
                Core3d,
                Node3d::FinishMainCullingPhase,
            )
            .add_render_graph_edges(
                Core3d,
                (
                    Node3d::EarlyPrepass,
                    Node3d::FinishEarlyCullingPhase,
                    Node3d::DownsampleDepth,
                    Node3d::Prepass,
                    Node3d::FinishMainCullingPhase,
                    Node3d::DeferredPrepass,
                ),
            );
    }
}

fn configure_occlusion_culling_view_targets(
    mut view_targets: Query<&mut Camera3d, With<OcclusionCulling>>,
) {
    for mut camera_3d in &mut view_targets {
        let mut depth_texture_usages = TextureUsages::from(camera_3d.depth_texture_usages);
        depth_texture_usages |= TextureUsages::TEXTURE_BINDING;
        camera_3d.depth_texture_usages = depth_texture_usages.into();
    }
}
