//! Shows how to use animation clips to animate UI properties.

use std::any::TypeId;

use bevy::{
    animation::{AnimationTarget, AnimationTargetId},
    prelude::*,
    reflect::ParsedPath,
};

// Holds information about the animation we programmatically create.
struct AnimationInfo {
    // The name of the animation target (in this case, the text).
    target_name: Name,
    // The ID of the animation target, derived from the name.
    target_id: AnimationTargetId,
    // The animation graph asset.
    graph: Handle<AnimationGraph>,
    // The index of the node within that graph.
    node_index: AnimationNodeIndex,
}

// The entry point.
fn main() {
    App::new()
        .add_plugins(DefaultPlugins)
        // Note that we don't need any systems other than the setup system,
        // because Bevy automatically updates animations every frame.
        .add_systems(Startup, setup)
        .run();
}

impl AnimationInfo {
    // Programmatically creates the UI animation.
    fn create(
        animation_graphs: &mut Assets<AnimationGraph>,
        animation_clips: &mut Assets<AnimationClip>,
    ) -> AnimationInfo {
        // Create an ID that identifies the text node we're going to animate.
        let animation_target_name = Name::new("Text");
        let animation_target_id = AnimationTargetId::from_name(&animation_target_name);

        // Allocate an animation clip.
        let mut animation_clip = AnimationClip::default();

        // Create a curve that animates font size.
        animation_clip.add_curve_to_target(
            animation_target_id,
            VariableCurve {
                keyframe_timestamps: vec![0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0],
                keyframes: Keyframes {
                    component: TypeId::of::<Text>(),
                    path: ParsedPath::parse("sections[0].style.font_size")
                        .unwrap()
                        .into(),
                    // NB: We must explicitly use the `f32` suffix here to match
                    // the type of `font_size`. If we don't, then Rust will default
                    // to `f64`. That will make Bevy emit a `MalformedKeyframes`
                    // error when we try to animate, because the types of the
                    // keyframes and the animatable property won't match.
                    keyframes: Box::new(vec![
                        24.0f32, 80.0f32, 24.0f32, 80.0f32, 24.0f32, 80.0f32, 24.0f32,
                    ]),
                },
                interpolation: Interpolation::Linear,
            },
        );

        // Create a curve that animates font color.
        // Note that this should have the same time duration as the previous curve.
        animation_clip.add_curve_to_target(
            animation_target_id,
            VariableCurve {
                keyframe_timestamps: vec![0.0, 1.0, 2.0, 3.0],
                keyframes: Keyframes {
                    component: TypeId::of::<Text>(),
                    // The final `.0` is there to grab the `Srgba` value from the
                    // `Color` enum.
                    path: ParsedPath::parse("sections[0].style.color.0")
                        .unwrap()
                        .into(),
                    keyframes: Box::new(vec![Srgba::RED, Srgba::GREEN, Srgba::BLUE, Srgba::RED]),
                },
                interpolation: Interpolation::Linear,
            },
        );

        // Save our animation clip as an asset.
        let animation_clip_handle = animation_clips.add(animation_clip);

        // Create an animation graph with that clip.
        let (animation_graph, animation_node_index) =
            AnimationGraph::from_clip(animation_clip_handle);
        let animation_graph_handle = animation_graphs.add(animation_graph);

        AnimationInfo {
            target_name: animation_target_name,
            target_id: animation_target_id,
            graph: animation_graph_handle,
            node_index: animation_node_index,
        }
    }
}

// Creates all the entities in the scene.
fn setup(
    mut commands: Commands,
    asset_server: Res<AssetServer>,
    mut animation_graphs: ResMut<Assets<AnimationGraph>>,
    mut animation_clips: ResMut<Assets<AnimationClip>>,
) {
    // Create the animation.
    let AnimationInfo {
        target_name: animation_target_name,
        target_id: animation_target_id,
        graph: animation_graph,
        node_index: animation_node_index,
    } = AnimationInfo::create(&mut animation_graphs, &mut animation_clips);

    // Build an animation player that automatically plays the UI animation.
    let mut animation_player = AnimationPlayer::default();
    animation_player.play(animation_node_index).repeat();

    // Add a camera.
    commands.spawn(Camera2dBundle::default());

    // Build the UI. We have a parent node that covers the whole screen and
    // contains the `AnimationPlayer`, as well as a child node that contains the
    // text to be animated.
    commands
        .spawn(NodeBundle {
            // Cover the whole screen, and center contents.
            style: Style {
                position_type: PositionType::Absolute,
                top: Val::Px(0.0),
                left: Val::Px(0.0),
                right: Val::Px(0.0),
                bottom: Val::Px(0.0),
                justify_content: JustifyContent::Center,
                align_items: AlignItems::Center,
                ..default()
            },
            ..default()
        })
        .insert(animation_player)
        .insert(animation_graph)
        .with_children(|builder| {
            // Build the text node.
            let player = builder.parent_entity();
            builder
                .spawn(
                    TextBundle::from_section(
                        "Bevy",
                        TextStyle {
                            font: asset_server.load("fonts/FiraSans-Bold.ttf"),
                            font_size: 24.0,
                            // Because we're animating sRGBA values, make sure
                            // that this is `Color::Srgba`. If this isn't done,
                            // Bevy will emit a `MalformedKeyframes` error, as
                            // the type of the property won't match the type of
                            // the keyframes.
                            color: Color::Srgba(Srgba::RED),
                        },
                    )
                    .with_text_justify(JustifyText::Center),
                )
                // Mark as an animation target.
                .insert(AnimationTarget {
                    id: animation_target_id,
                    player,
                })
                .insert(animation_target_name);
        });
}
