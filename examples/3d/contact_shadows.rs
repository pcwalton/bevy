//! Demonstrates contact shadows, also known as screen-space shadows.

use std::marker::PhantomData;

use bevy::{
    core_pipeline::prepass::DepthPrepass, ecs::system::EntityCommands, pbr::ContactShadowsSettings,
    prelude::*,
};

const LIGHT_ROTATION_SPEED: f32 = 0.01;

#[derive(Resource, Default)]
struct AppStatus {
    contact_shadows: ContactShadows,
    shadow_maps: ShadowMaps,
    light_rotation: LightRotation,
    light_type: LightType,
}

#[derive(Clone, Copy, PartialEq, Default)]
enum ContactShadows {
    #[default]
    Enabled,
    Disabled,
}

#[derive(Clone, Copy, PartialEq, Default)]
enum ShadowMaps {
    Enabled,
    #[default]
    Disabled,
}

#[derive(Clone, Copy, PartialEq, Default)]
enum LightRotation {
    Stationary,
    #[default]
    Rotating,
}

#[derive(Clone, Copy, PartialEq, Default)]
enum LightType {
    #[default]
    Directional,
    Point,
    Spot,
}

/// A marker component that we place on all radio `Button`s.
///
/// The type parameter specifies the setting that this button controls.
#[derive(Component, Deref, DerefMut)]
struct RadioButton<T>(T);

/// A marker component that we place on all `Text` inside radio buttons.
///
/// The type parameter specifies the setting that this button controls.
#[derive(Component, Deref, DerefMut)]
struct RadioButtonText<T>(T);

/// An event that's sent whenever the user changes one of the settings by
/// clicking a radio button.
///
/// The type parameter specifies the setting that was changed.
#[derive(Event)]
struct RadioButtonChangeEvent<T>(PhantomData<T>);

#[derive(Component)]
struct LightContainer;

trait AppSetting: Clone + Copy + PartialEq + Default + Send + Sync + 'static {
    fn get(app_status: &AppStatus) -> Self;
    fn set(self, app_status: &mut AppStatus);
}

fn main() {
    App::new()
        .add_plugins(DefaultPlugins.set(WindowPlugin {
            primary_window: Some(Window {
                title: "Bevy Contact Shadows Example".into(),
                ..default()
            }),
            ..default()
        }))
        .init_resource::<AppStatus>()
        .add_event::<RadioButtonChangeEvent<ContactShadows>>()
        .add_event::<RadioButtonChangeEvent<ShadowMaps>>()
        .add_event::<RadioButtonChangeEvent<LightRotation>>()
        .add_event::<RadioButtonChangeEvent<LightType>>()
        .add_systems(Startup, setup)
        .add_systems(Update, rotate_light)
        .add_systems(
            Update,
            (
                handle_ui_interactions_for_app_setting::<ContactShadows>,
                update_radio_buttons_for_app_setting::<ContactShadows>,
                handle_contact_shadows_change,
            )
                .chain(),
        )
        .add_systems(
            Update,
            (
                handle_ui_interactions_for_app_setting::<ShadowMaps>,
                update_radio_buttons_for_app_setting::<ShadowMaps>,
                handle_shadow_maps_change,
            )
                .chain(),
        )
        .add_systems(
            Update,
            (
                handle_ui_interactions_for_app_setting::<LightRotation>,
                update_radio_buttons_for_app_setting::<LightRotation>,
            )
                .chain(),
        )
        .add_systems(
            Update,
            (
                handle_ui_interactions_for_app_setting::<LightType>,
                update_radio_buttons_for_app_setting::<LightType>,
                handle_light_type_change,
            )
                .chain(),
        )
        .run();
}

fn setup(mut commands: Commands, asset_server: Res<AssetServer>) {
    commands
        .spawn(Camera3dBundle {
            transform: Transform::from_xyz(5.0, 3.0, 5.0)
                .looking_at(Vec3::new(0.0, 0.3, 0.0), Vec3::Y),
            ..default()
        })
        .insert(DepthPrepass)
        .insert(ContactShadowsSettings::default());

    let directional_light = commands
        .spawn(DirectionalLightBundle {
            directional_light: DirectionalLight {
                contact_shadows: true,
                ..default()
            },
            ..default()
        })
        .id();

    let point_light = commands
        .spawn(PointLightBundle {
            point_light: PointLight {
                contact_shadows: true,
                ..default()
            },
            ..default()
        })
        .id();

    let spot_light = commands
        .spawn(SpotLightBundle {
            spot_light: SpotLight {
                contact_shadows: true,
                ..default()
            },
            ..default()
        })
        .id();

    commands
        .spawn(SpatialBundle {
            transform: Transform::from_xyz(5.0, 10.0, 0.0).looking_at(Vec3::ZERO, Vec3::Y),
            ..default()
        })
        .insert(LightContainer)
        .add_child(directional_light)
        .add_child(point_light)
        .add_child(spot_light);

    commands.spawn(SceneBundle {
        scene: asset_server.load(GltfAssetLabel::Scene(0).from_asset("models/Grass/Grass.gltf")),
        ..default()
    });

    spawn_buttons(&mut commands);
}

fn spawn_buttons(commands: &mut Commands) {
    commands
        .spawn(NodeBundle {
            style: Style {
                flex_direction: FlexDirection::Column,
                position_type: PositionType::Absolute,
                row_gap: Val::Px(6.0),
                left: Val::Px(10.0),
                bottom: Val::Px(10.0),
                ..default()
            },
            ..default()
        })
        .with_children(|parent| {
            spawn_option_buttons(
                parent,
                "Contact Shadows",
                &[
                    (ContactShadows::Disabled, "Off"),
                    (ContactShadows::Enabled, "On"),
                ],
            );

            spawn_option_buttons(
                parent,
                "Shadow Maps",
                &[(ShadowMaps::Disabled, "Off"), (ShadowMaps::Enabled, "On")],
            );

            spawn_option_buttons(
                parent,
                "Light Rotation",
                &[
                    (LightRotation::Stationary, "Off"),
                    (LightRotation::Rotating, "On"),
                ],
            );

            spawn_option_buttons(
                parent,
                "Light Type",
                &[
                    (LightType::Directional, "Directional"),
                    (LightType::Point, "Point"),
                    (LightType::Spot, "Spot"),
                ],
            );
        });
}

fn spawn_option_buttons<T>(parent: &mut ChildBuilder, title: &str, options: &[(T, &str)])
where
    T: Clone + Send + Sync + 'static,
{
    // Add the parent node for the row.
    parent
        .spawn(NodeBundle {
            style: Style {
                align_items: AlignItems::Center,
                ..default()
            },
            ..default()
        })
        .with_children(|parent| {
            spawn_ui_text(parent, title, Color::WHITE).insert(Style {
                width: Val::Px(150.0),
                ..default()
            });

            for (option_index, (option_value, option_name)) in options.iter().enumerate() {
                spawn_option_button(
                    parent,
                    option_value,
                    option_name,
                    option_index == 0,
                    option_index == 0,
                    option_index == options.len() - 1,
                );
            }
        });
}

/// Spawns a single radio button that allows configuration of a setting.
///
/// The type parameter specifies the particular setting: one of `LightType`,
/// `ShadowFilter`, or `SoftShadows`.
fn spawn_option_button<T>(
    parent: &mut ChildBuilder,
    option_value: &T,
    option_name: &str,
    is_selected: bool,
    is_first: bool,
    is_last: bool,
) where
    T: Clone + Send + Sync + 'static,
{
    let (bg_color, fg_color) = if is_selected {
        (Color::WHITE, Color::BLACK)
    } else {
        (Color::BLACK, Color::WHITE)
    };

    // Add the button node.
    parent
        .spawn(ButtonBundle {
            style: Style {
                border: UiRect::all(Val::Px(1.0)).with_left(if is_first {
                    Val::Px(1.0)
                } else {
                    Val::Px(0.0)
                }),
                justify_content: JustifyContent::Center,
                align_items: AlignItems::Center,
                padding: UiRect::axes(Val::Px(12.0), Val::Px(6.0)),
                ..default()
            },
            border_color: BorderColor(Color::WHITE),
            border_radius: BorderRadius::ZERO
                .with_left(if is_first { Val::Px(6.0) } else { Val::Px(0.0) })
                .with_right(if is_last { Val::Px(6.0) } else { Val::Px(0.0) }),
            image: UiImage::default().with_color(bg_color),
            ..default()
        })
        .insert(RadioButton(option_value.clone()))
        .with_children(|parent| {
            spawn_ui_text(parent, option_name, fg_color)
                .insert(RadioButtonText(option_value.clone()));
        });
}

/// Spawns text for the UI.
///
/// Returns the `EntityCommands`, which allow further customization of the text
/// style.
fn spawn_ui_text<'a>(
    parent: &'a mut ChildBuilder,
    label: &str,
    color: Color,
) -> EntityCommands<'a> {
    parent.spawn(TextBundle::from_section(
        label,
        TextStyle {
            font_size: 18.0,
            color,
            ..default()
        },
    ))
}

fn rotate_light(
    mut lights: Query<&mut Transform, With<LightContainer>>,
    app_status: Res<AppStatus>,
) {
    if app_status.light_rotation != LightRotation::Rotating {
        return;
    }

    for mut transform in lights.iter_mut() {
        transform.rotate_y(LIGHT_ROTATION_SPEED);
    }
}

fn update_radio_buttons_for_app_setting<S>(
    mut contact_shadows_buttons: Query<(&mut UiImage, &RadioButton<S>)>,
    mut contact_shadows_button_texts: Query<(&mut Text, &RadioButtonText<S>), Without<UiImage>>,
    app_status: Res<AppStatus>,
) where
    S: AppSetting,
{
    let setting = S::get(&app_status);

    for (mut button_style, button) in contact_shadows_buttons.iter_mut() {
        update_ui_radio_button(&mut button_style, button, setting);
    }

    for (mut button_text_style, button_text) in contact_shadows_button_texts.iter_mut() {
        update_ui_radio_button_text(&mut button_text_style, button_text, setting);
    }
}

/// Checks for clicks on the radio buttons and sends `RadioButtonChangeEvent`s
/// as necessary.
fn handle_ui_interactions_for_app_setting<S>(
    mut interactions: Query<(&Interaction, &RadioButton<S>), With<Button>>,
    mut events: EventWriter<RadioButtonChangeEvent<S>>,
    mut app_status: ResMut<AppStatus>,
) where
    S: AppSetting,
{
    for (interaction, radio_button) in interactions.iter_mut() {
        // Only handle clicks.
        if *interaction == Interaction::Pressed {
            radio_button.set(&mut app_status);
            events.send(RadioButtonChangeEvent(PhantomData));
        }
    }
}

/// Updates the style of the button part of a radio button to reflect its
/// selected status.
fn update_ui_radio_button<T>(image: &mut UiImage, radio_button: &RadioButton<T>, value: T)
where
    T: PartialEq,
{
    *image = UiImage::default().with_color(if value == **radio_button {
        Color::WHITE
    } else {
        Color::BLACK
    });
}

/// Updates the style of the label of a radio button to reflect its selected
/// status.
fn update_ui_radio_button_text<T>(text: &mut Text, radio_button_text: &RadioButtonText<T>, value: T)
where
    T: PartialEq,
{
    let text_color = if value == **radio_button_text {
        Color::BLACK
    } else {
        Color::WHITE
    };

    for section in &mut text.sections {
        section.style.color = text_color;
    }
}

fn handle_contact_shadows_change(
    mut lights: Query<AnyOf<(&mut DirectionalLight, &mut PointLight, &mut SpotLight)>>,
    mut events: EventReader<RadioButtonChangeEvent<ContactShadows>>,
    app_status: Res<AppStatus>,
) {
    for _ in events.read() {
        for (mut maybe_directional_light, mut maybe_point_light, mut maybe_spot_light) in
            lights.iter_mut()
        {
            if let Some(ref mut directional_light) = maybe_directional_light {
                directional_light.contact_shadows =
                    app_status.contact_shadows == ContactShadows::Enabled;
            }
            if let Some(ref mut point_light) = maybe_point_light {
                point_light.contact_shadows = app_status.contact_shadows == ContactShadows::Enabled;
            }
            if let Some(ref mut spot_light) = maybe_spot_light {
                spot_light.contact_shadows = app_status.contact_shadows == ContactShadows::Enabled;
            }
        }
    }
}

fn handle_shadow_maps_change(
    mut lights: Query<AnyOf<(&mut DirectionalLight, &mut PointLight, &mut SpotLight)>>,
    mut events: EventReader<RadioButtonChangeEvent<ShadowMaps>>,
    app_status: Res<AppStatus>,
) {
    for _ in events.read() {
        for (mut maybe_directional_light, mut maybe_point_light, mut maybe_spot_light) in
            lights.iter_mut()
        {
            if let Some(ref mut directional_light) = maybe_directional_light {
                directional_light.shadows_enabled = app_status.shadow_maps == ShadowMaps::Enabled;
            }
            if let Some(ref mut point_light) = maybe_point_light {
                point_light.shadows_enabled = app_status.shadow_maps == ShadowMaps::Enabled;
            }
            if let Some(ref mut spot_light) = maybe_spot_light {
                spot_light.shadows_enabled = app_status.shadow_maps == ShadowMaps::Enabled;
            }
        }
    }
}

fn handle_light_type_change(
    mut lights: Query<(
        &mut Visibility,
        AnyOf<(&DirectionalLight, &PointLight, &SpotLight)>,
    )>,
    mut events: EventReader<RadioButtonChangeEvent<LightType>>,
    app_status: Res<AppStatus>,
) {
    for _ in events.read() {
        for (mut visibility, (maybe_directional_light, maybe_point_light, maybe_spot_light)) in
            lights.iter_mut()
        {
            let is_visible = match app_status.light_type {
                LightType::Directional => maybe_directional_light.is_some(),
                LightType::Point => maybe_point_light.is_some(),
                LightType::Spot => maybe_spot_light.is_some(),
            };
            *visibility = if is_visible {
                Visibility::Visible
            } else {
                Visibility::Hidden
            };
        }
    }
}

impl AppSetting for ContactShadows {
    fn get(app_status: &AppStatus) -> Self {
        app_status.contact_shadows
    }
    fn set(self, app_status: &mut AppStatus) {
        app_status.contact_shadows = self;
    }
}

impl AppSetting for ShadowMaps {
    fn get(app_status: &AppStatus) -> Self {
        app_status.shadow_maps
    }
    fn set(self, app_status: &mut AppStatus) {
        app_status.shadow_maps = self;
    }
}

impl AppSetting for LightRotation {
    fn get(app_status: &AppStatus) -> Self {
        app_status.light_rotation
    }
    fn set(self, app_status: &mut AppStatus) {
        app_status.light_rotation = self;
    }
}

impl AppSetting for LightType {
    fn get(app_status: &AppStatus) -> Self {
        app_status.light_type
    }
    fn set(self, app_status: &mut AppStatus) {
        app_status.light_type = self;
    }
}
