//! This example demonstrates how to use BSN to compose scenes.
use bevy::{
    ecs::template::template,
    prelude::*,
    scene2::prelude::{Scene, SpawnScene, *},
};
use bevy_ecs::template::{ErasedTemplate, TemplateContext};

fn main() {
    App::new()
        .add_plugins(DefaultPlugins)
        .add_systems(Startup, setup)
        .run();
}

fn setup(world: &mut World) -> Result {
    let asset_server = world.resource::<AssetServer>();
    asset_server.load_scene("scene://ui.bsn", ui());
    asset_server.load_scene("scene://patch_scene.bsn", patch_scene());
    let top_level_handle = asset_server.load_scene("scene://top_level.bsn", top_level());
    world.spawn_scene(bsn![Camera2d])?;
    world.spawn(ScenePatchInstance(top_level_handle));
    Ok(())
}

fn top_level() -> impl Scene {
    bsn! {
        :"scene://ui.bsn"
        @DescendantPatch::new("OkButton", Text("Hello world".to_owned()))
    }
}

#[derive(Default)]
struct DescendantPatch {
    name: String,
    subtemplate: Option<Box<dyn ErasedTemplate>>,
}

impl DescendantPatch {
    fn new<T>(name: &str, subtemplate: T) -> DescendantPatch
    where
        T: Template + Send + Sync + 'static,
        T::Output: Component,
    {
        DescendantPatch {
            name: name.to_owned(),
            subtemplate: Some(Box::new(subtemplate)),
        }
    }
}

impl Template for DescendantPatch {
    type Output = ();

    fn build_template(&self, context: &mut TemplateContext) -> Result<Self::Output> {
        let Some(kids) = context.entity.get::<Children>() else {
            return Ok(());
        };
        let kids: Vec<Entity> = kids.iter().collect();
        for kid in kids.into_iter() {
            if context
                .entity
                .world()
                .get::<Name>(kid)
                .is_none_or(|name| **name != self.name)
            {
                continue;
            }

            let Some(kids) = context.entity.world().get::<Children>(kid) else {
                continue;
            };
            let kids: Vec<Entity> = kids.iter().collect();
            for kid in kids.into_iter() {
                context.entity.world_scope(|world| match self.subtemplate {
                    Some(ref subtemplate) => subtemplate.apply(&mut TemplateContext {
                        entity: &mut world.entity_mut(kid),
                        scoped_entities: &mut *context.scoped_entities,
                        entity_scopes: context.entity_scopes,
                    }),
                    None => Ok(()),
                })?;
            }
        }

        Ok(())
    }

    fn clone_template(&self) -> Self {
        DescendantPatch {
            name: self.name.clone(),
            subtemplate: self
                .subtemplate
                .as_ref()
                .map(|subtemplate| subtemplate.clone_template()),
        }
    }
}

fn patch_scene() -> impl Scene {
    bsn! {
        Text("hello world")
    }
}

fn ui() -> impl Scene {
    bsn! {
        Node {
            width: Val::Percent(100.0),
            height: Val::Percent(100.0),
            align_items: AlignItems::Center,
            justify_content: JustifyContent::Center,
            column_gap: Val::Px(5.),
        }
        Children [
            (
                #OkButton
                button("Ok")
                on(|_event: On<Pointer<Press>>| println!("Ok pressed!"))
            ),
            (
                #CancelButton
                button("Cancel")
                on(|_event: On<Pointer<Press>>| println!("Cancel pressed!"))
                BackgroundColor(Color::srgb(0.4, 0.15, 0.15))
            ),
        ]
    }
}

fn button(label: &'static str) -> impl Scene {
    bsn! {
        Button
        Node {
            width: Val::Px(150.0),
            height: Val::Px(65.0),
            border: UiRect::all(Val::Px(5.0)),
            border_radius: BorderRadius::MAX,
            justify_content: JustifyContent::Center,
            align_items: AlignItems::Center,
        }
        BorderColor::from(Color::BLACK)
        BackgroundColor(Color::srgb(0.15, 0.15, 0.15))
        Children [(
            #ButtonText
            Text(label)
            // The `template` wrapper can be used for types that can't implement or don't yet have a template
            template(|context| {
                Ok(TextFont {
                    font: context
                        .resource::<AssetServer>()
                        .load("fonts/FiraSans-Bold.ttf").into(),
                    font_size: FontSize::Px(33.0),
                    ..default()
                })
            })
            TextColor(Color::srgb(0.9, 0.9, 0.9))
            TextShadow
        )]
    }
}
