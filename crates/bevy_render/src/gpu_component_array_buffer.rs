use crate::storage::ShaderBuffer;

use bevy_app::{App, Plugin, PostUpdate};
use bevy_asset::RenderAssetUsages;
use bevy_ecs::{
    prelude::Entity,
    query::{QueryFilter, QueryItem, ReadOnlyQueryData},
    resource::Resource,
    system::{Commands, Query, ResMut},
};
use bevy_mesh::MeshTag;
use bytemuck::Pod;
use core::marker::PhantomData;
use encase::ShaderType;
use wgpu::{BufferDescriptor, BufferUsages};

/// This plugin prepares the components of the corresponding type for the GPU
/// by storing them in a [`RawBufferVec`].
pub struct GpuComponentArrayBufferPlugin<C>(PhantomData<C>)
where
    C: GpuComponentArrayBuffer;

pub trait GpuComponentArrayBuffer: Send + Sync + 'static {
    type QueryData: ReadOnlyQueryData;
    type QueryFilter: QueryFilter;
    type Out: Pod + ShaderType + Default;

    fn extract_component(item: QueryItem<'_, '_, Self::QueryData>) -> Option<Self::Out>;

    fn buffer_descriptor() -> BufferDescriptor<'static> {
        BufferDescriptor {
            label: Some("GPU component array"),
            mapped_at_creation: false,
            size: 1,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
        }
    }
}

#[derive(Resource)]
pub struct GpuComponentArray<C>
where
    C: GpuComponentArrayBuffer,
{
    pub buffer: ShaderBuffer,
    pub tag_to_entity: Vec<Entity>,
    phantom: PhantomData<C>,
}

impl<C> Plugin for GpuComponentArrayBufferPlugin<C>
where
    C: GpuComponentArrayBuffer,
{
    fn build(&self, app: &mut App) {
        app.add_systems(PostUpdate, update_components::<C>);
    }

    fn finish(&self, app: &mut App) {
        app.init_resource::<GpuComponentArray<C>>();
    }
}

impl<C> Default for GpuComponentArrayBufferPlugin<C>
where
    C: GpuComponentArrayBuffer,
{
    fn default() -> Self {
        Self(PhantomData::<C>)
    }
}

impl<C> Default for GpuComponentArray<C>
where
    C: GpuComponentArrayBuffer,
{
    fn default() -> Self {
        GpuComponentArray {
            buffer: ShaderBuffer {
                data: None,
                buffer_description: C::buffer_descriptor(),
                asset_usage: RenderAssetUsages::RENDER_WORLD | RenderAssetUsages::MAIN_WORLD,
                // FIXME: Is this desired?
                copy_on_resize: true,
            },
            tag_to_entity: vec![],
            phantom: PhantomData,
        }
    }
}

fn update_components<C>(
    mut commands: Commands,
    query: Query<(Entity, Option<&MeshTag>, C::QueryData), C::QueryFilter>,
    mut component_array: ResMut<GpuComponentArray<C>>,
) where
    C: GpuComponentArrayBuffer,
{
    for (entity, maybe_tag, item) in &query {
        match C::extract_component(item) {
            None => {
                // TODO: remove
            }
            Some(data) => match maybe_tag {
                None => {
                    let tag = component_array.len();
                    component_array.push(entity, data);
                    commands.entity(entity).insert(MeshTag(tag as u32));
                }
                Some(tag) => {
                    component_array.set(tag.0 as usize, data);
                }
            },
        }
    }
}

impl<C> GpuComponentArray<C>
where
    C: GpuComponentArrayBuffer,
{
    fn len(&self) -> usize {
        match self.buffer.data {
            None => 0,
            Some(ref data) => data.len() / size_of::<C::Out>(),
        }
    }

    fn push(&mut self, entity: Entity, data: C::Out) {
        self.buffer
            .data
            .get_or_insert_default()
            .extend_from_slice(bytemuck::cast_slice(&[data]));
        self.tag_to_entity.push(entity);
        debug_assert_eq!(
            self.buffer.data.as_ref().unwrap().len(),
            self.tag_to_entity.len()
        );
    }

    fn set(&mut self, index: usize, data: C::Out) {
        bytemuck::cast_slice_mut(self.buffer.data.get_or_insert_default().as_mut_slice())[index] =
            data;
    }
}
