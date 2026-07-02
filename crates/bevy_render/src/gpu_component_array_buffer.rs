use crate::storage::ShaderBuffer;

use bevy_app::{App, Plugin, PostUpdate};
use bevy_asset::{Assets, Handle, RenderAssetUsages};
use bevy_ecs::{
    prelude::Entity,
    query::{QueryFilter, QueryItem, ReadOnlyQueryData},
    resource::Resource,
    system::{Commands, If, Query, ResMut},
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
    pub buffer: Handle<ShaderBuffer>,
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
}

impl<C> Default for GpuComponentArrayBufferPlugin<C>
where
    C: GpuComponentArrayBuffer,
{
    fn default() -> Self {
        Self(PhantomData::<C>)
    }
}

impl<C> GpuComponentArray<C>
where
    C: GpuComponentArrayBuffer,
{
    pub fn new(shader_buffer_assets: &mut Assets<ShaderBuffer>) -> Self {
        let buffer = shader_buffer_assets.add(ShaderBuffer {
            data: Some(vec![0; size_of::<C>()]),
            buffer_description: C::buffer_descriptor(),
            asset_usage: RenderAssetUsages::MAIN_WORLD | RenderAssetUsages::RENDER_WORLD,
            copy_on_resize: true,
        });

        GpuComponentArray {
            buffer,
            tag_to_entity: vec![],
            phantom: PhantomData,
        }
    }
}

fn update_components<C>(
    mut commands: Commands,
    query: Query<(Entity, Option<&MeshTag>, C::QueryData), C::QueryFilter>,
    mut component_array: If<ResMut<GpuComponentArray<C>>>,
    mut shader_buffers: ResMut<Assets<ShaderBuffer>>,
) where
    C: GpuComponentArrayBuffer,
{
    let Some(mut buffer) = shader_buffers.get_mut(&mut component_array.buffer) else {
        return;
    };

    for (entity, maybe_tag, item) in &query {
        match C::extract_component(item) {
            None => {
                // TODO: remove
            }
            Some(data) => match maybe_tag {
                None => {
                    let tag = component_array.len();
                    component_array.push(&mut buffer, entity, data);
                    commands.entity(entity).insert(MeshTag(tag as u32));
                    println!("gpu component array buffer processed new mesh");
                }
                Some(tag) => {
                    component_array.set(&mut buffer, tag.0 as usize, data);
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
        self.tag_to_entity.len()
    }

    fn is_empty(&self) -> bool {
        self.tag_to_entity.is_empty()
    }

    fn push(&mut self, buffer: &mut ShaderBuffer, entity: Entity, data: C::Out) {
        let data_buffer = buffer.data.get_or_insert_default();
        if self.is_empty() {
            data_buffer.clear();
        }
        data_buffer.extend_from_slice(bytemuck::cast_slice(&[data]));

        self.tag_to_entity.push(entity);

        debug_assert_eq!(data_buffer.len() / size_of::<C::Out>(), self.len());
    }

    fn set(&mut self, buffer: &mut ShaderBuffer, index: usize, data: C::Out) {
        bytemuck::cast_slice_mut(buffer.data.get_or_insert_default().as_mut_slice())[index] = data;
    }
}
