//! Keyframes of animation clips.

use std::any::{Any, TypeId};

use bevy_derive::{Deref, DerefMut};
use bevy_math::{Quat, Vec3};
use bevy_reflect::OffsetAccess;
use bevy_reflect::{
    utility::NonGenericTypeInfoCell, Access, ApplyError, DynamicStruct, FieldIter, FromReflect,
    FromType, GetTypeRegistration, ParsedPath, PartialReflect, Reflect, ReflectFromPtr,
    ReflectKind, ReflectMut, ReflectOwned, ReflectRef, Struct, TypeInfo, TypePath,
    TypeRegistration, Typed, ValueInfo,
};
use bevy_render::mesh::morph::MorphWeights;
use bevy_transform::prelude::Transform;

/// The field to be animated within the component.
///
/// A [`KeyframePath`] can be constructed using `into()` on a [`ParsedPath`].
/// See the [`ParsedPath`] documentation for more details as to how to spell
/// paths.
///
/// For example, to animate [`Transform::rotation`], you could use
/// `ParsedPath::parse("rotation").unwrap().into()`. To animate font size in a
/// [`bevy_ui::Text`] component, you could use
/// `ParsedPath::parse("sections[0].style.font_size").unwrap().into()`. To
/// target the component itself (which is a rare use case that requires that the
/// component itself be [`crate::Animatable`]), supply an empty [`ParsedPath`].
#[derive(Clone, Debug, Reflect, Deref, DerefMut)]
#[reflect_value]
pub struct KeyframePath(pub ParsedPath);

impl From<ParsedPath> for KeyframePath {
    fn from(path: ParsedPath) -> Self {
        Self(path)
    }
}

/// The property to animate, and the values that that property takes on at each
/// keyframe.
///
/// Bevy can animate any property on a component that is reflectable and
/// implements [`crate::Animatable`]. Convenience methods are provided for
/// animating translations, rotations, scaling, and morph weights, which are the
/// common cases. However, animation is not limited to these values.
#[derive(Debug, TypePath)]
pub struct Keyframes {
    /// The [`TypeId`] of the component that contains the property to be
    /// animated.
    ///
    /// Typically, you use [`TypeId::of`] to construct this value. For instance,
    /// to animate a field on [`Transform`], you would use
    /// `TypeId::of::<Transform>()`.
    pub component: TypeId,

    /// Contains the [`ParsedPath`] that specifies the property on the component
    /// to be animated.
    ///
    /// Use [`ParsedPath::parse`] and [`Into::into`] to construct this. For
    /// example, to animate the `translation` property on the [`Transform`]
    /// component, you could use `path:
    /// ParsedPath::parse("translation").unwrap().into()`.
    pub path: KeyframePath,

    /// A boxed [`Vec`] of keyframe values, cast to a [`PartialReflect`].
    ///
    /// This must be a flat vector of keyframe values. The minimum number of
    /// values depends on the type of [`crate::Interpolation`] that this crate
    /// uses: 1 per keyframe if the interpolation is [`Interpolation::Step`] or
    /// [`Interpolation::Linear`], or 3 per keyframe if the interpolation is
    /// [`Interpolation::CubicBezier`]. The type of each keyframe value must
    /// precisely match the type of the property.
    ///
    /// For example, to animate the translation property on an animation with 3
    /// keyframes, this might be set to `keyframes: Box::new(vec![vec3(1.0, 2.0,
    /// 3.0), vec3(-3.0, -5.0, 7.0), vec3(1.5, 4.2, 0.9)])`
    ///
    /// Note that, if the property is of type `f32`, you must explicitly specify
    /// float values with an `f32` suffix, as the Rust compiler doesn't know the
    /// type of the property you're trying to animate, and it will default to
    /// `f64` in the absence of such information. So, for example, if you're
    /// animating the `font_size` property of a text section, writing
    /// `Box::new(vec![24.0, 72.0])` would be *incorrect* and would fail with
    /// [`crate::AnimationEvaluationError::MalformedKeyframes`]. Instead, write
    /// `Box::new(vec![24.0f32, 72.0f32])`.
    pub keyframes: Box<dyn PartialReflect>,
}

impl Keyframes {
    /// Returns the number of keyframes.
    pub fn len(&self) -> usize {
        match self.keyframes.reflect_ref() {
            ReflectRef::List(list) => list.len(),
            ReflectRef::Array(array) => array.len(),
            _ => 0,
        }
    }

    /// Returns true if the number of keyframes is zero.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Constructs keyframes that animate translation.
    pub fn translation(keyframes: Vec<Vec3>) -> Keyframes {
        Keyframes {
            component: TypeId::of::<Transform>(),
            // As a micro-optimization, we build the `ParsedPath` manually
            // instead of using `ParsedPath::parse`.
            path: KeyframePath(ParsedPath(vec![OffsetAccess {
                access: Access::FieldIndex(0),
                offset: None,
            }])),
            keyframes: Box::new(keyframes),
        }
    }

    /// Constructs keyframes that animate rotation.
    pub fn rotation(keyframes: Vec<Quat>) -> Keyframes {
        Keyframes {
            component: TypeId::of::<Transform>(),
            // As a micro-optimization, we build the `ParsedPath` manually
            // instead of using `ParsedPath::parse`.
            path: KeyframePath(ParsedPath(vec![OffsetAccess {
                access: Access::FieldIndex(1),
                offset: None,
            }])),
            keyframes: Box::new(keyframes),
        }
    }

    /// Constructs keyframes that animate scale.
    pub fn scale(keyframes: Vec<Vec3>) -> Keyframes {
        Keyframes {
            component: TypeId::of::<Transform>(),
            // As a micro-optimization, we build the `ParsedPath` manually
            // instead of using `ParsedPath::parse`.
            path: KeyframePath(ParsedPath(vec![OffsetAccess {
                access: Access::FieldIndex(2),
                offset: None,
            }])),
            keyframes: Box::new(keyframes),
        }
    }

    /// Constructs keyframes that animate morph weights.
    ///
    /// Morph weights are expected to be flattened with keyframes at the
    /// outermost level. For example, if the mesh has 3 morph targets, and the
    /// animation has 2 keyframes, the list would consist of (target 0 keyframe
    /// 0, target 1 keyframe 0, target 2 keyframe 0, target 0 keyframe 1, target
    /// 1 keyframe 1, target 2 keyframe 1).
    pub fn weights(keyframes: Vec<f32>) -> Keyframes {
        Keyframes {
            component: TypeId::of::<MorphWeights>(),
            // As a micro-optimization, we build the `ParsedPath` manually
            // instead of using `ParsedPath::parse`.
            path: KeyframePath(ParsedPath(vec![])),
            keyframes: Box::new(keyframes),
        }
    }
}

// We have to implement `Clone` manually because boxed `PartialReflect` objects
// aren't cloneable in the usual way.
impl Clone for Keyframes {
    fn clone(&self) -> Self {
        Self {
            component: self.component,
            path: self.path.clone(),
            keyframes: self.keyframes.clone_value(),
        }
    }
}

// We have to implement `PartialReflect` manually because the
// `#[derive(Reflect)]` macro doesn't know how to delegate to
// `Box<dyn PartialReflect>` values.
impl PartialReflect for Keyframes {
    fn get_represented_type_info(&self) -> Option<&'static TypeInfo> {
        Some(<Self as Typed>::type_info())
    }

    #[inline]
    fn into_partial_reflect(self: Box<Self>) -> Box<dyn PartialReflect> {
        self
    }

    fn as_partial_reflect(&self) -> &dyn PartialReflect {
        self
    }

    fn as_partial_reflect_mut(&mut self) -> &mut dyn PartialReflect {
        self
    }

    fn try_into_reflect(self: Box<Self>) -> Result<Box<dyn Reflect>, Box<dyn PartialReflect>> {
        Ok(self)
    }

    fn try_as_reflect(&self) -> Option<&dyn Reflect> {
        Some(self)
    }

    fn try_as_reflect_mut(&mut self) -> Option<&mut dyn Reflect> {
        Some(self)
    }

    fn try_apply(&mut self, value: &dyn PartialReflect) -> Result<(), ApplyError> {
        match value.reflect_ref() {
            ReflectRef::Struct(struct_value) => {
                for (i, value) in struct_value.iter_fields().enumerate() {
                    let name = struct_value.name_at(i).unwrap();
                    if let Some(field_value) = self.field_mut(name) {
                        field_value.try_apply(value)?;
                    }
                }
                Ok(())
            }
            _ => Err(ApplyError::MismatchedKinds {
                from_kind: value.reflect_kind(),
                to_kind: ReflectKind::Struct,
            }),
        }
    }

    fn reflect_ref(&self) -> ReflectRef {
        ReflectRef::Struct(self)
    }

    fn reflect_mut(&mut self) -> ReflectMut {
        ReflectMut::Struct(self)
    }

    fn reflect_owned(self: Box<Self>) -> ReflectOwned {
        ReflectOwned::Struct(self)
    }

    fn clone_value(&self) -> Box<dyn PartialReflect> {
        Box::new(self.clone_dynamic())
    }
}

// As above, we have to implement `Reflect` manually because the
// `#[derive(Reflect)]` macro doesn't know how to delegate to `Box<dyn
// PartialReflect>` values.
impl Reflect for Keyframes {
    fn into_any(self: Box<Self>) -> Box<dyn Any> {
        self
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }

    fn into_reflect(self: Box<Self>) -> Box<dyn Reflect> {
        self
    }

    fn as_reflect(&self) -> &dyn Reflect {
        self
    }

    fn as_reflect_mut(&mut self) -> &mut dyn Reflect {
        self
    }

    fn set(&mut self, value: Box<dyn Reflect>) -> Result<(), Box<dyn Reflect>> {
        *self = value.take()?;
        Ok(())
    }
}

// We have to implement `FromReflect` manually because the derive macro doesn't
// know how to construct the `keyframes` field.
impl FromReflect for Keyframes {
    fn from_reflect(reflect: &dyn PartialReflect) -> Option<Self> {
        match PartialReflect::reflect_ref(reflect) {
            ReflectRef::Struct(reflect_struct) => Some(Self {
                component: FromReflect::from_reflect(reflect_struct.field_at(0)?)?,
                path: FromReflect::from_reflect(reflect_struct.field_at(1)?)?,
                keyframes: reflect_struct.field_at(2)?.clone_value(),
            }),
            _ => None,
        }
    }
}

// We have to implement `GetTypeRegistration` manually because we implemented
// `PartialReflect` manually.
impl GetTypeRegistration for Keyframes {
    fn get_type_registration() -> TypeRegistration {
        let mut registration = TypeRegistration::of::<Self>();
        registration.insert::<ReflectFromPtr>(FromType::<Self>::from_type());
        registration
    }
}

// We have to implement `Typed` manually because we implemented `PartialReflect`
// manually.
impl Typed for Keyframes {
    fn type_info() -> &'static TypeInfo {
        static CELL: NonGenericTypeInfoCell = NonGenericTypeInfoCell::new();
        CELL.get_or_set(|| TypeInfo::Value(ValueInfo::new::<Self>()))
    }
}

// We have to implement `Struct` manually because we implemented
// `PartialReflect` manually.
impl Struct for Keyframes {
    fn field(&self, name: &str) -> Option<&dyn PartialReflect> {
        match name {
            "component" => Some(&self.component),
            "path" => Some(&self.path),
            "keyframes" => Some(self.keyframes.as_partial_reflect()),
            _ => None,
        }
    }

    fn field_mut(&mut self, name: &str) -> Option<&mut dyn PartialReflect> {
        match name {
            "component" => Some(&mut self.component),
            "path" => Some(&mut self.path),
            "keyframes" => Some(self.keyframes.as_partial_reflect_mut()),
            _ => None,
        }
    }

    fn field_at(&self, index: usize) -> Option<&dyn PartialReflect> {
        match index {
            0 => Some(&self.component),
            1 => Some(&self.path),
            2 => Some(self.keyframes.as_partial_reflect()),
            _ => None,
        }
    }

    fn field_at_mut(&mut self, index: usize) -> Option<&mut dyn PartialReflect> {
        match index {
            0 => Some(&mut self.component),
            1 => Some(&mut self.path),
            2 => Some(self.keyframes.as_partial_reflect_mut()),
            _ => None,
        }
    }

    fn name_at(&self, index: usize) -> Option<&str> {
        match index {
            0 => Some("component"),
            1 => Some("path"),
            2 => Some("keyframes"),
            _ => None,
        }
    }

    fn field_len(&self) -> usize {
        3
    }

    fn iter_fields(&self) -> FieldIter {
        FieldIter::new(self)
    }

    fn clone_dynamic(&self) -> DynamicStruct {
        let mut dynamic = DynamicStruct::default();
        dynamic.set_represented_type(PartialReflect::get_represented_type_info(self));
        dynamic.insert_boxed("component", PartialReflect::clone_value(&self.component));
        dynamic.insert_boxed("path", PartialReflect::clone_value(&self.path));
        dynamic.insert_boxed("keyframes", self.keyframes.clone_value());
        dynamic
    }
}
