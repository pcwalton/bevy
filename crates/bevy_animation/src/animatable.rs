//! Traits and type for interpolating between values.

use crate::{util, AnimationEvaluationError, Interpolation};
use bevy_color::{Laba, LinearRgba, Oklaba, Srgba, Xyza};
use bevy_math::*;
use bevy_reflect::{FromType, PartialReflect, Reflect};
use bevy_render::mesh::morph::MorphWeights;
use bevy_transform::prelude::Transform;

/// An individual input for [`Animatable::blend`].
pub struct BlendInput<T> {
    /// The individual item's weight. This may not be bound to the range `[0.0, 1.0]`.
    pub weight: f32,
    /// The input value to be blended.
    pub value: T,
    /// Whether or not to additively blend this input into the final result.
    pub additive: bool,
}

/// An animatable value type.
pub trait Animatable: Reflect + Sized + Send + Sync + 'static {
    /// Interpolates between `a` and `b` with  a interpolation factor of `time`.
    ///
    /// The `time` parameter here may not be clamped to the range `[0.0, 1.0]`.
    fn interpolate(a: &Self, b: &Self, time: f32) -> Self;

    /// Blends one or more values together.
    ///
    /// Implementors should return a default value when no inputs are provided here.
    fn blend(inputs: impl Iterator<Item = BlendInput<Self>>) -> Self;
}

/// A structure that allows Bevy to operate on reflected [`Animatable`]
/// implementations.
///
/// You can use the [`bevy_reflect::TypeRegistration::data`] method to obtain a
/// [`ReflectAnimatable`] for any animatable type.
///
/// The method implementations for this type differ from those on
/// [`Animatable`]; instead of exposing [`Animatable::interpolate`] directly, we
/// expose a higher-level API that interpolates between keyframes and updates a
/// destination. This is for performance, as [`Animatable::interpolate`] returns
/// a new value, and boxing the result during the animation system (which is hot
/// code) would be expensive.
#[derive(Clone)]
pub struct ReflectAnimatable {
    /// Blends the value in the first keyframe into `dest` with the given
    /// `weight`.
    ///
    /// `keyframes` must be a [`Vec`] of keyframe values. The resulting value is
    /// blended into the `dest` value according to the given `weight`; thus if
    /// `weight` is 0, `dest` is unchanged, while if `weight` is 1, the
    /// interpolated value overwrites the `dest`.
    pub interpolate_first_keyframe: fn(
        dest: &mut dyn PartialReflect,
        keyframes: &dyn Reflect,
        weight: f32,
    ) -> Result<(), AnimationEvaluationError>,

    /// Interpolates between the two keyframes with indexes `step_start` and
    /// `step_start + 1`, using the given `interpolation` mode.
    ///
    /// `keyframes` must be a [`Vec`] of keyframe values. `time` ranges from 0
    /// (the `step_start` value) to 1 (the `step_end` value). `duration` is the
    /// amount of time between the `step_start` and `step_start + 1` keyframes.
    /// The resulting value is blended into the `dest` value according to the
    /// given `weight`; thus if `weight` is 0, `dest` is unchanged, while if
    /// `weight` is 1, the interpolated value overwrites the `dest`.
    pub interpolate_keyframes: fn(
        dest: &mut dyn PartialReflect,
        keyframes: &dyn Reflect,
        interpolation: Interpolation,
        step_start: usize,
        time: f32,
        weight: f32,
        duration: f32,
    ) -> Result<(), AnimationEvaluationError>,
}

macro_rules! impl_float_animatable {
    ($ty: ty, $base: ty) => {
        impl Animatable for $ty {
            #[inline]
            fn interpolate(a: &Self, b: &Self, t: f32) -> Self {
                let t = <$base>::from(t);
                (*a) * (1.0 - t) + (*b) * t
            }

            #[inline]
            fn blend(inputs: impl Iterator<Item = BlendInput<Self>>) -> Self {
                let mut value = Default::default();
                for input in inputs {
                    if input.additive {
                        value += <$base>::from(input.weight) * input.value;
                    } else {
                        value = Self::interpolate(&value, &input.value, input.weight);
                    }
                }
                value
            }
        }
    };
}

macro_rules! impl_color_animatable {
    ($ty: ident) => {
        impl Animatable for $ty {
            #[inline]
            fn interpolate(a: &Self, b: &Self, t: f32) -> Self {
                let value = *a * (1. - t) + *b * t;
                value
            }

            #[inline]
            fn blend(inputs: impl Iterator<Item = BlendInput<Self>>) -> Self {
                let mut value = Default::default();
                for input in inputs {
                    if input.additive {
                        value += input.weight * input.value;
                    } else {
                        value = Self::interpolate(&value, &input.value, input.weight);
                    }
                }
                value
            }
        }
    };
}

impl_float_animatable!(f32, f32);
impl_float_animatable!(Vec2, f32);
impl_float_animatable!(Vec3A, f32);
impl_float_animatable!(Vec4, f32);

impl_float_animatable!(f64, f64);
impl_float_animatable!(DVec2, f64);
impl_float_animatable!(DVec3, f64);
impl_float_animatable!(DVec4, f64);

impl_color_animatable!(LinearRgba);
impl_color_animatable!(Laba);
impl_color_animatable!(Oklaba);
impl_color_animatable!(Srgba);
impl_color_animatable!(Xyza);

// Vec3 is special cased to use Vec3A internally for blending
impl Animatable for Vec3 {
    #[inline]
    fn interpolate(a: &Self, b: &Self, t: f32) -> Self {
        (*a) * (1.0 - t) + (*b) * t
    }

    #[inline]
    fn blend(inputs: impl Iterator<Item = BlendInput<Self>>) -> Self {
        let mut value = Vec3A::ZERO;
        for input in inputs {
            if input.additive {
                value += input.weight * Vec3A::from(input.value);
            } else {
                value = Vec3A::interpolate(&value, &Vec3A::from(input.value), input.weight);
            }
        }
        Self::from(value)
    }
}

impl Animatable for bool {
    #[inline]
    fn interpolate(a: &Self, b: &Self, t: f32) -> Self {
        util::step_unclamped(*a, *b, t)
    }

    #[inline]
    fn blend(inputs: impl Iterator<Item = BlendInput<Self>>) -> Self {
        inputs
            .max_by(|a, b| FloatOrd(a.weight).cmp(&FloatOrd(b.weight)))
            .map(|input| input.value)
            .unwrap_or(false)
    }
}

impl Animatable for Transform {
    fn interpolate(a: &Self, b: &Self, t: f32) -> Self {
        Self {
            translation: Vec3::interpolate(&a.translation, &b.translation, t),
            rotation: Quat::interpolate(&a.rotation, &b.rotation, t),
            scale: Vec3::interpolate(&a.scale, &b.scale, t),
        }
    }

    fn blend(inputs: impl Iterator<Item = BlendInput<Self>>) -> Self {
        let mut translation = Vec3A::ZERO;
        let mut scale = Vec3A::ZERO;
        let mut rotation = Quat::IDENTITY;

        for input in inputs {
            if input.additive {
                translation += input.weight * Vec3A::from(input.value.translation);
                scale += input.weight * Vec3A::from(input.value.scale);
                rotation = rotation.slerp(input.value.rotation, input.weight);
            } else {
                translation = Vec3A::interpolate(
                    &translation,
                    &Vec3A::from(input.value.translation),
                    input.weight,
                );
                scale = Vec3A::interpolate(&scale, &Vec3A::from(input.value.scale), input.weight);
                rotation = Quat::interpolate(&rotation, &input.value.rotation, input.weight);
            }
        }

        Self {
            translation: Vec3::from(translation),
            rotation,
            scale: Vec3::from(scale),
        }
    }
}

impl Animatable for Quat {
    /// Performs a slerp to smoothly interpolate between quaternions.
    #[inline]
    fn interpolate(a: &Self, b: &Self, t: f32) -> Self {
        // We want to smoothly interpolate between the two quaternions by default,
        // rather than using a quicker but less correct linear interpolation.
        a.slerp(*b, t)
    }

    #[inline]
    fn blend(inputs: impl Iterator<Item = BlendInput<Self>>) -> Self {
        let mut value = Self::IDENTITY;
        for input in inputs {
            value = Self::interpolate(&value, &input.value, input.weight);
        }
        value
    }
}

/// An abstraction over a list of keyframes.
///
/// See the documentation in [`MORPH_WEIGHTS_REFLECT_ANIMATABLE`] for the reason
/// why we need this trait instead of using `Vec<T>` durectly.
trait KeyframeList {
    type Output: Animatable;
    fn get(&self, index: usize) -> Option<&Self::Output>;
}

/// The standard implementation of [`KeyframeList`].
impl<T> KeyframeList for Vec<T>
where
    T: Animatable,
{
    type Output = T;

    fn get(&self, index: usize) -> Option<&Self::Output> {
        self.as_slice().get(index)
    }
}

/// Information needed to look up morph weight values in the flattened morph
/// weight keyframes vector.
struct MorphWeightsKeyframes<'a> {
    /// The flattened list of weights.
    keyframes: &'a Vec<f32>,
    /// The morph target we're interpolating.
    morph_target_index: usize,
    /// The total number of morph targets in the mesh.
    morph_target_count: usize,
}

impl<'a> KeyframeList for MorphWeightsKeyframes<'a> {
    type Output = f32;

    fn get(&self, keyframe_index: usize) -> Option<&Self::Output> {
        self.keyframes
            .as_slice()
            .get(keyframe_index * self.morph_target_count + self.morph_target_index)
    }
}

/// Interpolates between keyframes and stores the result in `dest`.
///
/// This is factored out so that it can be shared between the standard
/// implementation of [`ReflectAnimatable`] and the special implementation for
/// [`MorphWeights`].
fn interpolate_keyframes<T>(
    dest: &mut T,
    keyframes: &impl KeyframeList<Output = T>,
    interpolation: Interpolation,
    step_start: usize,
    time: f32,
    weight: f32,
    duration: f32,
) -> Result<(), AnimationEvaluationError>
where
    T: Animatable + Clone,
{
    let value = match interpolation {
        Interpolation::Step => {
            let Some(start_keyframe) = keyframes.get(step_start) else {
                return Err(AnimationEvaluationError::KeyframeNotPresent);
            };
            (*start_keyframe).clone()
        }

        Interpolation::Linear => {
            let (Some(start_keyframe), Some(end_keyframe)) =
                (keyframes.get(step_start), keyframes.get(step_start + 1))
            else {
                return Err(AnimationEvaluationError::KeyframeNotPresent);
            };

            T::interpolate(start_keyframe, end_keyframe, time)
        }

        Interpolation::CubicSpline => {
            let (
                Some(start_keyframe),
                Some(start_tangent_keyframe),
                Some(end_tangent_keyframe),
                Some(end_keyframe),
            ) = (
                keyframes.get(step_start * 3 + 1),
                keyframes.get(step_start * 3 + 2),
                keyframes.get(step_start * 3 + 3),
                keyframes.get(step_start * 3 + 4),
            )
            else {
                return Err(AnimationEvaluationError::KeyframeNotPresent);
            };

            interpolate_with_cubic_bezier(
                start_keyframe,
                start_tangent_keyframe,
                end_tangent_keyframe,
                end_keyframe,
                time,
                duration,
            )
        }
    };

    *dest = T::interpolate(dest, &value, weight);

    Ok(())
}

/// Evaluates a cubic Bézier curve at a value `t`, given two endpoints and the
/// derivatives at those endpoints.
///
/// The derivatives are linearly scaled by `duration`.
fn interpolate_with_cubic_bezier<T>(p0: &T, d0: &T, d3: &T, p3: &T, t: f32, duration: f32) -> T
where
    T: Animatable + Clone,
{
    // We're given two endpoints, along with the derivatives at those endpoints,
    // and have to evaluate the cubic Bézier curve at time t using only
    // (additive) blending and linear interpolation.
    //
    // Evaluating a Bézier curve via repeated linear interpolation when the
    // control points are known is straightforward via [de Casteljau
    // subdivision]. So the only remaining problem is to get the two off-curve
    // control points. The [derivative of the cubic Bézier curve] is:
    //
    //      B′(t) = 3(1 - t)²(P₁ - P₀) + 6(1 - t)t(P₂ - P₁) + 3t²(P₃ - P₂)
    //
    // Setting t = 0 and t = 1 and solving gives us:
    //
    //      P₁ = P₀ + B′(0) / 3
    //      P₂ = P₃ - B′(1) / 3
    //
    // These P₁ and P₂ formulas can be expressed as additive blends.
    //
    // So, to sum up, first we calculate the off-curve control points via
    // additive blending, and then we use repeated linear interpolation to
    // evaluate the curve.
    //
    // [de Casteljau subdivision]: https://en.wikipedia.org/wiki/De_Casteljau%27s_algorithm
    // [derivative of the cubic Bézier curve]: https://en.wikipedia.org/wiki/B%C3%A9zier_curve#Cubic_B%C3%A9zier_curves

    // Compute control points from derivatives.
    let p1 = T::blend(
        [
            BlendInput {
                weight: duration / 3.0,
                value: (*d0).clone(),
                additive: true,
            },
            BlendInput {
                weight: 1.0,
                value: (*p0).clone(),
                additive: true,
            },
        ]
        .into_iter(),
    );
    let p2 = T::blend(
        [
            BlendInput {
                weight: duration / -3.0,
                value: (*d3).clone(),
                additive: true,
            },
            BlendInput {
                weight: 1.0,
                value: (*p3).clone(),
                additive: true,
            },
        ]
        .into_iter(),
    );

    // Use de Casteljau subdivision to evaluate.
    let p0p1 = T::interpolate(p0, &p1, t);
    let p1p2 = T::interpolate(&p1, &p2, t);
    let p2p3 = T::interpolate(&p2, p3, t);
    let p0p1p2 = T::interpolate(&p0p1, &p1p2, t);
    let p1p2p3 = T::interpolate(&p1p2, &p2p3, t);
    T::interpolate(&p0p1p2, &p1p2p3, t)
}

impl<T> FromType<T> for ReflectAnimatable
where
    T: PartialReflect + Animatable + Clone,
{
    fn from_type() -> Self {
        Self {
            interpolate_first_keyframe: |dest,
                                         keyframes,
                                         weight|
             -> Result<(), AnimationEvaluationError> {
                let Some(keyframes) = keyframes.downcast_ref::<Vec<T>>() else {
                    return Err(AnimationEvaluationError::MalformedKeyframes);
                };

                let Some(value) = keyframes.first() else {
                    return Err(AnimationEvaluationError::KeyframeNotPresent);
                };

                let Some(dest) = dest.try_downcast_mut::<T>() else {
                    return Err(AnimationEvaluationError::PropertyNotPresent);
                };
                *dest = T::interpolate(dest, value, weight);

                Ok(())
            },

            interpolate_keyframes: |dest,
                                    keyframes,
                                    interpolation,
                                    step_start,
                                    time,
                                    weight,
                                    duration|
             -> Result<(), AnimationEvaluationError> {
                let Some(keyframes) = keyframes.downcast_ref::<Vec<T>>() else {
                    return Err(AnimationEvaluationError::MalformedKeyframes);
                };

                let Some(dest) = dest.try_downcast_mut::<T>() else {
                    return Err(AnimationEvaluationError::PropertyNotPresent);
                };

                interpolate_keyframes(
                    dest,
                    keyframes,
                    interpolation,
                    step_start,
                    time,
                    weight,
                    duration,
                )
            },
        }
    }
}

impl ReflectAnimatable {
    /// Blends the value in the first keyframe into `dest` with the given
    /// `weight`.
    ///
    /// `keyframes` must be a [`Vec`] of keyframe values. The resulting value is
    /// blended into the `dest` value according to the given `weight`; thus if
    /// `weight` is 0, `dest` is unchanged, while if `weight` is 1, the
    /// interpolated value overwrites the `dest`.
    pub(crate) fn interpolate_keyframe(
        &self,
        dest: &mut dyn PartialReflect,
        keyframes: &dyn Reflect,
        weight: f32,
    ) -> Result<(), AnimationEvaluationError> {
        (self.interpolate_first_keyframe)(dest, keyframes, weight)
    }

    /// Interpolates between the two keyframes with indexes `step_start` and
    /// `step_start + 1`, using the given `interpolation` mode.
    ///
    /// `keyframes` must be a [`Vec`] of keyframe values. `time` ranges from 0
    /// (the `step_start` value) to 1 (the `step_end` value). `duration` is the
    /// amount of time between the `step_start` and `step_start + 1` keyframes.
    /// The resulting value is blended into the `dest` value according to the
    /// given `weight`; thus if `weight` is 0, `dest` is unchanged, while if
    /// `weight` is 1, the interpolated value overwrites the `dest`.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn interpolate_keyframes(
        &self,
        dest: &mut dyn PartialReflect,
        keyframes: &dyn Reflect,
        interpolation: Interpolation,
        step_start: usize,
        time: f32,
        weight: f32,
        duration: f32,
    ) -> Result<(), AnimationEvaluationError> {
        (self.interpolate_keyframes)(
            dest,
            keyframes,
            interpolation,
            step_start,
            time,
            weight,
            duration,
        )
    }
}

/// A special case of [`ReflectAnimatable`] for [`MorphWeights`].
///
/// We use this special case instead of the normal implementation based on
/// [`Animatable`] for performance. If we used [`Animatable`], we would need the
/// keyframe list to be a `Vec<Vec<f32>>`: a list of keyframes, each of which
/// contains a list of morph weights. This would result in too many allocations.
/// Instead, we want a flat list of morph weights: `Vec<f32>`. Supporting this
/// optimization requires this special [`ReflectAnimatable`] implementation.
pub(crate) static MORPH_WEIGHTS_REFLECT_ANIMATABLE: ReflectAnimatable = ReflectAnimatable {
    interpolate_first_keyframe: |dest, keyframes, weight| {
        let Some(keyframes) = keyframes.downcast_ref::<Vec<f32>>() else {
            return Err(AnimationEvaluationError::MalformedKeyframes);
        };
        let Some(dest) = dest.try_downcast_mut::<MorphWeights>() else {
            return Err(AnimationEvaluationError::PropertyNotPresent);
        };

        // TODO: Go 4 weights at a time to make better use of SIMD.
        for (morph_target_index, morph_weight) in dest.weights_mut().iter_mut().enumerate() {
            *morph_weight = f32::interpolate(morph_weight, &keyframes[morph_target_index], weight);
        }

        Ok(())
    },

    interpolate_keyframes: |dest, keyframes, interpolation, step_start, time, weight, duration| {
        let Some(keyframes) = keyframes.downcast_ref::<Vec<f32>>() else {
            return Err(AnimationEvaluationError::MalformedKeyframes);
        };
        let Some(dest) = dest.try_downcast_mut::<MorphWeights>() else {
            return Err(AnimationEvaluationError::PropertyNotPresent);
        };

        // TODO: Go 4 weights at a time to make better use of SIMD.
        let morph_target_count = dest.weights().len();
        for (morph_target_index, morph_weight) in dest.weights_mut().iter_mut().enumerate() {
            interpolate_keyframes(
                morph_weight,
                &MorphWeightsKeyframes {
                    keyframes,
                    morph_target_index,
                    morph_target_count,
                },
                interpolation,
                step_start,
                time,
                weight,
                duration,
            )?;
        }

        Ok(())
    },
};
