//! A number parameter's time samples, and the value USD reads between them.

use std::sync::Arc;

use openusd::gf::{self, f16};
use openusd::sdf::compare_sample_times;

/// One sample's number, in the type its file gives it: the type decides what is read between
/// two samples.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum Number {
    Double(f64),
    Float(f32),
    Half(f16),
    Int(i32),
}

impl Number {
    pub(crate) fn value(self) -> f64 {
        match self {
            Number::Double(number) => number,
            Number::Float(number) => f64::from(number),
            Number::Half(number) => f64::from(number),
            Number::Int(number) => f64::from(number),
        }
    }
}

/// A parameter's composed time samples, ascending in stage time, of which there is at least one.
/// `None` is a blocked sample: from it to the next, the parameter has no value.
///
/// Evaluation reads these in place of the stage, which stays on the thread that edits it.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Samples(Arc<[(f64, Option<Number>)]>);

impl Samples {
    /// `None` for no samples, which animate nothing. The samples must be in the order
    /// `Attribute::time_samples` gives them.
    pub(crate) fn new(samples: Vec<(f64, Option<Number>)>) -> Option<Samples> {
        (!samples.is_empty()).then(|| Samples(samples.into()))
    }

    /// The frames that have a sample, ascending.
    pub(crate) fn frames(&self) -> impl Iterator<Item = f64> + '_ {
        self.0.iter().map(|&(frame, _)| frame)
    }

    /// The value at a frame as `UsdAttribute::Get` reads it from a stage interpolating linearly,
    /// by the same arithmetic as the `openusd` crate: `double`, `float` and `half` in a straight
    /// line between two samples, each in its own precision, and anything else, `int` or a pair
    /// of two types, held from the earlier one. The end samples hold outside them. `None` from a
    /// blocked sample to the next; a block after a sample leaves that sample held up to it.
    pub(crate) fn at(&self, frame: f64) -> Option<f64> {
        let samples = &*self.0;
        let (first, last) = (samples.first()?, samples.last()?);
        if compare_sample_times(frame, first.0).is_le() {
            return first.1.map(Number::value);
        }
        if compare_sample_times(frame, last.0).is_ge() {
            return last.1.map(Number::value);
        }
        let after = match samples.binary_search_by(|&(time, _)| compare_sample_times(time, frame)) {
            Ok(exact) => return samples[exact].1.map(Number::value),
            Err(after) => after,
        };
        let (from, earlier) = samples[after - 1];
        let (to, later) = samples[after];
        let earlier = earlier?;
        let Some(later) = later else {
            return Some(earlier.value());
        };
        let along = (frame - from) / (to - from);
        Some(match (earlier, later) {
            (Number::Double(a), Number::Double(b)) => gf::lerp(a, b, along),
            (Number::Float(a), Number::Float(b)) => f64::from(gf::lerp(a, b, along as f32)),
            (Number::Half(a), Number::Half(b)) => f64::from(gf::lerp_half(a, b, along as f32)),
            _ => earlier.value(),
        })
    }
}
