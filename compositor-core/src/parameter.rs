//! Parameter descriptions: what a node type offers to be set, read by the parameter panel,
//! scripts and validation alike.

use std::ops::RangeInclusive;

/// One parameter of a node type. OpenFX properties beyond these stay in the plugin adapter.
#[derive(Clone, Debug, PartialEq)]
pub struct Parameter {
    /// Unchanged across versions and labels: scripts and authored values address it.
    pub name: &'static str,
    pub label: &'static str,
    pub kind: ParameterKind,
}

/// Each kind carries its own default and limits. Kinds beyond Float arrive with their first user.
#[derive(Clone, Debug, PartialEq)]
pub enum ParameterKind {
    Float(FloatParameter),
}

/// A real number, as OpenFX's doubles are. Values are always finite.
#[derive(Clone, Debug, PartialEq)]
pub struct FloatParameter {
    pub default: f64,
    /// The range a slider covers. A typed value may go beyond it, as far as the limits.
    pub slider: RangeInclusive<f64>,
    /// The values the node accepts at all.
    pub limits: RangeInclusive<f64>,
}

#[cfg(test)]
mod tests;
