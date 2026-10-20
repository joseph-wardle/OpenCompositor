//! Parameters: what a node type offers to be set, read by the parameter panel, scripts and
//! validation alike, and the values a node resolves to.

use std::fmt;
use std::ops::RangeInclusive;

use crate::document::Node;

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

impl ParameterKind {
    pub fn default_value(&self) -> Value {
        match self {
            ParameterKind::Float(float) => Value::Float(float.default),
        }
    }
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

/// One parameter's value, of the kind its description names.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Value {
    Float(f64),
}

/// Checks a value against the named parameter's description. Returns the description's own name,
/// which outlives the caller's, with the value as it is stored.
pub(crate) fn validate(
    parameters: &'static [Parameter],
    name: &str,
    value: Value,
) -> Result<(&'static str, Value), ValueError> {
    let parameter = describe(parameters, name)?;
    let (ParameterKind::Float(float), Value::Float(value)) = (&parameter.kind, value);
    if !value.is_finite() {
        return Err(ValueError::NotFinite);
    }
    if !float.limits.contains(&value) {
        return Err(ValueError::OutOfLimits(float.limits.clone()));
    }
    // Adding zero turns -0 into 0, so equal values are stored as equal bits and a result's key,
    // which compares values, never tells them apart.
    Ok((parameter.name, Value::Float(value + 0.0)))
}

/// The named parameter's description.
pub(crate) fn describe(
    parameters: &'static [Parameter],
    name: &str,
) -> Result<&'static Parameter, ValueError> {
    parameters
        .iter()
        .find(|parameter| parameter.name == name)
        .ok_or(ValueError::UnknownParameter)
}

/// Why a value was refused.
#[derive(Clone, Debug, PartialEq)]
pub enum ValueError {
    /// The node's type has no parameter of that name.
    UnknownParameter,
    NotFinite,
    OutOfLimits(RangeInclusive<f64>),
}

impl fmt::Display for ValueError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ValueError::UnknownParameter => write!(f, "no such parameter"),
            ValueError::NotFinite => write!(f, "the value is not a finite number"),
            ValueError::OutOfLimits(limits) => write!(
                f,
                "the value is outside {} to {}",
                limits.start(),
                limits.end()
            ),
        }
    }
}

impl std::error::Error for ValueError {}

/// A node's values as evaluation sees them, one per parameter in description order.
#[derive(Clone, Debug, PartialEq)]
pub struct Resolved {
    values: Vec<(&'static str, Value)>,
}

impl Resolved {
    pub fn get(&self, name: &str) -> Option<Value> {
        self.values
            .iter()
            .find(|(parameter, _)| *parameter == name)
            .map(|&(_, value)| value)
    }
}

/// The boundary between what was authored and what executes: the node's authored values over its
/// version's defaults. Inherited opinions would join here if layered composition arrives.
///
/// Setting a value equal to its default and resetting it resolve alike, though only the first
/// keeps the value authored.
pub fn resolve(node: &Node) -> Resolved {
    let values = node
        .kind
        .parameters()
        .iter()
        .map(|parameter| {
            let value = node
                .authored_value(parameter.name)
                .unwrap_or_else(|| parameter.kind.default_value());
            (parameter.name, value)
        })
        .collect();
    Resolved { values }
}

#[cfg(test)]
mod tests;
