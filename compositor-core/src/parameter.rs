//! Parameters: what a node type offers to be set, read by the parameter panel, scripts and
//! validation alike, and the values a node resolves to.

use std::fmt;
use std::ops::RangeInclusive;
use std::path::Path;
use std::sync::Arc;

use crate::document::Node;

/// One parameter of a node type. OpenFX properties beyond these stay in the plugin adapter.
#[derive(Clone, Debug, PartialEq)]
pub struct Parameter {
    /// Unchanged across versions and labels: scripts and authored values address it.
    pub name: &'static str,
    pub label: &'static str,
    pub kind: ParameterKind,
}

/// Each kind carries its own default and limits. Kinds beyond these arrive with their first user.
#[derive(Clone, Debug, PartialEq)]
pub enum ParameterKind {
    Float(FloatParameter),
    /// A file, with no default: until one is set, the node has none, and reports it as missing.
    File,
}

impl ParameterKind {
    /// What an absent value reads as, if anything.
    pub fn default_value(&self) -> Option<Value> {
        match self {
            ParameterKind::Float(float) => Some(Value::Float(float.default)),
            ParameterKind::File => None,
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
#[derive(Clone, Debug, PartialEq)]
pub enum Value {
    Float(f64),
    /// Kept as the artist chose it, never empty. Shared, so cloning a node to move it does not
    /// copy the path. Resolving relative paths, variables or assets belongs between the value and
    /// the read.
    File(Arc<Path>),
}

/// Checks a value against the named parameter's description. Returns the description's own name,
/// which outlives the caller's, with the value as it is stored.
pub(crate) fn validate(
    parameters: &'static [Parameter],
    name: &str,
    value: Value,
) -> Result<(&'static str, Value), ValueError> {
    let parameter = describe(parameters, name)?;
    let value = match (&parameter.kind, value) {
        (ParameterKind::Float(float), Value::Float(value)) => {
            if !value.is_finite() {
                return Err(ValueError::NotFinite);
            }
            if !float.limits.contains(&value) {
                return Err(ValueError::OutOfLimits(float.limits.clone()));
            }
            // Adding zero turns -0 into 0, so equal values are stored as equal bits and a
            // result's key, which compares values, never tells them apart.
            Value::Float(value + 0.0)
        }
        // Whether the file exists is the read's to say.
        (ParameterKind::File, Value::File(path)) if path.as_os_str().is_empty() => {
            return Err(ValueError::EmptyPath);
        }
        (ParameterKind::File, value @ Value::File(_)) => value,
        (ParameterKind::Float(_) | ParameterKind::File, _) => return Err(ValueError::WrongKind),
    };
    Ok((parameter.name, value))
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
    /// The value is of another kind than the parameter, such as a number for a file.
    WrongKind,
    NotFinite,
    OutOfLimits(RangeInclusive<f64>),
    EmptyPath,
}

impl fmt::Display for ValueError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ValueError::UnknownParameter => write!(f, "no such parameter"),
            ValueError::WrongKind => write!(f, "the value is of another kind"),
            ValueError::NotFinite => write!(f, "the value is not a finite number"),
            ValueError::OutOfLimits(limits) => write!(
                f,
                "the value is outside {} to {}",
                limits.start(),
                limits.end()
            ),
            ValueError::EmptyPath => write!(f, "the path is empty"),
        }
    }
}

impl std::error::Error for ValueError {}

/// A node's values as evaluation sees them, in description order: one per parameter that has a
/// value or a default.
#[derive(Clone, Debug, PartialEq)]
pub struct Resolved {
    values: Vec<(&'static str, Value)>,
}

impl Resolved {
    pub fn get(&self, name: &str) -> Option<&Value> {
        self.values
            .iter()
            .find(|(parameter, _)| *parameter == name)
            .map(|(_, value)| value)
    }

    /// The named value if it is a number.
    pub fn float(&self, name: &str) -> Option<f64> {
        match self.get(name)? {
            &Value::Float(value) => Some(value),
            Value::File(_) => None,
        }
    }

    /// The named value if it is a file.
    pub fn file(&self, name: &str) -> Option<&Arc<Path>> {
        match self.get(name)? {
            Value::File(path) => Some(path),
            Value::Float(_) => None,
        }
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
        .filter_map(|parameter| {
            let value = (node.authored_value(parameter.name).cloned())
                .or_else(|| parameter.kind.default_value())?;
            Some((parameter.name, value))
        })
        .collect();
    Resolved { values }
}

#[cfg(test)]
mod tests;
