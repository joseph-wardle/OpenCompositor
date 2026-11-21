//! A node prim's other properties: those its type does not declare, which other tools or hand
//! edits author, shown in the panel and edited or removed one at a time.

use std::fmt;

use openusd::sdf::{self, AttributeSpecMut};
use openusd::usd::Attribute;
use openusd::vt;

use super::{EditError, Session};
use crate::author::{NumberType, OUTPUT, POSITION};
use crate::compile::{input_slot, is_animated};
use crate::network::{Changes, NodeId};
use crate::parameter::describe;

/// A property of a node's prim that is neither one of its parameters or inputs nor one the app
/// writes itself.
#[derive(Clone, Debug, PartialEq)]
pub struct Property {
    /// The full name, `userProperties:shot`.
    pub name: String,
    /// The value type as authored, `string` or `float3[]`, or `rel` for a relationship.
    pub type_name: String,
    /// The value, connections or targets as text.
    pub text: String,
    /// What a value set must be, if [`Session::set_property`] can set one.
    pub editor: Option<ScalarKind>,
    /// An `inputs:` attribute the node's type does not declare, which evaluation ignores.
    pub unused: bool,
    /// Whether the root layer has a spec for it, which [`Session::remove_property`] removes.
    pub removable: bool,
}

/// What a single value set on a property may be.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ScalarKind {
    Bool,
    Integer,
    Real,
    Text,
}

impl ScalarKind {
    /// The value text names, `true`, `0` or `1` for a bool, or `None` for text that is not one.
    pub fn parse(self, text: &str) -> Option<Scalar> {
        let trimmed = text.trim();
        match self {
            ScalarKind::Bool => match trimmed {
                "true" | "1" => Some(Scalar::Bool(true)),
                "false" | "0" => Some(Scalar::Bool(false)),
                _ => None,
            },
            ScalarKind::Integer => trimmed.parse().ok().map(Scalar::Integer),
            ScalarKind::Real => trimmed.parse().ok().map(Scalar::Real),
            ScalarKind::Text => Some(Scalar::Text(text.to_owned())),
        }
    }
}

/// A value set on a property, written in the property's own type.
#[derive(Clone, Debug, PartialEq)]
pub enum Scalar {
    Bool(bool),
    Integer(i64),
    Real(f64),
    Text(String),
}

/// Why a property cannot be set or removed.
#[derive(Clone, Debug, PartialEq)]
pub enum PropertyError {
    /// The node's prim has no such property among its other properties.
    Missing,
    /// It holds connections, targets, time samples, or a type with more than one value.
    NotEditable,
    /// The value is not of the kind the property's type takes.
    WrongKind(ScalarKind),
    /// The value is beyond what the property's type holds, or not finite.
    OutOfRange,
    /// Only another layer authors it.
    NotInRoot,
}

impl fmt::Display for PropertyError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PropertyError::Missing => write!(f, "the node has no such property"),
            PropertyError::NotEditable => write!(
                f,
                "only a single value, not connections, targets or time samples, can be set"
            ),
            PropertyError::WrongKind(kind) => write!(
                f,
                "it takes {}",
                match kind {
                    ScalarKind::Bool => "true or false",
                    ScalarKind::Integer => "a whole number",
                    ScalarKind::Real => "a number",
                    ScalarKind::Text => "text",
                }
            ),
            PropertyError::OutOfRange => write!(f, "its type cannot hold that value"),
            PropertyError::NotInRoot => {
                write!(f, "only another layer authors it; remove it there")
            }
        }
    }
}

impl std::error::Error for PropertyError {}

impl Session {
    /// The node's other properties, sorted by name, or none for a node the network does not have.
    pub fn other_properties(&self, id: NodeId) -> Vec<Property> {
        let Some(node) = self.network().node(id) else {
            return Vec::new();
        };
        let Ok(prim) = self.stage.prim(self.path(id).clone()) else {
            return Vec::new();
        };
        let declared = |name: &str| {
            name == POSITION
                || name == OUTPUT
                || name.strip_prefix("inputs:").is_some_and(|input| {
                    describe(node.kind.node_type().parameters, input).is_ok()
                        || input_slot(&node.kind, input).is_some()
                })
        };
        let removable = |path: &sdf::Path| {
            let root = self.stage.root_layer();
            root.borrow().data().has_spec(path)
        };
        let mut properties = Vec::new();
        for attribute in prim.attributes().unwrap_or_default() {
            if declared(attribute.name()) {
                continue;
            }
            let type_name = (attribute.type_name().ok().flatten())
                .map_or_else(String::new, |type_name| type_name.as_str().to_owned());
            let (text, editor) = shown(&attribute, &type_name);
            properties.push(Property {
                name: attribute.name().to_owned(),
                unused: attribute.name().starts_with("inputs:"),
                removable: removable(attribute.path()),
                type_name,
                text,
                editor,
            });
        }
        for relationship in prim.relationships().unwrap_or_default() {
            let targets = relationship.targets().unwrap_or_default();
            properties.push(Property {
                name: relationship.name().to_owned(),
                type_name: "rel".to_owned(),
                text: paths(&targets),
                editor: None,
                unused: false,
                removable: removable(relationship.path()),
            });
        }
        properties.sort_by(|a, b| a.name.cmp(&b.name));
        properties
    }

    /// Sets one of the node's other properties as one step, writing the value in its type in the
    /// root layer, over whatever weaker layers author.
    pub fn set_property(
        &mut self,
        id: NodeId,
        name: &str,
        value: Scalar,
    ) -> Result<Changes, EditError> {
        let refused = |error| EditError::Property {
            node: id,
            property: name.to_owned(),
            error,
        };
        let property = self.other_property(id, name).map_err(refused)?;
        let kind = property.editor.ok_or(refused(PropertyError::NotEditable))?;
        let value = typed(&property.type_name, kind, value).map_err(refused)?;
        let path =
            (self.path(id).append_property(name)).expect("a property's name is a property name");
        let attribute = (self.stage.attribute(path.clone())).expect("a listed attribute");
        let variability = attribute.variability().ok().flatten().unwrap_or_default();
        let custom = attribute.is_custom().unwrap_or(false);
        let type_name = property.type_name;
        Ok(
            self.commit_with("Set Property", Changes::default(), |layer| {
                if layer.attribute(path.clone()).ok().flatten().is_none() {
                    AttributeSpecMut::new(
                        layer.data_mut(),
                        path.clone(),
                        type_name.as_str(),
                        variability,
                        custom,
                    )?;
                }
                (layer.attribute_mut(path).ok().flatten())
                    .expect("the spec was found or declared")
                    .set_default(value)
            }),
        )
    }

    /// Removes the root layer's spec of one of the node's other properties as one step. What
    /// weaker layers author for it remains.
    pub fn remove_property(&mut self, id: NodeId, name: &str) -> Result<Changes, EditError> {
        let refused = |error| EditError::Property {
            node: id,
            property: name.to_owned(),
            error,
        };
        let property = self.other_property(id, name).map_err(refused)?;
        if !property.removable {
            return Err(refused(PropertyError::NotInRoot));
        }
        let path =
            (self.path(id).append_property(name)).expect("a property's name is a property name");
        Ok(
            self.commit_with("Remove Property", Changes::default(), |layer| {
                layer.remove_spec(&path)?;
                Ok(())
            }),
        )
    }

    /// A parameter's number type as authored, if other than `double`.
    pub(crate) fn number_type(&self, id: NodeId, parameter: &str) -> Option<&'static str> {
        let path = self
            .path(id)
            .append_property(format!("inputs:{parameter}"))
            .ok()?;
        match NumberType::of(&self.stage, &path)? {
            NumberType::Double => None,
            number => Some(number.name()),
        }
    }

    fn other_property(&self, id: NodeId, name: &str) -> Result<Property, PropertyError> {
        if self.network().node(id).is_none() {
            return Err(PropertyError::Missing);
        }
        (self.other_properties(id).into_iter())
            .find(|property| property.name == name)
            .ok_or(PropertyError::Missing)
    }
}

/// An attribute's value as text, and what a value set on it must be if one can be.
fn shown(attribute: &Attribute, type_name: &str) -> (String, Option<ScalarKind>) {
    let connections = attribute.connections().unwrap_or_default();
    if !connections.is_empty() {
        return (format!("→ {}", paths(&connections)), None);
    }
    if is_animated(attribute) {
        return ("animated".to_owned(), None);
    }
    let value = attribute.get::<vt::Value>().ok().flatten();
    let text = match &value {
        None => String::new(),
        Some(vt::Value::Bool(value)) => value.to_string(),
        Some(vt::Value::AssetPath(asset)) => asset.authored_path.to_string(),
        Some(vt::Value::None) => "None".to_owned(),
        Some(value) => value.to_string(),
    };
    // A blocked value can be set again.
    (text, scalar_kind(type_name))
}

fn scalar_kind(type_name: &str) -> Option<ScalarKind> {
    match type_name {
        "bool" => Some(ScalarKind::Bool),
        "uchar" | "int" | "uint" | "int64" | "uint64" => Some(ScalarKind::Integer),
        "half" | "float" | "double" => Some(ScalarKind::Real),
        "string" | "token" | "asset" => Some(ScalarKind::Text),
        _ => None,
    }
}

/// The value in the type named, if it is of the kind the type takes and within its range.
fn typed(type_name: &str, kind: ScalarKind, value: Scalar) -> Result<vt::Value, PropertyError> {
    let out_of_range = |_| PropertyError::OutOfRange;
    Ok(match (type_name, value) {
        ("bool", Scalar::Bool(value)) => vt::Value::Bool(value),
        ("uchar", Scalar::Integer(value)) => {
            vt::Value::Uchar(value.try_into().map_err(out_of_range)?)
        }
        ("int", Scalar::Integer(value)) => vt::Value::Int(value.try_into().map_err(out_of_range)?),
        ("uint", Scalar::Integer(value)) => {
            vt::Value::Uint(value.try_into().map_err(out_of_range)?)
        }
        ("int64", Scalar::Integer(value)) => vt::Value::Int64(value),
        ("uint64", Scalar::Integer(value)) => {
            vt::Value::Uint64(value.try_into().map_err(out_of_range)?)
        }
        ("half" | "float" | "double", Scalar::Real(value)) if !value.is_finite() => {
            return Err(PropertyError::OutOfRange);
        }
        ("half", Scalar::Real(value)) => {
            let half = openusd::gf::f16::from_f64(value);
            if !half.is_finite() {
                return Err(PropertyError::OutOfRange);
            }
            vt::Value::Half(half)
        }
        ("float", Scalar::Real(value)) => {
            let float = value as f32;
            if !float.is_finite() {
                return Err(PropertyError::OutOfRange);
            }
            vt::Value::Float(float)
        }
        ("double", Scalar::Real(value)) => vt::Value::Double(value),
        ("string", Scalar::Text(text)) => vt::Value::String(text),
        ("token", Scalar::Text(text)) => vt::Value::Token(text.as_str().into()),
        ("asset", Scalar::Text(text)) => vt::Value::AssetPath(sdf::AssetPath::new(&text)),
        _ => return Err(PropertyError::WrongKind(kind)),
    })
}

fn paths(paths: &[sdf::Path]) -> String {
    (paths.iter())
        .map(|path| format!("<{}>", path.as_str()))
        .collect::<Vec<_>>()
        .join(", ")
}
