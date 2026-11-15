//! The form in which nodes are authored on the stage, and the edits that write it.
//!
//! A node is a typed prim under the comp root, its parameters and image inputs `inputs:`
//! attributes, UsdShade style, its position UsdUI's:
//!
//! ```usda
//! def OcMerge "Merge3"
//! {
//!     uniform float2 ui:nodegraph:node:pos = (120, 80)
//!     double inputs:mix = 0.5
//!     opaque inputs:background.connect = </Comp/Reader1.outputs:image>
//!     opaque inputs:layer1.connect = </Comp/Exposure2.outputs:image>
//!     opaque outputs:image
//! }
//! ```
//!
//! An unset parameter has no attribute. A repeating input's slots are `inputs:layer1` and on,
//! keeping their numbers. A known type's output is the catalog's, authored or not.

use open_ng::Point;
use openusd::gf::Vec2f;
use openusd::sdf::{self, AttributeSpecMut, LayerEdit, PrimSpecMut, Specifier, Variability};
use openusd::usd::Stage;
use openusd::vt;

use crate::network::{InputSlot, NodeKind, Timeline};
use crate::parameter::Value;

/// The type of the prim whose children are the graph's nodes.
pub(crate) const COMP_TYPE: &str = "OcComp";
pub(crate) const POSITION: &str = "ui:nodegraph:node:pos";
pub(crate) const OUTPUT: &str = "outputs:image";

/// The attribute an input slot is authored as.
pub(crate) fn input_attribute(slot: InputSlot<'_>) -> String {
    match slot.slot {
        0 => format!("inputs:{}", slot.input),
        number => format!("inputs:{}{number}", slot.input),
    }
}

/// How a parameter's value is written: the type of the attribute it goes to, so that an
/// attribute authored as `float` stays one, or `double` for a new number.
#[derive(Clone, Copy, Debug)]
pub(crate) enum NumberType {
    Double,
    Float,
    Half,
    Int,
}

impl NumberType {
    /// The type an existing attribute keeps, if it is a number.
    pub(crate) fn of(stage: &Stage, attribute: &sdf::Path) -> Option<NumberType> {
        let type_name = stage
            .attribute(attribute.clone())
            .ok()?
            .type_name()
            .ok()??;
        match type_name.as_str() {
            "double" => Some(NumberType::Double),
            "float" => Some(NumberType::Float),
            "half" => Some(NumberType::Half),
            "int" => Some(NumberType::Int),
            _ => None,
        }
    }

    pub(crate) fn name(self) -> &'static str {
        match self {
            NumberType::Double => "double",
            NumberType::Float => "float",
            NumberType::Half => "half",
            NumberType::Int => "int",
        }
    }
}

/// Defines a node's prim with its position and its output.
pub(crate) fn author_node(
    layer: &mut LayerEdit<'_>,
    path: &sdf::Path,
    kind: &NodeKind,
    position: Point,
) -> Result<(), sdf::AuthoringError> {
    PrimSpecMut::new(
        layer.data_mut(),
        path.clone(),
        Specifier::Def,
        kind.type_name(),
    )?;
    author_position(layer, path, position)?;
    let output = path.append_property(OUTPUT)?;
    AttributeSpecMut::new(
        layer.data_mut(),
        output,
        "opaque",
        Variability::Varying,
        false,
    )?;
    Ok(())
}

/// Defines the comp root a project's first node goes under, and makes it the layer's default prim
/// if the layer has none: another tool's default prim decides what references to its file bring
/// in.
pub(crate) fn author_root(
    layer: &mut LayerEdit<'_>,
    root: &sdf::Path,
) -> Result<(), sdf::AuthoringError> {
    PrimSpecMut::new(layer.data_mut(), root.clone(), Specifier::Def, COMP_TYPE)?;
    if layer.default_prim().is_some() {
        return Ok(());
    }
    let name = root.name().expect("the comp root is a prim at the top");
    layer.set_default_prim(name)
}

fn author_position(
    layer: &mut LayerEdit<'_>,
    node: &sdf::Path,
    position: Point,
) -> Result<(), sdf::AuthoringError> {
    let path = node.append_property(POSITION)?;
    let value = vt::Value::Vec2f(Vec2f {
        x: position.x,
        y: position.y,
    });
    attribute_spec(layer, &path, "float2", Variability::Uniform)?.set_default(value)
}

/// Sets a parameter's value, as `number` says for a number.
pub(crate) fn author_value(
    layer: &mut LayerEdit<'_>,
    node: &sdf::Path,
    parameter: &str,
    value: &Value,
    number: Option<NumberType>,
) -> Result<(), sdf::AuthoringError> {
    let path = node.append_property(format!("inputs:{parameter}"))?;
    let (type_name, value) = match value {
        Value::File(file) => {
            let file = file
                .to_str()
                .expect("validation refuses a path that is not Unicode");
            ("asset", vt::Value::AssetPath(sdf::AssetPath::new(file)))
        }
        &Value::Float(number_value) => {
            // An `int` keeps its type only for a whole number it holds.
            let number = match number.unwrap_or(NumberType::Double) {
                NumberType::Int
                    if number_value.fract() != 0.0 || number_value.abs() > f64::from(i32::MAX) =>
                {
                    NumberType::Double
                }
                number => number,
            };
            let value = match number {
                NumberType::Double => vt::Value::Double(number_value),
                NumberType::Float => vt::Value::Float(number_value as f32),
                NumberType::Half => vt::Value::Half(openusd::gf::f16::from_f64(number_value)),
                NumberType::Int => vt::Value::Int(number_value as i32),
            };
            (number.name(), value)
        }
    };
    let mut spec = attribute_spec(layer, &path, type_name, Variability::Varying)?;
    if spec
        .type_name()
        .is_some_and(|authored| authored.as_str() != type_name)
    {
        // The crate refuses a type that disagrees with the default the spec holds.
        spec.clear_default();
        spec.set_type_name(type_name)?;
    }
    spec.set_default(value)
}

/// Removes a property's spec from the layer.
fn remove_property(
    layer: &mut LayerEdit<'_>,
    node: &sdf::Path,
    property: &str,
) -> Result<(), sdf::AuthoringError> {
    layer.remove_spec(&node.append_property(property)?)?;
    Ok(())
}

/// Connects an input's attribute to a node's image, or disconnects it: by removing its spec, or,
/// where a weaker layer connects it, by an empty list, as UsdShade's `DisconnectSource` blocks.
fn author_connection(
    layer: &mut LayerEdit<'_>,
    node: &sdf::Path,
    wire: &Wire,
) -> Result<(), sdf::AuthoringError> {
    let path = node.append_property(&wire.attribute)?;
    let targets = match &wire.source {
        Some(source) => vec![source.append_property(OUTPUT)?],
        None if wire.block => Vec::new(),
        None => return remove_property(layer, node, &wire.attribute),
    };
    attribute_spec(layer, &path, "opaque", Variability::Varying)?
        .edit_connections()
        .set(targets)
}

pub(crate) fn author_range(
    layer: &mut LayerEdit<'_>,
    timeline: Timeline,
) -> Result<(), sdf::AuthoringError> {
    layer.set_start_time_code(timeline.first())?;
    layer.set_end_time_code(timeline.last())
}

pub(crate) fn author_rate(
    layer: &mut LayerEdit<'_>,
    timeline: Timeline,
) -> Result<(), sdf::AuthoringError> {
    layer.set_frames_per_second(timeline.rate())?;
    // Time codes are frames: USD reads a time code per second other than the rate as a retime.
    layer.set_time_codes_per_second(timeline.rate())
}

/// The attribute spec at `path`, declared with the type and variability given if the layer has
/// none, never `custom`.
fn attribute_spec<'a>(
    layer: &'a mut LayerEdit<'_>,
    path: &sdf::Path,
    type_name: &str,
    variability: Variability,
) -> Result<AttributeSpecMut<'a>, sdf::AuthoringError> {
    if layer.attribute(path.clone()).ok().flatten().is_none() {
        AttributeSpecMut::new(
            layer.data_mut(),
            path.clone(),
            type_name,
            variability,
            false,
        )?;
    }
    Ok(layer
        .attribute_mut(path.clone())
        .ok()
        .flatten()
        .expect("the spec was found or declared"))
}

/// Authors a node's `active` opinion, or with `None` removes it.
pub(crate) fn author_active(
    layer: &mut LayerEdit<'_>,
    node: &sdf::Path,
    active: Option<bool>,
) -> Result<(), sdf::AuthoringError> {
    match active {
        Some(active) => PrimSpecMut::over(layer.data_mut(), node.clone())?.set_active(active),
        None => (layer.data_mut()).erase_field(node, sdf::FieldKey::Active.as_str()),
    }
    Ok(())
}

/// One input's connection as a session edit authors it.
#[derive(Debug)]
pub(crate) struct Wire {
    pub(crate) consumer: sdf::Path,
    pub(crate) attribute: String,
    /// `None` disconnects.
    pub(crate) source: Option<sdf::Path>,
    /// Whether a layer other than the edited one connects the input, which a disconnect must
    /// block.
    pub(crate) block: bool,
}

pub(crate) fn author_wires(
    layer: &mut LayerEdit<'_>,
    wires: &[Wire],
) -> Result<(), sdf::AuthoringError> {
    for wire in wires {
        author_connection(layer, &wire.consumer, wire)?;
    }
    Ok(())
}

pub(crate) fn author_moves(
    layer: &mut LayerEdit<'_>,
    moves: &[(sdf::Path, Point)],
) -> Result<(), sdf::AuthoringError> {
    for (node, position) in moves {
        author_position(layer, node, *position)?;
    }
    Ok(())
}

/// A parameter's value as a session edit authors it.
#[derive(Debug)]
pub(crate) struct ValueEdit {
    pub(crate) node: sdf::Path,
    pub(crate) parameter: &'static str,
    /// As validation stores it; `None` removes the value.
    pub(crate) value: Option<Value>,
    /// The type an existing numeric attribute keeps.
    pub(crate) number: Option<NumberType>,
}

impl ValueEdit {
    pub(crate) fn author(&self, layer: &mut LayerEdit<'_>) -> Result<(), sdf::AuthoringError> {
        match &self.value {
            Some(value) => author_value(layer, &self.node, self.parameter, value, self.number),
            None => remove_property(layer, &self.node, &format!("inputs:{}", self.parameter)),
        }
    }
}
