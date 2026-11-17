//! The parameter panel's dock: the app's [`Panel`] sent to the shim's `ParameterPanel`, and the
//! gestures it sends back made into session edits.

use std::os::unix::ffi::OsStrExt;
use std::path::Path;
use std::sync::Arc;

use cxx_qt_lib::QString;
use opencompositor_core::{
    Command, EditError, Event, FRAME_LIMIT, FileIntent, FloatParameter, Panel, Parameter,
    ParameterKind, Property, PropertyError, RefusedConnection, Request, Row, Session, Value,
    ValueState,
};

use crate::qt::ffi::{self, PanelGestureKind, RowKind};

/// Shows the app's panel again, putting back what a refused or unchanged edit left in a field.
fn refresh() {
    let panel = crate::read_app(|app| app.panel().cloned());
    show(panel.as_ref());
}

/// Shows the panel, or that no node is selected.
pub fn show(panel: Option<&Panel>) {
    let data = ffi::PanelData {
        shown: panel.is_some(),
        node: panel.map_or(0, |panel| open_ng::NodeId::from(panel.node).0),
        name: QString::from(panel.map_or("", |panel| &panel.name)),
        type_label: QString::from(panel.map_or("", type_label)),
        pinned: panel.is_some_and(|panel| panel.pinned),
        problem: QString::from(
            panel
                .and_then(|panel| panel.problem.as_deref())
                .unwrap_or(""),
        ),
        rows: panel.map_or_else(Vec::new, |panel| {
            (panel.rows.iter().map(row))
                .chain(panel.refused.iter().map(connection_row))
                .chain(
                    panel
                        .properties
                        .iter()
                        .map(|property| property_row(panel, property)),
                )
                .collect()
        }),
    };
    // SAFETY: the dock owns the panel, and lives until the main window is destroyed after the
    // event loop.
    unsafe { ffi::show_panel(crate::widgets().panel, &data) };
}

/// The node's type, where its name does not say it.
fn type_label(panel: &Panel) -> &str {
    if panel.name == panel.type_label {
        ""
    } else {
        panel.type_label
    }
}

fn row(row: &Row) -> ffi::PanelRow {
    let parameter = row.parameter;
    let (kind, tooltip, text, ticks, tick) = match (&parameter.kind, &row.value) {
        (ParameterKind::Float(float), value) => {
            let value = match value {
                Some(Value::Float(value)) => *value,
                _ => float.default,
            };
            let slider = Slider::of(float);
            let tooltip = format!("Default: {}", number(float.default));
            (
                RowKind::Float,
                tooltip,
                QString::from(&number(value)),
                slider.ticks,
                slider.tick(value),
            )
        }
        (ParameterKind::Frame { default }, value) => {
            let text = match value {
                Some(Value::Float(frame)) => number(*frame),
                _ => String::new(),
            };
            let tooltip = match default {
                Some(default) => format!("Default: {}", number(*default)),
                None => "Unset, it follows what the node reads".to_owned(),
            };
            (RowKind::Frame, tooltip, QString::from(&text), 0, 0)
        }
        (ParameterKind::File, Some(Value::File(path))) => (
            RowKind::File,
            String::new(),
            ffi::decode_name(path.as_os_str().as_bytes()),
            0,
            0,
        ),
        (ParameterKind::File, _) => (RowKind::File, String::new(), QString::default(), 0, 0),
    };
    let (label, tooltip) = match row.number {
        Some(number) => (
            format!("{} ({number})", parameter.label),
            format!("{tooltip}\nAuthored as {number}, and set so")
                .trim()
                .to_owned(),
        ),
        None => (parameter.label.to_owned(), tooltip),
    };
    ffi::PanelRow {
        name: QString::from(parameter.name),
        label: QString::from(&label),
        kind,
        authored: row.state == ValueState::Authored,
        tooltip: QString::from(&tooltip),
        text,
        ticks,
        tick,
        editable: true,
        unused: false,
    }
}

fn property_row(panel: &Panel, property: &Property) -> ffi::PanelRow {
    let mut tooltip = format!("{} {}", property.type_name, property.name);
    if property.unused {
        tooltip += &format!("\nNot an input of {}, so unused", panel.type_label);
    }
    if !property.removable {
        tooltip += "\nAuthored by another layer";
    }
    ffi::PanelRow {
        name: QString::from(&property.name),
        label: QString::from(&property.name),
        kind: RowKind::Property,
        authored: property.removable,
        tooltip: QString::from(&tooltip),
        text: QString::from(&property.text),
        ticks: 0,
        tick: 0,
        editable: property.editor.is_some(),
        unused: property.unused,
    }
}

/// The row's name, which no parameter or other property has.
fn connection_name(refused: &RefusedConnection) -> String {
    format!("inputs:{}", refused.input)
}

fn connection_row(refused: &RefusedConnection) -> ffi::PanelRow {
    let targets: Vec<String> = (refused.targets.iter())
        .map(|target| format!("<{target}>"))
        .collect();
    ffi::PanelRow {
        name: QString::from(&connection_name(refused)),
        label: QString::from(&refused.label),
        kind: RowKind::Connection,
        authored: true,
        tooltip: QString::from(&format!(
            "Connected to {}, but {}, so unconnected",
            targets.join(", "),
            refused.why
        )),
        text: QString::from(&format!("→ {}: {}", targets.join(", "), refused.why)),
        ticks: 0,
        tick: 0,
        editable: false,
        unused: false,
    }
}

/// A number as the panel shows it: up to four decimals, without trailing zeros.
pub fn number(value: f64) -> String {
    let text = format!("{value:.4}");
    let text = text.trim_end_matches('0').trim_end_matches('.');
    match text {
        "-0" => "0".to_owned(),
        text => text.to_owned(),
    }
}

/// A float slider's positions: its range in steps of the power of ten nearest a thousandth of it.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Slider {
    /// The first position's value, and each step's, in steps.
    start: i64,
    /// The step is ten to this power.
    exponent: i32,
    ticks: i32,
}

impl Slider {
    fn of(float: &FloatParameter) -> Slider {
        let (start, end) = (*float.slider.start(), *float.slider.end());
        let exponent = ((end - start) / 1000.0).log10().round() as i32;
        let step = 10f64.powi(exponent);
        let start = (start / step).round() as i64;
        let ticks = ((end / step).round() as i64 - start) as i32;
        Slider {
            start,
            exponent,
            ticks,
        }
    }

    /// The position's value, the nearest double to its decimal.
    fn value(self, tick: i32) -> f64 {
        let steps = (self.start + i64::from(tick)) as f64;
        match self.exponent {
            // Dividing by an exact power of ten rounds once, where multiplying by an inexact
            // step would not.
            exponent if exponent < 0 => steps / 10f64.powi(-exponent),
            exponent => steps * 10f64.powi(exponent),
        }
    }

    /// The position nearest the value, at the ends beyond them.
    fn tick(self, value: f64) -> i32 {
        let steps = (value / 10f64.powi(self.exponent)).round() as i64 - self.start;
        steps.clamp(0, i64::from(self.ticks)) as i32
    }
}

/// Edits the session for what the artist did to the panel.
pub fn gesture(gesture: ffi::PanelGesture) {
    match gesture.kind {
        PanelGestureKind::Pin => return crate::dispatch(Event::Command(Command::PinPanel)),
        PanelGestureKind::Unpin => return crate::dispatch(Event::Command(Command::UnpinPanel)),
        PanelGestureKind::Renamed => return rename(String::from(&gesture.text).trim()),
        _ => {}
    }
    // The gesture is on the panel last shown, since the shim sends none while it refreshes.
    let name = String::from(&gesture.parameter);
    let property = crate::read_app(|app| {
        let panel = app.panel()?;
        let property = panel
            .properties
            .iter()
            .find(|property| property.name == name)?;
        Some((panel.node, property.clone()))
    });
    if let Some((node, property)) = property {
        return property_gesture(&gesture, node, &property);
    }
    let refused = crate::read_app(|app| {
        let panel = app.panel()?;
        let refused = (panel.refused.iter()).find(|refused| connection_name(refused) == name)?;
        Some((panel.node, refused.clone()))
    });
    if let Some((node, refused)) = refused {
        if gesture.kind == PanelGestureKind::Reset
            && let Err(error) = crate::edit(|session| session.disconnect(node, refused.input))
        {
            crate::perform(Request::ShowStatus(format!(
                "Cannot disconnect {}: {error}",
                refused.label
            )));
        }
        return refresh();
    }
    let shown = crate::read_app(|app| {
        let panel = app.panel()?;
        let row = panel.rows.iter().find(|row| row.parameter.name == name)?;
        Some((panel.node, row.parameter))
    });
    let Some((node, parameter)) = shown else {
        return;
    };
    let tick_value = |tick| match &parameter.kind {
        ParameterKind::Float(float) => Value::Float(Slider::of(float).value(tick)),
        ParameterKind::Frame { .. } | ParameterKind::File => {
            unreachable!("only a float row has a slider")
        }
    };
    let edited = match gesture.kind {
        PanelGestureKind::DragBegan => {
            let value = tick_value(gesture.tick);
            crate::edit(|session| {
                session.begin_edit(node, parameter.label)?;
                session.edit_value(parameter.name, value)
            })
        }
        PanelGestureKind::Dragged => {
            let value = tick_value(gesture.tick);
            crate::edit(|session| session.edit_value(parameter.name, value))
        }
        PanelGestureKind::DragEnded => {
            crate::edit(Session::finish_edit);
            Ok(Default::default())
        }
        PanelGestureKind::DragCancelled => Ok(crate::edit(Session::cancel_edit)),
        PanelGestureKind::Reset => crate::edit(|session| session.reset_value(node, parameter.name)),
        PanelGestureKind::Reload => return crate::dispatch(Event::ReloadFiles(node)),
        PanelGestureKind::Browse => {
            crate::perform(Request::ChooseFiles(FileIntent::SetFile {
                node,
                parameter: parameter.name,
            }));
            return;
        }
        PanelGestureKind::Typed => match &parameter.kind {
            // An emptied field unsets the value, so that it follows what the node reads again.
            ParameterKind::Frame { .. } if String::from(&gesture.text).trim().is_empty() => {
                crate::edit(|session| session.reset_value(node, parameter.name))
            }
            ParameterKind::Float(_) | ParameterKind::Frame { .. } => {
                // Text that is no number goes in as NaN, which the session refuses as from a
                // script.
                let value = String::from(&gesture.text)
                    .trim()
                    .parse()
                    .unwrap_or(f64::NAN);
                crate::edit(|session| session.set_value(node, parameter.name, Value::Float(value)))
            }
            // An emptied field unsets the file, as its placeholder says.
            ParameterKind::File if String::from(&gesture.text).trim().is_empty() => {
                crate::edit(|session| session.reset_value(node, parameter.name))
            }
            ParameterKind::File => {
                let path: Arc<Path> = crate::path(&gesture.text).into();
                crate::edit(|session| session.set_value(node, parameter.name, Value::File(path)))
            }
        },
        _ => unreachable!("handled above"),
    };
    match edited {
        Ok(_) => {}
        // The edit was cancelled, or the node deleted, by an undo since the drag began.
        Err(EditError::NoEditInProgress) => {}
        Err(error) => crate::perform(Request::ShowStatus(refusal(parameter, &error))),
    }
    // A field keeps what was typed or dragged into it until shown again, which only a change to
    // the network does; after an edit the session refused, or a drag an undo or a script cut
    // short, the panel must catch up with the network.
    if matches!(
        gesture.kind,
        PanelGestureKind::Typed | PanelGestureKind::DragEnded | PanelGestureKind::DragCancelled
    ) {
        refresh();
    }
}

/// Sets or removes one of the node's other properties, refusing text its type does not read.
fn property_gesture(
    gesture: &ffi::PanelGesture,
    node: opencompositor_core::NodeId,
    property: &Property,
) {
    let edited = match gesture.kind {
        PanelGestureKind::Reset => {
            crate::edit(|session| session.remove_property(node, &property.name))
        }
        PanelGestureKind::Typed => {
            let Some(kind) = property.editor else {
                return refresh();
            };
            match kind.parse(&String::from(&gesture.text)) {
                Some(value) => {
                    crate::edit(|session| session.set_property(node, &property.name, value))
                }
                None => Err(EditError::Property {
                    node,
                    property: property.name.clone(),
                    error: PropertyError::WrongKind(kind),
                }),
            }
        }
        _ => return,
    };
    if let Err(error) = edited {
        let why = match &error {
            EditError::Property { error, .. } => error.to_string(),
            error => error.to_string(),
        };
        let verb = match gesture.kind {
            PanelGestureKind::Reset => "remove",
            _ => "set",
        };
        crate::perform(Request::ShowStatus(format!(
            "Cannot {verb} {}: {why}",
            property.name
        )));
    }
    refresh();
}

/// Renames the shown node, putting its name back in the field if the session refuses.
fn rename(name: &str) {
    let Some((node, old)) = crate::read_app(|app| {
        let panel = app.panel()?;
        Some((panel.node, panel.name.clone()))
    }) else {
        return;
    };
    if let Err(error) = crate::edit(|session| session.rename(node, name)) {
        let why = match &error {
            EditError::Name { error, .. } => error.to_string(),
            error => error.to_string(),
        };
        crate::perform(Request::ShowStatus(format!(
            "Cannot rename \"{old}\" to \"{name}\": {why}"
        )));
        refresh();
    }
}

/// Why the panel refused an edit. A typed number the parameter cannot take names its limits.
fn refusal(parameter: &Parameter, error: &EditError) -> String {
    match (&parameter.kind, error) {
        (ParameterKind::Float(float), EditError::InvalidValue { .. }) => format!(
            "{} takes numbers from {} to {}",
            parameter.label,
            number(*float.limits.start()),
            number(*float.limits.end())
        ),
        (ParameterKind::Frame { .. }, EditError::InvalidValue { .. }) => format!(
            "{} takes whole frames from {} to {}",
            parameter.label,
            number(-FRAME_LIMIT),
            number(FRAME_LIMIT)
        ),
        (_, EditError::InvalidValue { error, .. }) => {
            format!("Cannot change {}: {error}", parameter.label)
        }
        _ => format!("Cannot change {}: {error}", parameter.label),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn numbers_show_up_to_four_decimals_without_trailing_zeros() {
        let shown = [0.0, -0.0, 1.0, -2.5, 0.123456, 10.0, -0.00001].map(number);
        assert_eq!(shown, ["0", "0", "1", "-2.5", "0.1235", "10", "0"]);
    }

    #[test]
    fn a_slider_steps_by_the_power_of_ten_nearest_a_thousandth_of_its_range() {
        let float = |slider| FloatParameter {
            default: 0.0,
            slider,
            limits: -100.0..=100.0,
        };
        let stops = Slider::of(&float(-6.0..=6.0));
        assert_eq!((stops.exponent, stops.ticks), (-2, 1200));
        assert_eq!(
            [0, 600, 1077, 1200].map(|tick| stops.value(tick)),
            [-6.0, 0.0, 4.77, 6.0]
        );
        assert_eq!(
            [-6.0, 4.77, 100.0, -7.0].map(|value| stops.tick(value)),
            [0, 1077, 1200, 0]
        );
        let wide = Slider::of(&float(0.0..=50_000.0));
        assert_eq!((wide.exponent, wide.ticks, wide.value(3)), (2, 500, 300.0));
    }
}
