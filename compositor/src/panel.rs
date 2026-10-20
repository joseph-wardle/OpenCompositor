//! The parameter panel's dock: the app's [`Panel`] sent to the shim's `ParameterPanel`, and the
//! gestures it sends back made into session edits.

use std::cell::RefCell;
use std::os::unix::ffi::OsStrExt;
use std::path::Path;
use std::sync::Arc;

use cxx_qt_lib::QString;
use opencompositor_core::{
    Command, EditError, Event, FileIntent, FloatParameter, Panel, Parameter, ParameterKind,
    Request, Row, Session, Value, ValueState,
};

use crate::qt::ffi::{self, PanelGestureKind, RowKind};

thread_local! {
    /// The panel the shim shows. Every gesture is on it, since the shim sends none while it
    /// refreshes.
    static SHOWN: RefCell<Option<Panel>> = const { RefCell::new(None) };
}

/// Shows the panel, or that no node is selected.
pub fn show(panel: Option<Panel>) {
    let rebuild = SHOWN.with(|cell| {
        cell.borrow().as_ref().map(|shown| shown.node) != panel.as_ref().map(|panel| panel.node)
    });
    send(panel.as_ref(), rebuild);
    SHOWN.with(|cell| *cell.borrow_mut() = panel);
}

/// Shows the panel again, putting back what a refused or unchanged edit left in a field.
fn refresh() {
    SHOWN.with(|cell| send(cell.borrow().as_ref(), false));
}

fn send(panel: Option<&Panel>, rebuild: bool) {
    let data = ffi::PanelData {
        shown: panel.is_some(),
        title: QString::from(&panel.map(title).unwrap_or_default()),
        pinned: panel.is_some_and(|panel| panel.pinned),
        rebuild,
        rows: panel.map_or_else(Vec::new, |panel| panel.rows.iter().map(row).collect()),
    };
    // SAFETY: the dock owns the panel, and lives until the main window is destroyed after the
    // event loop.
    unsafe { ffi::show_panel(crate::widgets().panel, &data) };
}

/// The node's name, and its type where the name does not say it.
fn title(panel: &Panel) -> String {
    if panel.name == panel.type_label {
        panel.name.clone()
    } else {
        format!("{} ({})", panel.name, panel.type_label)
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
        (ParameterKind::File, Some(Value::File(path))) => (
            RowKind::File,
            String::new(),
            ffi::decode_name(path.as_os_str().as_bytes()),
            0,
            0,
        ),
        (ParameterKind::File, _) => (RowKind::File, String::new(), QString::default(), 0, 0),
    };
    ffi::PanelRow {
        name: QString::from(parameter.name),
        label: QString::from(parameter.label),
        kind,
        authored: row.state == ValueState::Authored,
        tooltip: QString::from(&tooltip),
        text,
        ticks,
        tick,
    }
}

/// A number as the panel shows it: up to four decimals, without trailing zeros.
fn number(value: f64) -> String {
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
        _ => {}
    }
    let name = String::from(&gesture.parameter);
    let shown = SHOWN.with(|cell| {
        let shown = cell.borrow();
        let panel = shown.as_ref()?;
        let row = panel.rows.iter().find(|row| row.parameter.name == name)?;
        Some((panel.node, row.parameter))
    });
    let Some((node, parameter)) = shown else {
        return;
    };
    let tick_value = |tick| match &parameter.kind {
        ParameterKind::Float(float) => Value::Float(Slider::of(float).value(tick)),
        ParameterKind::File => unreachable!("only a float row has a slider"),
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
        PanelGestureKind::Browse => {
            crate::perform(Request::ChooseFiles(FileIntent::SetFile {
                node,
                parameter: parameter.name,
            }));
            return;
        }
        PanelGestureKind::Typed => {
            // Text that is no number goes in as NaN, which the session refuses as from a script.
            let value = String::from(&gesture.text)
                .trim()
                .parse()
                .unwrap_or(f64::NAN);
            crate::edit(|session| session.set_value(node, parameter.name, Value::Float(value)))
        }
        PanelGestureKind::FileTyped => {
            let path: Arc<Path> = crate::path(&gesture.text).into();
            crate::edit(|session| session.set_value(node, parameter.name, Value::File(path)))
        }
        _ => unreachable!("handled above"),
    };
    match edited {
        Ok(_) => {}
        // The edit was cancelled, or the node deleted, by an undo since the drag began.
        Err(EditError::NoEditInProgress) => {}
        Err(error) => crate::perform(Request::ShowStatus(refusal(parameter, &error))),
    }
    if matches!(
        gesture.kind,
        PanelGestureKind::Typed | PanelGestureKind::FileTyped
    ) {
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
