//! Setting, keying and resetting a node's parameter values, in one step or through an edit in
//! progress.

use std::path::{Component, Path};

use openusd::sdf;

use super::{Changes, EditError, EditInProgress, Session};
use crate::author::{NumberType, ValueChange, ValueEdit};
use crate::compile::{Animation, animation, takes_keys};
use crate::network::NodeId;
use crate::parameter::{ParameterKind, Value, ValueError, describe, resolve};

/// How a parameter that takes keys stands at the session's frame.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum KeyState {
    /// The same at every frame.
    Constant,
    /// Animated by the project file's own keys, with or without one at this frame.
    Keyed { here: bool },
    /// Animated by another file. A key replaces that animation whole, as the stronger opinion
    /// does in USD.
    Elsewhere,
}

impl Session {
    /// Begins an edit of a node's values, finishing any edit in progress first.
    pub fn begin_edit(&mut self, node: NodeId, label: &'static str) -> Result<(), EditError> {
        if self.network().node(node).is_none() {
            return Err(EditError::MissingNode(node));
        }
        self.finish_edit();
        self.editing = Some(EditInProgress::new(node, label));
        Ok(())
    }

    /// The node an edit in progress belongs to.
    pub fn editing(&self) -> Option<NodeId> {
        self.editing.as_ref().map(|edit| edit.node)
    }

    /// Sets a value of the edited node on the stage, recording no step yet: at every frame, or
    /// for an animated parameter as a key at the session's frame. A value equal to the default
    /// stays set.
    ///
    /// Once the edit has set a key, going to another frame records the edit so far as a step and
    /// carries on, so that each frame's key is undone alone.
    pub fn edit_value(&mut self, parameter: &str, value: Value) -> Result<Changes, EditError> {
        let id = self.editing().ok_or(EditError::NoEditInProgress)?;
        let (edit, _) = self.value_edit(id, parameter, Some(value))?;
        Ok(self.join_edit(edit))
    }

    /// Removes the root layer's value and keys of the edited node's parameter, which then reads
    /// as a weaker layer has it, or as its default.
    pub fn edit_reset(&mut self, parameter: &str) -> Result<Changes, EditError> {
        let id = self.editing().ok_or(EditError::NoEditInProgress)?;
        let (edit, _) = self.value_edit(id, parameter, None)?;
        Ok(self.join_edit(edit))
    }

    fn join_edit(&mut self, edit: ValueEdit) -> Changes {
        let inverse = self.author(|layer| edit.author(layer));
        let editing = self.editing.as_mut().expect("an edit is in progress");
        editing.inverse.merge(inverse);
        editing.keyed |= matches!(edit.change, ValueChange::Key { .. });
        self.update(Changes::default())
    }

    /// Sets a node's value as one step named after the parameter, as a typed value does: at
    /// every frame, or for an animated parameter as a key at the session's frame. A value equal
    /// to the default stays set.
    pub fn set_value(
        &mut self,
        node: NodeId,
        parameter: &str,
        value: Value,
    ) -> Result<Changes, EditError> {
        let (edit, label) = self.value_edit(node, parameter, Some(value))?;
        Ok(self.commit(label, |layer| edit.author(layer)))
    }

    /// Removes the root layer's value and keys of a node's parameter as one step, which then
    /// reads as a weaker layer has it, or as its default.
    pub fn reset_value(&mut self, node: NodeId, parameter: &str) -> Result<Changes, EditError> {
        let (edit, _) = self.value_edit(node, parameter, None)?;
        Ok(self.commit("Reset", |layer| edit.author(layer)))
    }

    /// Sets a key at the session's frame holding the parameter's value there, or removes the
    /// project file's own key at that frame, as one step. The first key leaves the value set for
    /// every frame beneath it, which reads again once the last key is removed.
    pub fn toggle_key(&mut self, node: NodeId, parameter: &str) -> Result<Changes, EditError> {
        let invalid = |error| EditError::InvalidValue {
            node,
            parameter: parameter.to_owned(),
            error,
        };
        let state = self.key_state(node, parameter).map_err(invalid)?;
        let shown = (self.network().node(node)).ok_or(EditError::MissingNode(node))?;
        let described = describe(shown.kind.node_type().parameters, parameter).map_err(invalid)?;
        let attribute = self.parameter_attribute(node, described.name);
        let (label, change) = if state == (KeyState::Keyed { here: true }) {
            ("Remove Key", ValueChange::RemoveKey { frame: self.frame })
        } else {
            let value = (resolve(shown, self.frame).float(described.name))
                .expect("a parameter that takes keys has a value at every frame");
            let frame = self.frame;
            ("Set Key", ValueChange::Key { frame, value })
        };
        let edit = ValueEdit {
            number: NumberType::of(&self.stage, &attribute),
            node: self.path(node).clone(),
            parameter: described.name,
            change,
        };
        Ok(self.commit(label, |layer| edit.author(layer)))
    }

    /// How a node's parameter stands at the session's frame, refused for one that takes no keys
    /// or is animated in a way this version does not read.
    pub fn key_state(&self, node: NodeId, parameter: &str) -> Result<KeyState, ValueError> {
        let shown = (self.network().node(node)).ok_or(ValueError::UnknownParameter)?;
        let described = describe(shown.kind.node_type().parameters, parameter)?;
        if !takes_keys(&shown.kind, described) {
            return Err(ValueError::NoKeys);
        }
        let attribute = self.parameter_attribute(node, described.name);
        match self.animation(&attribute) {
            Animation::None => Ok(KeyState::Constant),
            Animation::Clips | Animation::Spline => Err(ValueError::Animated),
            Animation::Samples => {
                let root = self.stage.root_layer();
                let root = root.borrow();
                // The root layer's opinion is the strongest, so its samples are the ones read.
                let keys =
                    (root.attribute(attribute).ok().flatten()).and_then(|spec| spec.time_samples());
                Ok(match keys {
                    None => KeyState::Elsewhere,
                    Some(keys) => KeyState::Keyed {
                        here: (keys.iter())
                            .any(|&(time, _)| sdf::compare_sample_times(time, self.frame).is_eq()),
                    },
                })
            }
        }
    }

    fn parameter_attribute(&self, node: NodeId, parameter: &str) -> sdf::Path {
        (self
            .path(node)
            .append_property(format!("inputs:{parameter}")))
        .expect("a parameter's name is a property name")
    }

    fn animation(&self, attribute: &sdf::Path) -> Animation {
        (self.stage.attribute(attribute.clone()))
            .map_or(Animation::None, |attribute| animation(&attribute))
    }

    /// The edit that sets a node's value, or removes it for `None`, checked against the
    /// parameter's description, and the parameter's label.
    fn value_edit(
        &self,
        id: NodeId,
        parameter: &str,
        value: Option<Value>,
    ) -> Result<(ValueEdit, &'static str), EditError> {
        let invalid = |error| EditError::InvalidValue {
            node: id,
            parameter: parameter.to_owned(),
            error,
        };
        let node = self.network().node(id).ok_or(EditError::MissingNode(id))?;
        let described = describe(node.kind.node_type().parameters, parameter).map_err(invalid)?;
        let mut edited = node.clone();
        match value {
            Some(value) if matches!(described.kind, ParameterKind::File) => self
                .as_authored(value)
                .and_then(|value| edited.set_value(parameter, value)),
            Some(value) => edited.set_value(parameter, value),
            None => edited.reset(parameter),
        }
        .map_err(invalid)?;
        let attribute = self.parameter_attribute(id, described.name);
        let number = NumberType::of(&self.stage, &attribute);
        let change = match (
            self.animation(&attribute),
            edited.authored_value(described.name),
        ) {
            // A value set beside a spline or clips would never be read, and a reset would leave
            // them.
            (Animation::Clips | Animation::Spline, _) => {
                return Err(invalid(ValueError::Animated));
            }
            (Animation::Samples, _) if !takes_keys(&node.kind, described) => {
                return Err(invalid(ValueError::Animated));
            }
            (_, None) => ValueChange::Clear,
            // A value set for every frame would never be read under the samples.
            (Animation::Samples, Some(&Value::Float(value))) => {
                if !number.is_none_or(|number| number.holds(value)) {
                    return Err(invalid(ValueError::NotInt));
                }
                let frame = self.frame;
                ValueChange::Key { frame, value }
            }
            (_, Some(value)) => ValueChange::Set(value.clone()),
        };
        let edit = ValueEdit {
            number,
            node: self.path(id).clone(),
            parameter: described.name,
            change,
        };
        Ok((edit, described.label))
    }

    /// A value as it is authored for the one the artist set. A file's bare relative path gains `./`, which USD anchors to the
    /// project file alone, never searching for it elsewhere; with no project file to anchor it, a
    /// relative path is refused.
    pub(super) fn as_authored(&self, value: Value) -> Result<Value, ValueError> {
        let Value::File(path) = &value else {
            return Ok(value);
        };
        if path.as_os_str().is_empty() || path.is_absolute() {
            return Ok(value);
        }
        if self.file().is_none() {
            return Err(ValueError::Unanchored);
        }
        match path.components().next() {
            Some(Component::Normal(_)) => Ok(Value::File(Path::new(".").join(path).into())),
            _ => Ok(value),
        }
    }

    /// Records the edit in progress as one step, unless it changed nothing on the whole.
    pub fn finish_edit(&mut self) {
        let Some(edit) = self.editing.take() else {
            return;
        };
        self.record_step(edit.label, edit.inverse, None);
    }

    /// Ends the edit in progress, restoring the stage from when it began, or from the last frame
    /// change if it had set a key by then.
    pub fn cancel_edit(&mut self) -> Changes {
        let Some(edit) = self.editing.take() else {
            return Changes::default();
        };
        (edit.inverse.restore(&self.stage)).expect("a recorded state restores");
        self.update(Changes::default())
    }
}
