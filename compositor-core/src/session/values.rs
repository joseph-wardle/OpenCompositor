//! Setting and resetting a node's parameter values, in one step or through an edit in progress.

use std::path::{Component, Path};

use super::{Changes, EditError, EditInProgress, Session};
use crate::author::{NumberType, ValueEdit};
use crate::compile::is_animated;
use crate::history::Inverse;
use crate::network::NodeId;
use crate::parameter::{ParameterKind, Value, ValueError, describe};

impl Session {
    /// Begins an edit of a node's values, finishing any edit in progress first.
    pub fn begin_edit(&mut self, node: NodeId, label: &'static str) -> Result<(), EditError> {
        if self.network().node(node).is_none() {
            return Err(EditError::MissingNode(node));
        }
        self.finish_edit();
        self.editing = Some(EditInProgress {
            label,
            node,
            inverse: Inverse::default(),
        });
        Ok(())
    }

    /// The node an edit in progress belongs to.
    pub fn editing(&self) -> Option<NodeId> {
        self.editing.as_ref().map(|edit| edit.node)
    }

    /// Sets a value of the edited node on the stage, recording no step yet. A value equal to the
    /// default stays set.
    pub fn edit_value(&mut self, parameter: &str, value: Value) -> Result<Changes, EditError> {
        let id = self.editing().ok_or(EditError::NoEditInProgress)?;
        let (edit, _) = self.value_edit(id, parameter, Some(value))?;
        Ok(self.join_edit(edit))
    }

    /// Removes the root layer's value of the edited node's parameter, which then reads as a weaker
    /// layer's value, or its default.
    pub fn edit_reset(&mut self, parameter: &str) -> Result<Changes, EditError> {
        let id = self.editing().ok_or(EditError::NoEditInProgress)?;
        let (edit, _) = self.value_edit(id, parameter, None)?;
        Ok(self.join_edit(edit))
    }

    fn join_edit(&mut self, edit: ValueEdit) -> Changes {
        let inverse = self.author(|layer| edit.author(layer));
        let editing = self.editing.as_mut().expect("an edit is in progress");
        editing.inverse.merge(inverse);
        self.update(Changes::default())
    }

    /// Sets a node's value as one step named after the parameter, as a typed value does. A value
    /// equal to the default stays set.
    pub fn set_value(
        &mut self,
        node: NodeId,
        parameter: &str,
        value: Value,
    ) -> Result<Changes, EditError> {
        let (edit, label) = self.value_edit(node, parameter, Some(value))?;
        Ok(self.commit(label, |layer| edit.author(layer)))
    }

    /// Removes the root layer's value of a node's parameter as one step, which then reads as a
    /// weaker layer's value, or its default.
    pub fn reset_value(&mut self, node: NodeId, parameter: &str) -> Result<Changes, EditError> {
        let (edit, _) = self.value_edit(node, parameter, None)?;
        Ok(self.commit("Reset", |layer| edit.author(layer)))
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
        let described = describe(node.kind.parameters(), parameter).map_err(invalid)?;
        let mut edited = node.clone();
        match value {
            Some(value) if matches!(described.kind, ParameterKind::File) => self
                .as_authored(value)
                .and_then(|value| edited.set_value(parameter, value)),
            Some(value) => edited.set_value(parameter, value),
            None => edited.reset(parameter),
        }
        .map_err(invalid)?;
        let path = self.path(id).clone();
        let attribute = (path.append_property(format!("inputs:{}", described.name)))
            .expect("a parameter's name is a property name");
        // A default set under samples would never be read, and a reset would leave them.
        if (self.stage.attribute(attribute.clone())).is_ok_and(|attribute| is_animated(&attribute))
        {
            return Err(invalid(ValueError::Animated));
        }
        let edit = ValueEdit {
            number: NumberType::of(&self.stage, &attribute),
            node: path,
            parameter: described.name,
            value: edited.authored_value(described.name).cloned(),
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

    /// Ends the edit in progress, restoring the stage from when it began.
    pub fn cancel_edit(&mut self) -> Changes {
        let Some(edit) = self.editing.take() else {
            return Changes::default();
        };
        (edit.inverse.restore(&self.stage)).expect("a recorded state restores");
        self.update(Changes::default())
    }
}
