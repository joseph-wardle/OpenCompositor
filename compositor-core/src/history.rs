//! Undo and redo as the specs each step touched, each as it stood before the step.
//!
//! A transaction's overlay names exactly the specs it changes, its parents' child lists among
//! them, so recording their earlier states at the commit seam costs what the edit touched, and
//! restoring them writes only what differs.
//!
//! Not the crate's `UndoStage`, which has no redo, and makes one step of several commits only
//! inside one call, where a drag's commits arrive over many events.

use std::cell::RefCell;
use std::collections::BTreeMap;
use std::rc::Rc;

use openusd::sdf::{self, AbstractData};
use openusd::usd::{PendingChange, Stage, StageAuthoringError, StageSink};
use openusd::vt;

/// A spec's type and fields, sorted by name, or `None` where the layer held no spec.
type SpecState = Option<(sdf::SpecType, Vec<(String, vt::Value)>)>;

fn spec_state(data: &dyn AbstractData, path: &sdf::Path) -> SpecState {
    let spec_type = data.spec_type(path)?;
    let mut fields: Vec<(String, vt::Value)> = (data.list_fields(path).unwrap_or_default())
        .into_iter()
        .filter_map(|field| {
            let value = data.field(path, &field).ok()?.into_owned();
            Some((field, value))
        })
        .collect();
    fields.sort_by(|a, b| a.0.cmp(&b.0));
    Some((spec_type, fields))
}

/// Every spec some transactions touched, by layer identifier, each as it stood before the first
/// of them.
#[derive(Debug, Default)]
pub(crate) struct Inverse(BTreeMap<String, BTreeMap<sdf::Path, SpecState>>);

impl Inverse {
    /// What makes the layer `identifier`, now `current`, hold exactly the specs of `wanted`.
    pub(crate) fn matching(
        identifier: &str,
        current: &dyn AbstractData,
        wanted: &dyn AbstractData,
    ) -> Inverse {
        let specs = (current.spec_paths().into_iter())
            .chain(wanted.spec_paths())
            .map(|path| {
                let state = spec_state(wanted, &path);
                (path, state)
            })
            .collect();
        Inverse(BTreeMap::from([(identifier.to_owned(), specs)]))
    }

    /// Folds in what later transactions touched, keeping each spec's earlier state.
    pub(crate) fn merge(&mut self, later: Inverse) {
        for (layer, specs) in later.0 {
            let recorded = self.0.entry(layer).or_default();
            for (path, state) in specs {
                recorded.entry(path).or_insert(state);
            }
        }
    }

    /// Has what was recorded of the layer `from` restore the layer `to`, as when the root layer
    /// is saved under another name.
    pub(crate) fn rename_layer(&mut self, from: &str, to: &str) {
        if let Some(specs) = self.0.remove(from) {
            self.0.insert(to.to_owned(), specs);
        }
    }

    /// Whether every spec stands as recorded, so that the transactions changed nothing in all.
    pub(crate) fn changes_nothing(&self, stage: &Stage) -> bool {
        self.0.iter().all(|(identifier, specs)| {
            let Ok(Some(layer)) = stage.layer(identifier) else {
                return specs.values().all(Option::is_none);
            };
            let layer = layer.borrow();
            (specs.iter()).all(|(path, state)| spec_state(layer.data(), path) == *state)
        })
    }

    /// Writes every recorded spec back as one transaction.
    pub(crate) fn restore(&self, stage: &Stage) -> Result<(), StageAuthoringError> {
        let layers: Vec<&str> = self.0.keys().map(String::as_str).collect();
        stage.batch_edit(&layers, |edits| {
            for (edit, specs) in edits.iter_mut().zip(self.0.values()) {
                let data = edit.data_mut();
                for (path, state) in specs {
                    restore_spec(data, path, state);
                }
            }
            Ok(())
        })?;
        Ok(())
    }
}

/// Makes one spec its recorded state, writing only the fields that differ. Child and property
/// order are fields, so they come back too.
fn restore_spec(data: &mut dyn AbstractData, path: &sdf::Path, state: &SpecState) {
    let current = data.spec_type(path);
    let Some((spec_type, fields)) = state else {
        if current.is_some() {
            data.erase_spec(path);
        }
        return;
    };
    if current != Some(*spec_type) {
        if current.is_some() {
            data.erase_spec(path);
        }
        data.create_spec(path.clone(), *spec_type);
    }
    for field in data.list_fields(path).unwrap_or_default() {
        if !fields.iter().any(|(name, _)| *name == field) {
            data.erase_field(path, &field);
        }
    }
    for (name, value) in fields {
        if data
            .field(path, name)
            .is_ok_and(|current| *current == *value)
        {
            continue;
        }
        data.set_field(path, name, value.clone());
    }
}

/// While armed, records the earlier state of every spec each transaction on the stage touches,
/// in any of its layers.
#[derive(Clone, Debug, Default)]
pub(crate) struct Recorder(Rc<RefCell<Option<Inverse>>>);

impl Recorder {
    pub(crate) fn install(&self, stage: &Stage) {
        stage
            .add_sink(RecorderSink(self.clone()))
            .expect("a new stage is not being edited");
    }

    /// Records the transactions `author` makes, and returns what they touched.
    pub(crate) fn record<T>(&self, author: impl FnOnce() -> T) -> (T, Inverse) {
        let armed = self.0.borrow_mut().replace(Inverse::default());
        assert!(armed.is_none(), "recordings do not nest");
        let result = author();
        let inverse = self.0.borrow_mut().take().unwrap_or_default();
        (result, inverse)
    }
}

struct RecorderSink(Recorder);

impl StageSink for RecorderSink {
    fn before_commit(&self, _stage: &Stage, change: &PendingChange<'_>) {
        let mut armed = self.0.0.borrow_mut();
        let Some(inverse) = armed.as_mut() else {
            return;
        };
        let specs = inverse
            .0
            .entry(change.layer_identifier.to_owned())
            .or_default();
        for (path, _) in change.change_list.entries() {
            (specs.entry(path.clone())).or_insert_with(|| spec_state(change.base, path));
        }
    }
}
