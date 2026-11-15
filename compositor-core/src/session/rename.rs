//! Renaming a node: its prim moved to the new name by the namespace editor, every connection to
//! it following, as one step that keeps the node's ID.

use std::cell::Cell;
use std::fmt;

use openusd::sdf::{self, AbstractData};
use openusd::usd::NamespaceEditor;

use super::{Changes, EditError, Session, Step};
use crate::network::NodeId;

/// Why a node cannot take a name.
#[derive(Clone, Debug, PartialEq)]
pub enum NameError {
    Empty,
    StartsWithDigit,
    /// A character other than an ASCII letter, a digit or `_`, which other USD tools may refuse.
    Character(char),
    /// Another prim has the name.
    Taken,
    /// Another node had the name earlier this session, deleted or renamed since, and undo can
    /// bring it back.
    Used,
    /// A layer other than the root defines the node or connects to it, by its identifier. Save
    /// writes the root layer alone, so the rename would not last.
    InLayer(String),
    /// The namespace editor refused the rename.
    Refused(String),
}

impl fmt::Display for NameError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            NameError::Empty => write!(f, "a name cannot be empty"),
            NameError::StartsWithDigit => write!(f, "a name cannot start with a digit"),
            NameError::Character(character) => write!(
                f,
                "a name holds only letters, digits and _, not {character:?}"
            ),
            NameError::Taken => write!(f, "another node has that name"),
            NameError::Used => write!(
                f,
                "another node had that name earlier this session, and undo can bring it back"
            ),
            NameError::InLayer(layer) => {
                write!(
                    f,
                    "{layer} defines or connects to the node; rename it there"
                )
            }
            NameError::Refused(why) => write!(f, "{why}"),
        }
    }
}

impl std::error::Error for NameError {}

impl Session {
    /// Renames a node as one step, keeping its ID, and moves every connection to it with it.
    /// Refused for a name other USD tools may not read, one another prim has or another node had
    /// this session, and for a node another layer defines or connects to.
    pub fn rename(&mut self, id: NodeId, name: &str) -> Result<Changes, EditError> {
        if self.network().node(id).is_none() {
            return Err(EditError::MissingNode(id));
        }
        let refused = |error| EditError::Name {
            node: id,
            name: name.to_owned(),
            error,
        };
        let from = self.path(id).clone();
        if from.name() == Some(name) {
            return Ok(Changes::default());
        }
        check_name(name).map_err(refused)?;
        let parent = from.parent().expect("a node is under its comp");
        let to = parent.append_path(name).expect("a checked name");
        if self.is_taken(&to) {
            match self.compiler.ids().get(&to) {
                // A name this node had before.
                Some(other) if other == id && !self.is_composed(&to) => {}
                Some(other) if other != id && !self.is_composed(&to) => {
                    return Err(refused(NameError::Used));
                }
                _ => return Err(refused(NameError::Taken)),
            }
        }
        if let Some(layer) = self.held_elsewhere(&from) {
            return Err(refused(NameError::InLayer(layer)));
        }
        self.finish_edit();
        let prim = self
            .stage
            .prim(from.clone())
            .expect("a node's path names a prim");
        let (renamed, inverse) = self.recorder.record(|| {
            let mut editor = NamespaceEditor::new(&self.stage);
            editor.rename_prim(&prim, name)?;
            editor.apply()
        });
        renamed.map_err(|error| refused(NameError::Refused(error.to_string())))?;
        self.compiler.rename(id, &to);
        self.undo.push(Step {
            label: "Rename",
            inverse,
            renamed: Some((id, from)),
        });
        self.redo.clear();
        Ok(self.update(Changes::default()))
    }

    fn is_composed(&self, path: &sdf::Path) -> bool {
        (self.stage.prim(path.clone()).ok()).is_some_and(|prim| prim.is_valid().unwrap_or(true))
    }

    /// The first layer of the root's stack, other than the root, with a spec at or beneath `path`
    /// or a path into it, which the namespace editor would change too.
    pub(super) fn held_elsewhere(&self, path: &sdf::Path) -> Option<String> {
        let root = self.stage.root_layer().identifier().to_owned();
        (self.stage.layer_stack().into_iter())
            .filter(|identifier| *identifier != root)
            .find(|identifier| {
                let Ok(Some(layer)) = self.stage.layer(identifier) else {
                    return false;
                };
                holds(layer.borrow().data(), path)
            })
    }
}

/// Whether `data` has a spec at or beneath `path`, or a field naming a path there.
fn holds(data: &dyn AbstractData, path: &sdf::Path) -> bool {
    let into = |named: &sdf::Path| named.has_prefix(path);
    data.spec_paths().iter().any(|spec| {
        into(spec)
            || (data.list_fields(spec).unwrap_or_default().iter()).any(|field| {
                let found = Cell::new(false);
                if let Ok(value) = data.field(spec, field) {
                    value.filter_map_paths(|named| {
                        found.set(found.get() || into(named));
                        Some(named.clone())
                    });
                }
                found.get()
            })
    })
}

/// Checks a name as a valid identifier, the prim names other USD tools read.
fn check_name(name: &str) -> Result<(), NameError> {
    match name.chars().next() {
        None => return Err(NameError::Empty),
        Some(first) if first.is_ascii_digit() => return Err(NameError::StartsWithDigit),
        Some(_) => {}
    }
    match (name.chars()).find(|&character| !(character.is_ascii_alphanumeric() || character == '_'))
    {
        Some(character) => Err(NameError::Character(character)),
        None => Ok(()),
    }
}
