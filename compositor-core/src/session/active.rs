//! Activating and deactivating nodes, by their prims' `active` opinion in the root layer.

use openusd::sdf;
use openusd::vt;

use super::{Changes, EditError, Session};
use crate::author::author_active;
use crate::network::NodeId;

impl Session {
    /// Activates the nodes as one step if the first is inactive, and deactivates them otherwise.
    /// Deactivating authors `active = false`. Activating removes that opinion, or authors
    /// `active = true` where a weaker layer deactivates the node.
    pub fn toggle_active(&mut self, ids: &[NodeId]) -> Result<Changes, EditError> {
        if let Some(&missing) = ids.iter().find(|&&id| self.network().node(id).is_none()) {
            return Err(EditError::MissingNode(missing));
        }
        let Some(&first) = ids.first() else {
            return Ok(Changes::default());
        };
        let network = self.network();
        let active = !network.node(first).expect("checked").is_active();
        let opinions: Vec<(sdf::Path, Option<bool>)> = (ids.iter())
            .filter(|&&id| network.node(id).expect("checked").is_active() != active)
            .map(|&id| {
                let path = self.path(id).clone();
                let opinion = match active {
                    false => Some(false),
                    true => self.inactive_elsewhere(&path).then_some(true),
                };
                (path, opinion)
            })
            .collect();
        let label = if active { "Activate" } else { "Deactivate" };
        Ok(self.commit(label, |layer| {
            for (path, opinion) in &opinions {
                author_active(layer, path, *opinion)?;
            }
            Ok(())
        }))
    }

    /// Whether the strongest `active` opinion below the root layer deactivates the prim.
    fn inactive_elsewhere(&self, path: &sdf::Path) -> bool {
        let root = self.stage.root_layer().identifier().to_owned();
        let field = sdf::FieldKey::Active.as_str();
        (self.stage.layer_stack().into_iter())
            .filter(|identifier| *identifier != root)
            .filter_map(|identifier| self.stage.layer(&identifier).ok().flatten())
            .find_map(|layer| {
                let layer = layer.borrow();
                let data = layer.data();
                let value = data
                    .has_field(path, field)
                    .then(|| data.field(path, field).ok())??;
                Some(matches!(*value, vt::Value::Bool(false)))
            })
            .unwrap_or(false)
    }
}
