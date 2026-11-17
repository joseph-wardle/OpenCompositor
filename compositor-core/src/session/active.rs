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

    /// Whether the prim's strongest `active` opinion, apart from the one this session authors on
    /// it in the root layer, deactivates it: one of a sublayer, or across a reference or variant.
    fn inactive_elsewhere(&self, path: &sdf::Path) -> bool {
        let field = sdf::FieldKey::Active.as_str();
        let Ok(prim) = self.stage.prim(path.clone()) else {
            return false;
        };
        (prim.prim_stack().unwrap_or_default().iter())
            .filter(|site| self.is_foreign(site, path))
            .find_map(|site| {
                let layer = site.layer.borrow();
                let data = layer.data();
                let value = data
                    .has_field(&site.path, field)
                    .then(|| data.field(&site.path, field).ok())??;
                Some(matches!(*value, vt::Value::Bool(false)))
            })
            .unwrap_or(false)
    }
}
