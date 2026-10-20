//! The node creation menu: the types sent to the shim's `CreationMenu`, recently chosen first,
//! and the one chosen dispatched with the anchor the menu opened with.

use std::cell::RefCell;

use cxx_qt_lib::{QList, QString, QStringList};
use opencompositor_core::{Anchor, Event, node_types};

use crate::qt::ffi;

thread_local! {
    /// The open menu's anchor, and its types' names in the order shown.
    static OPEN: RefCell<Option<(Anchor, Vec<&'static str>)>> = const { RefCell::new(None) };
    /// The types chosen this session, the latest first.
    static RECENT: RefCell<Vec<&'static str>> = const { RefCell::new(Vec::new()) };
}

pub fn show(anchor: Anchor) {
    let mut types: Vec<_> = node_types().iter().collect();
    RECENT.with(|cell| {
        let recent = cell.borrow();
        let rank = |name| recent.iter().position(|&recent| recent == name);
        // Stable, so the rest keep the catalog's order.
        types.sort_by_key(|node_type| rank(node_type.name).unwrap_or(usize::MAX));
    });
    let mut labels = QList::<QString>::default();
    for node_type in &types {
        labels.append(QString::from(node_type.label()));
    }
    let names = types.iter().map(|node_type| node_type.name).collect();
    OPEN.with(|cell| *cell.borrow_mut() = Some((anchor, names)));
    // SAFETY: the window owns the menu, and lives until it is destroyed after the event loop.
    unsafe { ffi::open_creation_menu(crate::widgets().creation_menu, &QStringList::from(&labels)) };
}

pub fn chosen(index: usize) {
    let Some((anchor, names)) = OPEN.with(|cell| cell.borrow_mut().take()) else {
        return;
    };
    let type_name = names[index];
    RECENT.with(|cell| {
        let mut recent = cell.borrow_mut();
        recent.retain(|&name| name != type_name);
        recent.insert(0, type_name);
    });
    crate::dispatch(Event::NodeTypeChosen { type_name, anchor });
}
