use std::path::{Path, PathBuf};

use cxx_qt_build::CxxQtBuilder;
use qt_build_utils::QtBuild;

fn main() {
    let private_includes = qt_private_include_dirs();
    let builder = CxxQtBuilder::new()
        .qt_module("Gui")
        .qt_module("Widgets")
        .file("src/qt.rs")
        .cpp_file("cpp/shim.cpp");
    // SAFETY: only adds include paths: the shim, and Qt's private QPA headers for Wayland handles.
    let builder = unsafe {
        builder.cc_builder(|cc| {
            cc.include("cpp").includes(&private_includes);
        })
    };
    builder.build();
    // cxx-qt-build follows the files it is given, not the headers they include.
    println!("cargo::rerun-if-changed=cpp");
}

/// Qt's private headers live under `<QT_INSTALL_HEADERS>/Qt<Module>/<QT_VERSION>/`, a path that
/// changes with every Qt release. The installation comes from the same `QtBuild::new` that
/// cxx-qt-build calls, so `QMAKE` (and its rerun-if-env-changed) selects both identically.
fn qt_private_include_dirs() -> Vec<PathBuf> {
    let qt = QtBuild::new(vec!["Core".into(), "Gui".into()]).expect("no usable Qt installation");
    let version = qt.version().to_string();
    let module_dir = |module: &str| -> PathBuf {
        qt.include_paths()
            .into_iter()
            .find(|path| path.ends_with(module))
            .unwrap_or_else(|| panic!("Qt has no {module} include directory"))
    };
    let (core, gui) = (module_dir("QtCore"), module_dir("QtGui"));

    let private_header = gui.join(&version).join("QtGui/qpa/qplatformwindow_p.h");
    assert!(
        private_header.exists(),
        "Qt {version} lacks private QPA headers ({}); install Qt with its private headers",
        private_header.display()
    );
    // qt-build-utils prints no rerun-if-changed for a qmake installation, so an in-place Qt
    // upgrade would otherwise keep a shim compiled against the old private ABI. qconfig.h
    // holds QT_VERSION_STR, and the versioned private header vanishes on any version change.
    println!(
        "cargo::rerun-if-changed={}",
        core.join("qconfig.h").display()
    );
    println!("cargo::rerun-if-changed={}", private_header.display());

    versioned_dirs(&core, &version)
        .into_iter()
        .chain(versioned_dirs(&gui, &version))
        .collect()
}

fn versioned_dirs(module_dir: &Path, version: &str) -> [PathBuf; 2] {
    let module = module_dir
        .file_name()
        .expect("module include dir has a name");
    let versioned = module_dir.join(version);
    [versioned.join(module), versioned]
}
