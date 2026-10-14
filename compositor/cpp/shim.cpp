#include "shim.h"
#include "opencompositor/src/qt.cxxqt.h"

#include <QtCore/QFile>
#include <QtCore/QTimer>
#include <QtGui/QGuiApplication>
#include <QtGui/QKeyEvent>
#include <QtGui/QPlatformSurfaceEvent>
#include <QtGui/QWheelEvent>
#include <QtWidgets/QApplication>
#include <QtWidgets/QFileDialog>
#include <QtWidgets/QMenuBar>
#include <QtWidgets/QStatusBar>

namespace shim {
static int argc = 1;
static char name[] = "opencompositor";
static char *argv[] = {name, nullptr};

void new_application() { new QApplication(argc, argv); }
int exec() { return QApplication::exec(); }
void exit(int code) { QCoreApplication::exit(code); }
QString platform_name() { return QGuiApplication::platformName(); }
QString qt_compiled_version() { return QStringLiteral(QT_VERSION_STR); }
QString qt_runtime_version() { return QString::fromLatin1(qVersion()); }
void single_shot(int ms, rust::Fn<void(size_t)> callback, size_t step) {
    QTimer::singleShot(ms, qApp, [=] { callback(step); });
}

std::unique_ptr<QMainWindow> new_main_window() {
    auto window = std::make_unique<QMainWindow>();
    window->statusBar();
    return window;
}
QMenu *new_menu(QMainWindow *window, const QString &title) {
    return window->menuBar()->addMenu(title);
}
void new_action(QMenu *menu, const QString &text, const QString &shortcut,
                rust::Fn<void(size_t)> triggered, size_t command) {
    auto *action = menu->addAction(text);
    action->setShortcut(QKeySequence(shortcut));
    QObject::connect(action, &QAction::triggered, [=] { triggered(command); });
}
QDockWidget *new_dock(QMainWindow *window, QMenu *menu, const QString &title, int area) {
    auto *dock = new QDockWidget(title, window);
    window->addDockWidget(Qt::DockWidgetArea(area), dock);
    menu->addAction(dock->toggleViewAction());
    return dock;
}
QWidget *container(Pane *pane, QDockWidget *dock) {
    auto *container = QWidget::createWindowContainer(pane, dock);
    dock->setWidget(container);
    return container;
}
void tabify(QMainWindow *window, QDockWidget *first, QDockWidget *second) {
    window->tabifyDockWidget(first, second);
}
void show_status(QMainWindow *window, const QString &text) {
    window->statusBar()->showMessage(text);
}
QString status(QMainWindow *window) { return window->statusBar()->currentMessage(); }
QStringList choose_files(QMainWindow *window, const QString &title, bool several) {
    if (several) return QFileDialog::getOpenFileNames(window, title);
    QString file = QFileDialog::getOpenFileName(window, title);
    return file.isEmpty() ? QStringList() : QStringList(file);
}
QByteArray encode_name(const QString &path) { return QFile::encodeName(path); }
void close_modal_dialog() {
    if (auto *dialog = QApplication::activeModalWidget()) dialog->close();
}

Pane *new_pane(size_t index) { return new Pane(index); }
Waker *new_waker() { return new Waker(); }
void delete_waker(Waker *waker) { delete waker; }

int event_type(const QEvent *event) { return event->type(); }
InputFields input_fields(const QEvent *event) {
    InputFields f{};
    if (auto *e = dynamic_cast<const QSinglePointEvent *>(event)) {
        f.x = e->position().x();
        f.y = e->position().y();
        f.button = e->button();
        f.shift = e->modifiers() & Qt::ShiftModifier;
        f.ctrl = e->modifiers() & Qt::ControlModifier;
    }
    if (auto *e = dynamic_cast<const QWheelEvent *>(event)) f.wheel_y = e->angleDelta().y();
    if (auto *e = dynamic_cast<const QKeyEvent *>(event)) {
        f.key = e->key();
        f.auto_repeat = e->isAutoRepeat();
        f.shift = e->modifiers() & Qt::ShiftModifier;
        f.ctrl = e->modifiers() & Qt::ControlModifier;
    }
    if (auto *e = dynamic_cast<const QPlatformSurfaceEvent *>(event))
        f.surface_about_to_be_destroyed =
            e->surfaceEventType() == QPlatformSurfaceEvent::SurfaceAboutToBeDestroyed;
    return f;
}
size_t wayland_display() {
    auto *app = qGuiApp->nativeInterface<QNativeInterface::QWaylandApplication>();
    return app ? size_t(app->display()) : 0;
}
QWaylandWindow *wayland_window(Pane &pane) { return pane.nativeInterface<QWaylandWindow>(); }
size_t wayland_surface(QWaylandWindow &window) { return size_t(window.surface()); }
size_t x11_display() {
    auto *app = qGuiApp->nativeInterface<QNativeInterface::QX11Application>();
    return app ? size_t(app->display()) : 0;
}
uint64_t x11_window(const Pane &pane) { return pane.winId(); }
} // namespace shim
