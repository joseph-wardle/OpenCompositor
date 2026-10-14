#include "shim.h"
#include "opencompositor/src/qt.cxxqt.h"

#include <QtCore/QFile>
#include <QtCore/QTimer>
#include <QtGui/QGuiApplication>
#include <QtGui/QInputMethodQueryEvent>
#include <QtGui/QKeyEvent>
#include <QtGui/QPlatformSurfaceEvent>
#include <QtGui/QWheelEvent>
#include <QtWidgets/QApplication>
#include <QtWidgets/QFileDialog>
#include <QtWidgets/QMenuBar>
#include <QtWidgets/QStatusBar>
#include <QtWidgets/QVBoxLayout>

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
void single_shot(int ms, rust::Fn<void(size_t)> callback, size_t argument) {
    QTimer::singleShot(ms, qApp, [=] { callback(argument); });
}

std::unique_ptr<QMainWindow> new_main_window() {
    auto window = std::make_unique<QMainWindow>();
    window->statusBar();
    return window;
}
QMenu *new_menu(QMainWindow *window, const QString &title) {
    return window->menuBar()->addMenu(title);
}
QMenu *new_submenu(QMenu *menu, const QString &title) { return menu->addMenu(title); }
void add_separator(QMenu *menu) { menu->addSeparator(); }
void add_note(QMenu *menu, const QString &text) { menu->addAction(text)->setEnabled(false); }
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
QLabel *container_with_caption(Pane *pane, QDockWidget *dock) {
    auto *widget = new QWidget(dock);
    auto *layout = new QVBoxLayout(widget);
    layout->setContentsMargins(0, 0, 0, 0);
    layout->setSpacing(0);
    layout->addWidget(QWidget::createWindowContainer(pane, widget), 1);
    auto *label = new QLabel(widget);
    // Clipped rather than widening the dock.
    label->setSizePolicy(QSizePolicy::Ignored, QSizePolicy::Fixed);
    label->setContentsMargins(6, 2, 6, 2);
    layout->addWidget(label);
    dock->setWidget(widget);
    return label;
}
void set_caption(QLabel *label, const QString &text) {
    label->setText(text);
    label->setToolTip(text);
}
QString caption(QLabel *label) { return label->text(); }
void tabify(QMainWindow *window, QDockWidget *first, QDockWidget *second) {
    window->tabifyDockWidget(first, second);
}
void show_status(QMainWindow *window, const QString &text) {
    // Long enough to read a sentence. A message answers what was just done, and goes stale.
    window->statusBar()->showMessage(text, 6000);
}
void show_error(QMainWindow *window, const QString &text) {
    window->statusBar()->showMessage(text);
}
QStringList choose_files(QMainWindow *window, const QString &title, bool several) {
    if (several) return QFileDialog::getOpenFileNames(window, title);
    QString file = QFileDialog::getOpenFileName(window, title);
    return file.isEmpty() ? QStringList() : QStringList(file);
}
QByteArray encode_name(const QString &path) { return QFile::encodeName(path); }
void close_modal_dialog() {
    if (auto *dialog = QApplication::activeModalWidget()) dialog->close();
}

namespace {
class KeyFilter : public QObject {
  public:
    explicit KeyFilter(rust::Fn<bool(int, InputFields)> handle) : QObject(qApp), handle(handle) {}
    bool eventFilter(QObject *, QEvent *event) override {
        switch (event->type()) {
        case QEvent::KeyPress:
        case QEvent::KeyRelease:
        case QEvent::ApplicationDeactivate:
            return handle(event->type(), input_fields(event));
        default:
            return false;
        }
    }

  private:
    rust::Fn<bool(int, InputFields)> handle;
};
} // namespace
void filter_keys(rust::Fn<bool(int, InputFields)> handle) {
    qApp->installEventFilter(new KeyFilter(handle));
}
bool keyboard_taken() {
    if (QApplication::activeModalWidget() || QApplication::activePopupWidget()) return true;
    QObject *focus = QGuiApplication::focusObject();
    if (!focus) return false;
    QInputMethodQueryEvent query(Qt::ImEnabled);
    QCoreApplication::sendEvent(focus, &query);
    return query.value(Qt::ImEnabled).toBool();
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
        f.native_scan_code = e->nativeScanCode();
        f.auto_repeat = e->isAutoRepeat();
        f.shift = e->modifiers() & Qt::ShiftModifier;
        f.ctrl = e->modifiers() & Qt::ControlModifier;
        f.alt = e->modifiers() & Qt::AltModifier;
        f.meta = e->modifiers() & Qt::MetaModifier;
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
