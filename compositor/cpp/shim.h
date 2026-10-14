// One-liners for what cxx cannot bind: constructors, statics, templates and casts. No logic.
#pragma once
#include <QtCore/QByteArray>
#include <QtCore/QEvent>
#include <QtCore/QString>
#include <QtCore/QStringList>
#include <QtWidgets/QDockWidget>
#include <QtWidgets/QLabel>
#include <QtWidgets/QMainWindow>
#include <QtWidgets/QMenu>
#include <memory>
#include <qpa/qplatformwindow_p.h>

#include "rust/cxx.h"

class Pane;
class Waker;
struct InputFields;

namespace shim {
using QWaylandWindow = QNativeInterface::Private::QWaylandWindow;

void new_application();
int exec();
void exit(int code);
QString platform_name();
QString qt_compiled_version();
QString qt_runtime_version();
void single_shot(int ms, rust::Fn<void(size_t)> callback, size_t argument);

std::unique_ptr<QMainWindow> new_main_window();
QMenu *new_menu(QMainWindow *window, const QString &title);
QMenu *new_submenu(QMenu *menu, const QString &title);
void add_separator(QMenu *menu);
void add_note(QMenu *menu, const QString &text);
void new_action(QMenu *menu, const QString &text, const QString &shortcut,
                rust::Fn<void(size_t)> triggered, size_t command);
QDockWidget *new_dock(QMainWindow *window, QMenu *menu, const QString &title, int area);
QWidget *container(Pane *pane, QDockWidget *dock);
QLabel *container_with_caption(Pane *pane, QDockWidget *dock);
void set_caption(QLabel *label, const QString &text);
QString caption(QLabel *label);
void tabify(QMainWindow *window, QDockWidget *first, QDockWidget *second);
void show_status(QMainWindow *window, const QString &text);
QStringList choose_files(QMainWindow *window, const QString &title, bool several);
QByteArray encode_name(const QString &path);
void close_modal_dialog();

void filter_keys(rust::Fn<bool(int, InputFields)> handle);
bool keyboard_taken();

Pane *new_pane(size_t index);
Waker *new_waker();
void delete_waker(Waker *waker);

int event_type(const QEvent *event);
InputFields input_fields(const QEvent *event);
size_t wayland_display();
QWaylandWindow *wayland_window(Pane &pane);
size_t wayland_surface(QWaylandWindow &window);
size_t x11_display();
uint64_t x11_window(const Pane &pane);
} // namespace shim
