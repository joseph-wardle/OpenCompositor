// What cxx cannot bind: one-liners for constructors, statics, templates and casts, and the
// parameter panel, creation menu and viewer strip, plain Qt classes that Rust fills and that send
// its gestures back.
#pragma once
#include <QtCore/QByteArray>
#include <QtCore/QEvent>
#include <QtCore/QString>
#include <QtCore/QStringList>
#include <QtGui/QAction>
#include <QtWidgets/QDockWidget>
#include <QtWidgets/QFrame>
#include <QtWidgets/QLabel>
#include <QtWidgets/QMainWindow>
#include <QtWidgets/QMenu>
#include <QtWidgets/QSlider>
#include <cstdint>
#include <memory>
#include <optional>
#include <vector>
#include <qpa/qplatformwindow_p.h>

#include "rust/cxx.h"

class Pane;
class Waker;
struct InputFields;
struct PanelData;
struct PanelRow;
struct PanelGesture;
enum class PanelGestureKind : uint8_t;
struct TimebarData;
struct TimebarGesture;
enum class TimebarGestureKind : uint8_t;
class QComboBox;
class QDoubleSpinBox;
class QLineEdit;
class QListWidget;
class QListWidgetItem;
class QScrollArea;
class QToolButton;

namespace shim {
using QWaylandWindow = QNativeInterface::Private::QWaylandWindow;

// One node's parameters in rows, shown as Rust sends them, each gesture on them sent back with the
// row's parameter name. Rows are rebuilt only when the node changed, otherwise refreshed in place,
// so that a drag survives the refreshes it causes. Nothing is sent during a refresh.
class ParameterPanel : public QWidget {
  public:
    ParameterPanel(rust::Fn<void(PanelGesture)> gestures, QWidget *parent);
    void show_panel(const PanelData &data);
    // The named row's slider, or null.
    QSlider *slider(const QString &name) const;

  protected:
    bool eventFilter(QObject *watched, QEvent *event) override;

  private:
    struct Row {
        QString name;
        QLabel *label;
        QLineEdit *field;
        // Null for a file.
        QSlider *slider;
        // The field's text as last refreshed, which Escape restores.
        QString shown;
        bool authored;
    };
    void rebuild(const PanelData &data);
    void refresh(Row &row, const PanelRow &data);
    Row *row(const QString &name);
    // The field's text, if the artist changed it.
    void committed(const QString &name);
    // Adds Reset to Default to the menu and pops it up, deleting it once closed.
    void popup_reset_menu(const QString &name, QMenu *menu, const QPoint &at);
    void send(PanelGestureKind kind, const QString &parameter, int tick = 0,
              const QString &text = QString());

    rust::Fn<void(PanelGesture)> gestures;
    bool refreshing = false;
    // The node the rows are of, if any.
    std::optional<uint64_t> node;
    QLabel *title;
    QToolButton *pin;
    QLabel *empty;
    QScrollArea *scroll;
    std::vector<Row> rows;
};

class FrameBar;

// The timeline under the viewer: its frames as a bar to scrub and zoom, ticked and marked as Rust
// sends them, between the range's first and last; under it the rate and how playback keeps up,
// the transport round the frame, and playback's looping and mode and the range. Frames read as numbers or as timecode. Each gesture is sent back;
// nothing is sent while Rust shows the timebar.
class Timebar : public QWidget {
  public:
    Timebar(rust::Fn<void(TimebarGesture)> gestures, const QStringList &rates, QWidget *parent);
    void show_timebar(const TimebarData &data);

  protected:
    bool eventFilter(QObject *watched, QEvent *event) override;

  private:
    // A field that sends what is typed into it as the gesture, if the artist changed it.
    QLineEdit *field(TimebarGestureKind kind, const QString &tooltip);
    QToolButton *button(TimebarGestureKind kind, const QIcon &icon, const QString &tooltip);
    // Shows the frame, first and last as numbers or timecode, sizing their fields to fit.
    void show_frames();
    void send(TimebarGestureKind kind, double frame = 0, const QString &text = QString(),
              bool on = false);

    rust::Fn<void(TimebarGesture)> gestures;
    bool refreshing = false;
    bool timecode = false;
    // As last shown, which the buttons send the other of.
    bool looping_on = false;
    bool real_time = false;
    // The frame, first and last as last shown: as numbers, then as timecode.
    QString texts[3][2];
    FrameBar *bar;
    QComboBox *rate;
    QLabel *status;
    QToolButton *timecode_button;
    QToolButton *play_backward;
    QToolButton *play_forward;
    QLineEdit *frame;
    QToolButton *looping;
    QToolButton *mode;
    QComboBox *range;
    QLineEdit *first;
    QLineEdit *last;
    QIcon play_backward_icon, play_forward_icon, stop_icon;
    QIcon loop_icon, once_icon;
};

// Under the viewer: its timebar and caption, then the readout and the viewer's exposure, which
// sends each value the artist sets in stops. Nothing is sent when Rust shows a value.
class ViewerStrip : public QWidget {
  public:
    ViewerStrip(double limit, double step, rust::Fn<void(double)> exposure_set,
                rust::Fn<void(TimebarGesture)> timebar_gestures, const QStringList &rates,
                QWidget *parent);
    Timebar *timebar;
    QLabel *caption;
    QLabel *readout;
    QDoubleSpinBox *exposure;
};

// The node creation menu: a field over the types, which typing filters. Choosing a type sends
// its index among the labels it was opened with; closing it any other way sends nothing.
class CreationMenu : public QFrame {
  public:
    CreationMenu(rust::Fn<void(size_t)> chosen, QWidget *parent);
    // Opens at the pointer with the labels, the first current.
    void open(const QStringList &labels);

  protected:
    bool eventFilter(QObject *watched, QEvent *event) override;

  private:
    void filter(const QString &text);
    // Makes the next shown item `step` away current, if there is one.
    void step_current(int step);
    void choose(QListWidgetItem *item);

    rust::Fn<void(size_t)> chosen;
    QLineEdit *field;
    QListWidget *list;
};

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
QAction *new_action(QMenu *menu, const QString &text, const QString &shortcut,
                rust::Fn<void(size_t)> triggered, size_t command);
void set_action(QAction *action, const QString &text, bool enabled);
QDockWidget *new_dock(QMainWindow *window, QMenu *menu, const QString &title, int area);
QWidget *container(Pane *pane, QDockWidget *dock);
ViewerStrip *container_with_strip(Pane *pane, QDockWidget *dock, double limit, double step,
                                  rust::Fn<void(double)> exposure_set,
                                  rust::Fn<void(TimebarGesture)> timebar_gestures,
                                  const QStringList &rates);
void show_timebar(ViewerStrip *strip, const TimebarData &data);
QLabel *strip_caption(ViewerStrip *strip);
void set_readout(ViewerStrip *strip, const QString &text);
void show_viewer_exposure(ViewerStrip *strip, double stops);
ParameterPanel *new_parameter_panel(QMainWindow *window, QMenu *menu, QDockWidget *viewer,
                                    QDockWidget *graph, rust::Fn<void(PanelGesture)> send);
void show_panel(ParameterPanel *panel, const PanelData &data);
QDockWidget *panel_dock(ParameterPanel *panel);
QSlider *panel_slider(ParameterPanel *panel, const QString &name);
int slider_span(QSlider *slider);
void drag_slider(QSlider *slider, int type, int pixel);
void press_key(QMainWindow *window, int key, const QString &text);
bool popup_open();
bool modal_open();
CreationMenu *new_creation_menu(QMainWindow *window, rust::Fn<void(size_t)> chosen);
void open_creation_menu(CreationMenu *menu, const QStringList &labels);
void set_caption(QLabel *label, const QString &text);
QString label_text(QLabel *label);
void tabify(QMainWindow *window, QDockWidget *first, QDockWidget *second);
void show_status(QMainWindow *window, const QString &text);
void show_error(QMainWindow *window, const QString &text);
QLabel *new_permanent_status(QMainWindow *window);
void set_permanent_status(QLabel *label, const QString &text);
QStringList choose_files(QMainWindow *window, const QString &title, bool several);
QByteArray encode_name(const QString &path);
QString decode_name(rust::Slice<const uint8_t> name);
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
