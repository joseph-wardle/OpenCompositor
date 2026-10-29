#include "shim.h"
#include "opencompositor/src/qt.cxxqt.h"

#include <QtCore/QFile>
#include <QtCore/QSignalBlocker>
#include <QtCore/QTimer>
#include <QtGui/QCursor>
#include <QtGui/QGuiApplication>
#include <QtGui/QInputMethodQueryEvent>
#include <QtGui/QKeyEvent>
#include <QtGui/QMouseEvent>
#include <QtGui/QPlatformSurfaceEvent>
#include <QtGui/QScreen>
#include <QtGui/QWheelEvent>
#include <QtWidgets/QApplication>
#include <QtWidgets/QDoubleSpinBox>
#include <QtWidgets/QFileDialog>
#include <QtWidgets/QGridLayout>
#include <QtWidgets/QHBoxLayout>
#include <QtWidgets/QLineEdit>
#include <QtWidgets/QListWidget>
#include <QtWidgets/QMenuBar>
#include <QtWidgets/QProxyStyle>
#include <QtWidgets/QPushButton>
#include <QtWidgets/QScrollArea>
#include <QtWidgets/QStatusBar>
#include <QtWidgets/QStyleOptionSlider>
#include <QtWidgets/QToolButton>
#include <QtWidgets/QVBoxLayout>
#include <algorithm>
#include <functional>

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
QAction *new_action(QMenu *menu, const QString &text, const QString &shortcut,
                    rust::Fn<void(size_t)> triggered, size_t command) {
    auto *action = menu->addAction(text);
    action->setShortcut(QKeySequence(shortcut));
    QObject::connect(action, &QAction::triggered, [=] { triggered(command); });
    return action;
}
void set_action(QAction *action, const QString &text, bool enabled) {
    action->setText(text);
    action->setEnabled(enabled);
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
namespace {
// Stops with their sign, as exposure is read; Ctrl+click resets to 0, as on the panel's sliders.
class ExposureBox : public QDoubleSpinBox {
  public:
    using QDoubleSpinBox::QDoubleSpinBox;

  protected:
    QString textFromValue(double value) const override {
        QString text = QDoubleSpinBox::textFromValue(value);
        return value > 0 ? QStringLiteral("+") + text : text;
    }
    void mousePressEvent(QMouseEvent *event) override {
        if (event->button() == Qt::LeftButton && event->modifiers() & Qt::ControlModifier) {
            setValue(0);
            event->accept();
            return;
        }
        QDoubleSpinBox::mousePressEvent(event);
    }
};
} // namespace

ViewerStrip::ViewerStrip(double limit, double step, rust::Fn<void(double)> exposure_set,
                         QWidget *parent)
    : QWidget(parent) {
    auto *layout = new QVBoxLayout(this);
    layout->setContentsMargins(6, 2, 6, 2);
    layout->setSpacing(2);
    caption = new QLabel;
    // Clipped rather than widening the dock, as is the readout.
    caption->setSizePolicy(QSizePolicy::Ignored, QSizePolicy::Fixed);
    layout->addWidget(caption);
    auto *row = new QHBoxLayout;
    readout = new QLabel;
    readout->setSizePolicy(QSizePolicy::Ignored, QSizePolicy::Fixed);
    readout->setTextInteractionFlags(Qt::TextSelectableByMouse);
    row->addWidget(readout, 1);
    row->addWidget(new QLabel(QStringLiteral("Exposure")));
    exposure = new ExposureBox;
    exposure->setRange(-limit, limit);
    exposure->setSingleStep(step);
    exposure->setDecimals(2);
    exposure->setToolTip(
        QStringLiteral("Stops on the shown values, which the image keeps. E and Shift+E step it "
                       "over the viewer, and Ctrl+E or Ctrl+click resets it."));
    // Typing takes the keyboard from the panes until Enter or a click elsewhere gives it back.
    exposure->setFocusPolicy(Qt::ClickFocus);
    row->addWidget(exposure);
    layout->addLayout(row);
    connect(exposure, &QDoubleSpinBox::valueChanged, this,
            [exposure_set](double stops) { exposure_set(stops); });
    connect(exposure, &QDoubleSpinBox::editingFinished, exposure, &QWidget::clearFocus);
}

ViewerStrip *container_with_strip(Pane *pane, QDockWidget *dock, double limit, double step,
                                  rust::Fn<void(double)> exposure_set) {
    auto *widget = new QWidget(dock);
    auto *layout = new QVBoxLayout(widget);
    layout->setContentsMargins(0, 0, 0, 0);
    layout->setSpacing(0);
    layout->addWidget(QWidget::createWindowContainer(pane, widget), 1);
    auto *strip = new ViewerStrip(limit, step, exposure_set, widget);
    layout->addWidget(strip);
    dock->setWidget(widget);
    return strip;
}
QLabel *strip_caption(ViewerStrip *strip) { return strip->caption; }
void set_readout(ViewerStrip *strip, const QString &text) { strip->readout->setText(text); }
void show_viewer_exposure(ViewerStrip *strip, double stops) {
    const QSignalBlocker blocker(strip->exposure);
    strip->exposure->setValue(stops);
}

namespace {
// A click on the groove moves the handle there, rather than a page towards it.
class AbsoluteSetStyle : public QProxyStyle {
  public:
    int styleHint(StyleHint hint, const QStyleOption *option, const QWidget *widget,
                  QStyleHintReturn *returned) const override {
        if (hint == SH_Slider_AbsoluteSetButtons) return Qt::LeftButton;
        return QProxyStyle::styleHint(hint, option, widget, returned);
    }
};

// A slider edited only by dragging: a press anywhere starts the drag, Escape during it cancels,
// Ctrl+click resets, and keys and the wheel pass to the panel. During a drag no shortcut fires.
class ParameterSlider : public QSlider {
  public:
    explicit ParameterSlider(QWidget *parent) : QSlider(Qt::Horizontal, parent) {
        static auto *style = [] {
            auto *style = new AbsoluteSetStyle;
            style->setParent(qApp);
            return style;
        }();
        setStyle(style);
        setFocusPolicy(Qt::NoFocus);
    }
    std::function<void()> cancelled;
    std::function<void()> reset;

    // How many pixels a drag along the groove covers.
    int span() {
        auto [groove, handle] = rects();
        return groove.width() - handle.width();
    }
    // Where to press or move for the handle to sit `pixel` along the span.
    QPoint point_at(int pixel) {
        auto [groove, handle] = rects();
        return QPoint(groove.x() + pixel + (handle.center() - handle.topLeft()).x(),
                      height() / 2);
    }

  protected:
    void mousePressEvent(QMouseEvent *event) override {
        if (event->button() == Qt::LeftButton && event->modifiers() & Qt::ControlModifier) {
            event->accept();
            reset();
            return;
        }
        QSlider::mousePressEvent(event);
        if (isSliderDown()) take_keys();
    }
    void mouseMoveEvent(QMouseEvent *event) override {
        // After a cancel the button is still held, and the handle stays put.
        if (!isSliderDown()) {
            event->ignore();
            return;
        }
        QSlider::mouseMoveEvent(event);
    }
    void mouseReleaseEvent(QMouseEvent *event) override {
        QSlider::mouseReleaseEvent(event);
        if (!isSliderDown()) release_keys();
    }
    void keyPressEvent(QKeyEvent *event) override {
        if (event->key() == Qt::Key_Escape && isSliderDown()) {
            {
                // Not released: the drag ends without its step.
                QSignalBlocker blocker(this);
                setSliderDown(false);
            }
            release_keys();
            cancelled();
            return;
        }
        event->ignore();
    }
    void wheelEvent(QWheelEvent *event) override { event->ignore(); }
    bool eventFilter(QObject *, QEvent *event) override {
        if (event->type() != QEvent::ShortcutOverride) return false;
        event->accept();
        return true;
    }

  private:
    // The window's key filter takes Escape unless something grabs the keyboard. Qt matches
    // shortcuts before a grab redirects the key, so the slider also claims every shortcut, which
    // would otherwise act mid-drag, as R opening a modal dialog the grab leaves deaf to keys.
    void take_keys() {
        grabKeyboard();
        qApp->installEventFilter(this);
    }
    void release_keys() {
        releaseKeyboard();
        qApp->removeEventFilter(this);
    }
    std::pair<QRect, QRect> rects() {
        QStyleOptionSlider option;
        initStyleOption(&option);
        return {style()->subControlRect(QStyle::CC_Slider, &option, QStyle::SC_SliderGroove, this),
                style()->subControlRect(QStyle::CC_Slider, &option, QStyle::SC_SliderHandle, this)};
    }
};

// Sets refreshing for as long as it lives.
class Refreshing {
  public:
    explicit Refreshing(bool &flag) : flag(flag) { flag = true; }
    ~Refreshing() { flag = false; }

  private:
    bool &flag;
};
} // namespace

ParameterPanel::ParameterPanel(rust::Fn<void(PanelGesture)> gestures, QWidget *parent)
    : QWidget(parent), gestures(gestures) {
    auto *layout = new QVBoxLayout(this);
    auto *header = new QHBoxLayout;
    title = new QLabel;
    title->setSizePolicy(QSizePolicy::Ignored, QSizePolicy::Preferred);
    QFont bold = title->font();
    bold.setBold(true);
    title->setFont(bold);
    pin = new QToolButton;
    pin->setText(QStringLiteral("Pin"));
    pin->setCheckable(true);
    pin->setEnabled(false);
    pin->setToolTip(QStringLiteral("Keep the panel on this node whatever is selected"));
    connect(pin, &QToolButton::toggled, this, [this](bool pinned) {
        send(pinned ? PanelGestureKind::Pin : PanelGestureKind::Unpin, QString());
    });
    header->addWidget(title, 1);
    header->addWidget(pin);
    layout->addLayout(header);
    empty = new QLabel(QStringLiteral("No node selected"));
    empty->setAlignment(Qt::AlignCenter);
    empty->setEnabled(false);
    layout->addWidget(empty, 1);
    scroll = new QScrollArea;
    scroll->setWidgetResizable(true);
    scroll->setFrameShape(QFrame::NoFrame);
    scroll->hide();
    layout->addWidget(scroll, 1);
}

void ParameterPanel::show_panel(const PanelData &data) {
    Refreshing refreshing(this->refreshing);
    pin->setChecked(data.pinned);
    pin->setEnabled(data.shown);
    title->setText(data.title);
    title->setToolTip(data.title);
    empty->setVisible(!data.shown);
    scroll->setVisible(data.shown);
    std::optional<uint64_t> shown = data.shown ? std::optional(data.node) : std::nullopt;
    if (shown != node) {
        node = shown;
        rebuild(data);
    }
    for (size_t i = 0; i < rows.size() && i < data.rows.size(); ++i) refresh(rows[i], data.rows[i]);
}

QSlider *ParameterPanel::slider(const QString &name) const {
    for (const Row &row : rows)
        if (row.name == name) return row.slider;
    return nullptr;
}

void ParameterPanel::rebuild(const PanelData &data) {
    // Later, since a gesture on one of the old rows may be running further up the stack.
    if (QWidget *old = scroll->takeWidget()) {
        old->hide();
        old->deleteLater();
    }
    rows.clear();
    auto *content = new QWidget;
    auto *grid = new QGridLayout(content);
    grid->setColumnStretch(2, 1);
    int index = 0;
    for (const PanelRow &data : data.rows) {
        QString name = data.name;
        auto *label = new QLabel(data.label);
        label->setContextMenuPolicy(Qt::CustomContextMenu);
        connect(label, &QLabel::customContextMenuRequested, this, [=, this](const QPoint &at) {
            popup_reset_menu(name, new QMenu(this), label->mapToGlobal(at));
        });
        auto *field = new QLineEdit;
        field->installEventFilter(this);
        field->setContextMenuPolicy(Qt::CustomContextMenu);
        connect(field, &QLineEdit::customContextMenuRequested, this, [=, this](const QPoint &at) {
            popup_reset_menu(name, field->createStandardContextMenu(), field->mapToGlobal(at));
        });
        connect(field, &QLineEdit::editingFinished, this, [=, this] { committed(name); });
        grid->addWidget(label, index, 0);
        QSlider *row_slider = nullptr;
        if (data.kind == RowKind::Float) {
            field->setFixedWidth(field->fontMetrics().horizontalAdvance(QStringLiteral("-000.0000")) +
                                 12);
            auto *slider = new ParameterSlider(content);
            slider->setRange(0, data.ticks);
            connect(slider, &QSlider::sliderPressed, this,
                    [=, this] { send(PanelGestureKind::DragBegan, name, slider->value()); });
            connect(slider, &QSlider::sliderMoved, this,
                    [=, this](int tick) { send(PanelGestureKind::Dragged, name, tick); });
            connect(slider, &QSlider::sliderReleased, this,
                    [=, this] { send(PanelGestureKind::DragEnded, name); });
            slider->cancelled = [=, this] { send(PanelGestureKind::DragCancelled, name); };
            slider->reset = [=, this] { send(PanelGestureKind::Reset, name); };
            grid->addWidget(field, index, 1);
            grid->addWidget(slider, index, 2);
            row_slider = slider;
        } else {
            field->setPlaceholderText(QStringLiteral("No file"));
            auto *browse = new QPushButton(QStringLiteral("Browse…"));
            // From the event loop: the dialog's nested loop may rebuild the rows, deleting the
            // button under its own click.
            connect(browse, &QPushButton::clicked, this, [=, this] {
                QTimer::singleShot(0, this, [=, this] { send(PanelGestureKind::Browse, name); });
            });
            auto *box = new QHBoxLayout;
            box->addWidget(field, 1);
            box->addWidget(browse);
            grid->addLayout(box, index, 1, 1, 2);
        }
        rows.push_back(Row{name, label, field, row_slider, QString(), false});
        ++index;
    }
    grid->setRowStretch(index, 1);
    scroll->setWidget(content);
}

void ParameterPanel::refresh(Row &row, const PanelRow &data) {
    row.authored = data.authored;
    QFont font = row.label->font();
    font.setBold(data.authored);
    row.label->setFont(font);
    row.label->setToolTip(data.tooltip);
    row.shown = data.text;
    row.field->setText(data.text);
    row.field->setCursorPosition(0);
    if (row.slider && !row.slider->isSliderDown()) {
        QSignalBlocker blocker(row.slider);
        row.slider->setValue(data.tick);
    }
}

ParameterPanel::Row *ParameterPanel::row(const QString &name) {
    for (Row &row : rows)
        if (row.name == name) return &row;
    return nullptr;
}

void ParameterPanel::committed(const QString &name) {
    Row *row = this->row(name);
    if (!row || row->field->text() == row->shown) return;
    send(PanelGestureKind::Typed, name, 0, row->field->text());
}

bool ParameterPanel::eventFilter(QObject *watched, QEvent *event) {
    if (event->type() == QEvent::KeyPress &&
        static_cast<QKeyEvent *>(event)->key() == Qt::Key_Escape) {
        for (Row &row : rows) {
            if (row.field != watched) continue;
            row.field->setText(row.shown);
            row.field->clearFocus();
            return true;
        }
    }
    return QWidget::eventFilter(watched, event);
}

void ParameterPanel::popup_reset_menu(const QString &name, QMenu *menu, const QPoint &at) {
    Row *row = this->row(name);
    if (!menu->isEmpty()) menu->addSeparator();
    QAction *reset = menu->addAction(QStringLiteral("Reset to Default"));
    reset->setEnabled(row && row->authored);
    connect(reset, &QAction::triggered, this, [=, this] { send(PanelGestureKind::Reset, name); });
    menu->setAttribute(Qt::WA_DeleteOnClose);
    menu->popup(at);
}

void ParameterPanel::send(PanelGestureKind kind, const QString &parameter, int tick,
                          const QString &text) {
    if (refreshing) return;
    gestures(PanelGesture{kind, parameter, tick, text});
}

ParameterPanel *new_parameter_panel(QMainWindow *window, QMenu *menu, QDockWidget *viewer,
                                    QDockWidget *graph, rust::Fn<void(PanelGesture)> send) {
    auto *dock = new_dock(window, menu, QStringLiteral("Parameters"), window->dockWidgetArea(viewer));
    auto *panel = new ParameterPanel(send, dock);
    dock->setWidget(panel);
    // Docks in one area keep the sizes they are given. Without a central widget, a dock in an
    // area of its own takes every spare pixel of the window.
    window->splitDockWidget(viewer, dock, Qt::Horizontal);
    window->splitDockWidget(viewer, graph, Qt::Vertical);
    window->resizeDocks({viewer, dock}, {1040, 360}, Qt::Horizontal);
    return panel;
}
void show_panel(ParameterPanel *panel, const PanelData &data) { panel->show_panel(data); }
QDockWidget *panel_dock(ParameterPanel *panel) {
    return static_cast<QDockWidget *>(panel->parentWidget());
}
QSlider *panel_slider(ParameterPanel *panel, const QString &name) { return panel->slider(name); }
int slider_span(QSlider *slider) { return static_cast<ParameterSlider *>(slider)->span(); }
void drag_slider(QSlider *slider, int type, int pixel) {
    // To the window, as the platform delivers the mouse, so that a modal dialog or a popup
    // blocks the drag as it blocks the checks' keys and a user's mouse.
    QPointF at = static_cast<ParameterSlider *>(slider)->point_at(pixel);
    QWidget *window = slider->window();
    QPointF in_window = slider->mapTo(window, at);
    auto button = type == QEvent::MouseMove ? Qt::NoButton : Qt::LeftButton;
    auto buttons = type == QEvent::MouseButtonRelease ? Qt::NoButton : Qt::LeftButton;
    QMouseEvent event(QEvent::Type(type), in_window, in_window, slider->mapToGlobal(at), button,
                      buttons, Qt::NoModifier);
    QCoreApplication::sendEvent(window->windowHandle(), &event);
}
void press_key(QMainWindow *window, int key, const QString &text) {
    // To the window, as the platform delivers keys, so that the key filter sees it first. Qt
    // sends a popup's keys to the popup whichever window receives them.
    for (auto type : {QEvent::KeyPress, QEvent::KeyRelease}) {
        QKeyEvent event(type, key, Qt::NoModifier, text);
        QCoreApplication::sendEvent(window->windowHandle(), &event);
    }
}
bool popup_open() { return QApplication::activePopupWidget(); }
bool modal_open() { return QApplication::activeModalWidget(); }

CreationMenu::CreationMenu(rust::Fn<void(size_t)> chosen, QWidget *parent)
    : QFrame(parent, Qt::Popup), chosen(chosen) {
    setFrameShape(QFrame::StyledPanel);
    auto *layout = new QVBoxLayout(this);
    layout->setContentsMargins(2, 2, 2, 2);
    layout->setSpacing(2);
    field = new QLineEdit;
    field->setPlaceholderText(QStringLiteral("Create a node"));
    field->installEventFilter(this);
    list = new QListWidget;
    list->setFocusPolicy(Qt::NoFocus);
    list->setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
    // The row under the pointer is current, as in a menu, so Enter chooses what is highlighted.
    list->setMouseTracking(true);
    layout->addWidget(field);
    layout->addWidget(list);
    connect(field, &QLineEdit::textChanged, this, &CreationMenu::filter);
    connect(field, &QLineEdit::returnPressed, this, [this] { choose(list->currentItem()); });
    connect(list, &QListWidget::itemClicked, this, &CreationMenu::choose);
    connect(list, &QListWidget::itemEntered, this, [this](QListWidgetItem *item) { list->setCurrentItem(item); });
}

void CreationMenu::open(const QStringList &labels) {
    list->clear();
    for (qsizetype i = 0; i < labels.size(); ++i) {
        auto *item = new QListWidgetItem(labels[i], list);
        item->setData(Qt::UserRole, QVariant::fromValue(size_t(i)));
    }
    field->clear();
    filter(QString());
    int rows = std::min<int>(labels.size(), 12);
    list->setFixedHeight(rows * list->sizeHintForRow(0) + 2 * list->frameWidth());
    setFixedWidth(220);
    adjustSize();
    // Within the screen, as a menu opens.
    QPoint at = QCursor::pos();
    if (QScreen *screen = QGuiApplication::screenAt(at)) {
        QRect available = screen->availableGeometry();
        at.setX(std::clamp(at.x(), available.left(), available.right() - width()));
        at.setY(std::clamp(at.y(), available.top(), available.bottom() - height()));
    }
    move(at);
    show();
    field->setFocus();
}

bool CreationMenu::eventFilter(QObject *watched, QEvent *event) {
    if (watched == field && event->type() == QEvent::KeyPress) {
        switch (static_cast<QKeyEvent *>(event)->key()) {
        case Qt::Key_Up:
            step_current(-1);
            return true;
        case Qt::Key_Down:
            step_current(1);
            return true;
        case Qt::Key_Escape:
            close();
            return true;
        default:
            break;
        }
    }
    return QFrame::eventFilter(watched, event);
}

void CreationMenu::filter(const QString &text) {
    QListWidgetItem *first = nullptr;
    for (int i = 0; i < list->count(); ++i) {
        QListWidgetItem *item = list->item(i);
        bool shown = item->text().contains(text, Qt::CaseInsensitive);
        item->setHidden(!shown);
        if (shown && !first) first = item;
    }
    list->setCurrentItem(first);
}

void CreationMenu::step_current(int step) {
    for (int row = list->currentRow() + step; row >= 0 && row < list->count(); row += step) {
        if (!list->item(row)->isHidden()) {
            list->setCurrentRow(row);
            return;
        }
    }
}

void CreationMenu::choose(QListWidgetItem *item) {
    if (!item || item->isHidden()) return;
    size_t index = item->data(Qt::UserRole).value<size_t>();
    close();
    // From the event loop: a Reader's file dialog runs a nested loop, which should not run inside
    // the key or click that closed the menu.
    QTimer::singleShot(0, this, [this, index] { chosen(index); });
}

CreationMenu *new_creation_menu(QMainWindow *window, rust::Fn<void(size_t)> chosen) {
    return new CreationMenu(chosen, window);
}
void open_creation_menu(CreationMenu *menu, const QStringList &labels) { menu->open(labels); }

void set_caption(QLabel *label, const QString &text) {
    label->setText(text);
    label->setToolTip(text);
}
QString label_text(QLabel *label) { return label->text(); }
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
QLabel *new_permanent_status(QMainWindow *window) {
    auto *label = new QLabel;
    label->hide();
    window->statusBar()->addPermanentWidget(label);
    return label;
}
void set_permanent_status(QLabel *label, const QString &text) {
    label->setText(text);
    label->setVisible(!text.isEmpty());
}
QStringList choose_files(QMainWindow *window, const QString &title, bool several) {
    if (several) return QFileDialog::getOpenFileNames(window, title);
    QString file = QFileDialog::getOpenFileName(window, title);
    return file.isEmpty() ? QStringList() : QStringList(file);
}
QByteArray encode_name(const QString &path) { return QFile::encodeName(path); }
QString decode_name(rust::Slice<const uint8_t> name) {
    return QFile::decodeName(
        QByteArray::fromRawData(reinterpret_cast<const char *>(name.data()), name.size()));
}
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
    if (QApplication::activeModalWidget() || QApplication::activePopupWidget() ||
        QWidget::keyboardGrabber())
        return true;
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
