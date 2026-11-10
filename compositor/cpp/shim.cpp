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
#include <QtGui/QPainter>
#include <QtGui/QPainterPath>
#include <QtGui/QTransform>
#include <QtGui/QPlatformSurfaceEvent>
#include <QtGui/QScreen>
#include <QtGui/QWheelEvent>
#include <QtWidgets/QApplication>
#include <QtWidgets/QComboBox>
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
#include <QtWidgets/QStyle>
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
// Only typing keeps the keyboard from the panes: Qt gives the box focus before it delivers a
// press, so a reset or an arrow click gives it back.
class ExposureBox : public QDoubleSpinBox {
  public:
    ExposureBox() {
        // A press on the text goes to the line edit rather than to mousePressEvent.
        lineEdit()->installEventFilter(this);
    }

  protected:
    QString textFromValue(double value) const override {
        QString text = QDoubleSpinBox::textFromValue(value);
        return value > 0 ? QStringLiteral("+") + text : text;
    }
    bool eventFilter(QObject *watched, QEvent *event) override {
        if (event->type() == QEvent::MouseButtonPress &&
            reset_by(static_cast<QMouseEvent *>(event))) {
            return true;
        }
        return QDoubleSpinBox::eventFilter(watched, event);
    }
    void mousePressEvent(QMouseEvent *event) override {
        if (!reset_by(event)) QDoubleSpinBox::mousePressEvent(event);
    }
    // On release, since clearing focus on the press would stop a held arrow repeating.
    void mouseReleaseEvent(QMouseEvent *event) override {
        QDoubleSpinBox::mouseReleaseEvent(event);
        clearFocus();
    }

  private:
    bool reset_by(QMouseEvent *event) {
        if (event->button() != Qt::LeftButton || !(event->modifiers() & Qt::ControlModifier)) {
            return false;
        }
        setValue(0);
        clearFocus();
        return true;
    }
};
} // namespace

namespace {
// The timebar's buttons' pictures, drawn rather than taken from the icon theme, which differs
// between desktops and lacks some.
enum class Glyph {
    FirstFrame,
    StepBackward,
    PlayBackward,
    Stop,
    PlayForward,
    StepForward,
    LastFrame,
    Loop,
    Once,
    RealTime,
};

QIcon glyph_icon(Glyph glyph, const QColor &color) {
    // Drawn on a 16 by 16 grid, at twice the pixels for high-density screens.
    constexpr int SCALE = 2;
    QPixmap pixmap(16 * SCALE, 16 * SCALE);
    pixmap.fill(Qt::transparent);
    pixmap.setDevicePixelRatio(SCALE);
    QPainter painter(&pixmap);
    painter.setRenderHint(QPainter::Antialiasing);
    painter.setPen(Qt::NoPen);
    painter.setBrush(color);
    auto triangle = [&](double left, double right, double top, double bottom) {
        // Points towards `right`, which may be left of `left`.
        const QPointF points[] = {{left, top}, {right, (top + bottom) / 2}, {left, bottom}};
        painter.drawPolygon(points, 3);
    };
    auto bar = [&](double x) { painter.drawRect(QRectF(x, 3.5, 1.6, 9)); };
    const QPen line(color, 1.6, Qt::SolidLine, Qt::RoundCap, Qt::RoundJoin);
    switch (glyph) {
    case Glyph::FirstFrame:
        bar(3);
        triangle(12, 5, 3.5, 12.5);
        break;
    case Glyph::StepBackward:
        triangle(10, 4, 4.5, 11.5);
        bar(11);
        break;
    case Glyph::PlayBackward:
        triangle(12.5, 3.5, 3, 13);
        break;
    case Glyph::Stop:
        painter.drawRect(QRectF(4, 4, 8, 8));
        break;
    case Glyph::PlayForward:
        triangle(3.5, 12.5, 3, 13);
        break;
    case Glyph::StepForward:
        bar(3.4);
        triangle(6, 12, 4.5, 11.5);
        break;
    case Glyph::LastFrame:
        triangle(4, 11, 3.5, 12.5);
        bar(11.4);
        break;
    case Glyph::Loop: {
        painter.setPen(line);
        painter.setBrush(Qt::NoBrush);
        QPainterPath top;
        top.moveTo(3, 9.5);
        top.lineTo(3, 7);
        top.quadTo(3, 5, 5, 5);
        top.lineTo(10.5, 5);
        QPainterPath bottom;
        bottom.moveTo(13, 6.5);
        bottom.lineTo(13, 9);
        bottom.quadTo(13, 11, 11, 11);
        bottom.lineTo(5.5, 11);
        painter.drawPath(top);
        painter.drawPath(bottom);
        painter.setPen(Qt::NoPen);
        painter.setBrush(color);
        triangle(10, 14, 2.4, 7.6);
        triangle(6, 2, 8.4, 13.6);
        break;
    }
    case Glyph::Once:
        painter.setPen(line);
        painter.drawLine(QPointF(2.5, 8), QPointF(9, 8));
        painter.setPen(Qt::NoPen);
        triangle(8.5, 12, 5, 11);
        bar(12.6);
        break;
    case Glyph::RealTime:
        painter.setPen(QPen(color, 1.4));
        painter.setBrush(Qt::NoBrush);
        painter.drawEllipse(QPointF(8, 8), 6.2, 6.2);
        painter.setPen(line);
        painter.drawLine(QPointF(8, 8), QPointF(8, 3.8));
        painter.drawLine(QPointF(8, 8), QPointF(11, 8));
        break;
    }
    painter.end();
    return QIcon(pixmap);
}
} // namespace

// The range's whole frames side by side. A press or a drag sends the frame under the pointer, once
// per frame it passes. The view is the bar's own, kept within the range.
class FrameBar : public QWidget {
  public:
    explicit FrameBar(std::function<void(double)> scrubbed) : scrubbed(std::move(scrubbed)) {
        setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Fixed);
        setToolTip(QStringLiteral(
            "Click or drag to go to a frame. The wheel zooms; a middle drag or Shift with the "
            "wheel scrolls. Green below: in the viewer cache. Red: a missing file. Shaded: "
            "outside a sequence's frames, holding its nearest."));
    }
    void show_frames(const TimebarData &data) {
        if (data.first != first || data.last != last) {
            first = data.first;
            last = data.last;
            fit();
        }
        if (data.frame != frame) {
            frame = data.frame;
            follow();
        }
        rate = data.rate;
        timecode = data.timecode;
        marks.assign(data.marks.begin(), data.marks.end());
        update();
    }
    QSize sizeHint() const override { return {200, fontMetrics().height() + 14}; }

  protected:
    void paintEvent(QPaintEvent *) override {
        QPainter painter(this);
        const double h = height();
        painter.fillRect(rect(), palette().color(QPalette::Base));
        const auto [from, to] = visible();
        for (int64_t index = from; index <= to && index < int64_t(marks.size()); ++index) {
            const double left = x_of(index), right = std::max(x_of(index + 1), left + 1);
            const FrameMark &mark = marks[size_t(index)];
            if (mark.held)
                painter.fillRect(QRectF(left, 0, right - left, h), palette().alternateBase());
            if (mark.missing)
                painter.fillRect(QRectF(left, 0, right - left, h), QColor(190, 60, 60, 150));
            if (mark.cached)
                painter.fillRect(QRectF(left, h - 3, right - left, 3), QColor(80, 170, 90));
        }
        paint_ticks(painter);
        if (span < count()) {
            // Zoomed: where the view lies in the range, along the top.
            QColor color = palette().color(QPalette::Highlight);
            color.setAlphaF(0.6);
            painter.fillRect(QRectF(offset / count() * width(), 0, span / count() * width(), 2),
                             color);
        }
        const double x = x_of(frame - first + 0.5);
        painter.setPen(QPen(palette().color(QPalette::Highlight), 2));
        painter.drawLine(QPointF(x, 0), QPointF(x, h));
    }
    void mousePressEvent(QMouseEvent *event) override {
        if (event->button() == Qt::LeftButton) {
            sent.reset();
            scrub(event->position().x());
        } else if (event->button() == Qt::MiddleButton) {
            dragged_from = {event->position().x(), offset};
        }
    }
    void mouseMoveEvent(QMouseEvent *event) override {
        if (event->buttons() & Qt::LeftButton) {
            scrub(event->position().x());
        } else if ((event->buttons() & Qt::MiddleButton) && dragged_from) {
            const auto [x, from] = *dragged_from;
            show_view(from - (event->position().x() - x) / cell(), span);
        }
    }
    void mouseReleaseEvent(QMouseEvent *event) override {
        if (event->button() == Qt::MiddleButton) dragged_from.reset();
    }
    void wheelEvent(QWheelEvent *event) override {
        const QPoint delta = event->angleDelta();
        const bool sideways = delta.x() != 0 || (event->modifiers() & Qt::ShiftModifier);
        const double notches = (sideways ? (delta.x() ? delta.x() : delta.y()) : delta.y()) / 120.0;
        if (sideways) {
            show_view(offset - notches * span / 8, span);
        } else {
            // Round the pointer: the frame under it stays under it.
            const double at = event->position().x() / width();
            const double zoomed = std::clamp(span * std::pow(0.8, notches), min_span(), count());
            show_view(offset + at * (span - zoomed), zoomed);
        }
        event->accept();
    }
    void resizeEvent(QResizeEvent *) override { show_view(offset, span); }

  private:
    double count() const { return std::max(std::floor(last - first) + 1, 1.0); }
    // Zoomed in no further than a few frames across.
    double min_span() const { return std::min(count(), 5.0); }
    double cell() const { return width() / span; }
    // Where the left of the frame so many from the first lies.
    double x_of(double index) const { return (index - offset) * cell(); }
    // The frames, from the first, the view shows any of.
    std::pair<int64_t, int64_t> visible() const {
        return {int64_t(std::floor(offset)), int64_t(std::ceil(offset + span)) - 1};
    }
    void fit() {
        offset = 0;
        span = count();
    }
    void show_view(double new_offset, double new_span) {
        span = std::clamp(new_span, min_span(), count());
        offset = std::clamp(new_offset, 0.0, count() - span);
        update();
    }
    // A frame that leaves the view turns it a page: to start the view going forwards, to end it
    // going backwards.
    void follow() {
        const double index = std::floor(frame - first);
        if (index < offset) show_view(index + 1 - span, span);
        else if (index + 1 > offset + span) show_view(index, span);
    }
    // Ticks hang from the top at each frame's middle: each frame's while they are 3 pixels
    // apart, longer at steps that keep 8 pixels apart, and longest and labelled at steps that
    // leave labels room. Steps below a second divide it, so they meet each second's; above, they
    // are 1, 2 or 5 seconds times a power of ten. The range's ends are always long and labelled.
    void paint_ticks(QPainter &painter) {
        const double cell = this->cell();
        const int64_t second = std::max<int64_t>(std::llround(rate), 1);
        std::vector<int64_t> steps;
        for (int64_t step : {1, 2, 4, 5, 6, 8, 10, 12, 15})
            if (step < second && second % step == 0) steps.push_back(step);
        for (int64_t scale = 1; scale <= 1'000'000; scale *= 10)
            for (int64_t step : {1, 2, 5}) steps.push_back(second * step * scale);
        auto step_for = [&](double pixels) {
            for (int64_t step : steps)
                if (step * cell >= pixels) return step;
            return steps.back();
        };
        const QFontMetrics metrics = fontMetrics();
        const auto [from, to] = visible();
        const int64_t range_first = std::llround(std::ceil(first));
        const int64_t range_last = std::llround(std::floor(last));
        const int64_t start = range_first + from;
        const int64_t end = std::min<int64_t>(range_first + to, range_last);
        const double label_width = std::max(metrics.horizontalAdvance(label(double(start))),
                                            metrics.horizontalAdvance(label(double(end))));
        const int64_t medium = step_for(8);
        const int64_t labelled = step_for(std::max(2 * label_width, 100.0));

        QColor color = palette().color(QPalette::Text);
        color.setAlphaF(0.45);
        painter.setPen(QPen(color, 1));
        auto x_at = [&](int64_t at) { return std::round(x_of(at - first + 0.5)) + 0.5; };
        auto tick = [&](int64_t at, double length) {
            painter.drawLine(QPointF(x_at(at), 0), QPointF(x_at(at), length));
        };
        // The first multiple of `of` from `start`, either side of zero.
        auto first_multiple = [&](int64_t of) {
            const int64_t past = ((start % of) + of) % of;
            return past == 0 ? start : start + of - past;
        };
        if (cell >= 3)
            for (int64_t at = start; at <= end; ++at) tick(at, 4);
        for (int64_t at = first_multiple(medium); at <= end; at += medium) tick(at, 7);
        for (int64_t at = first_multiple(labelled); at <= end; at += labelled) tick(at, 10);
        for (int64_t at : {range_first, range_last})
            if (at >= start && at <= end) tick(at, 10);

        // Centred under their ticks; the ends' kept inside the bar, the others dropped where
        // they would be cut off or meet an end's.
        color.setAlphaF(0.75);
        painter.setPen(color);
        const double baseline = 10 + metrics.ascent();
        auto place = [&](int64_t at) {
            const double text_width = metrics.horizontalAdvance(label(double(at)));
            return QRectF(x_at(at) - text_width / 2, 10, text_width, metrics.height());
        };
        std::vector<QRectF> ends;
        for (int64_t at : {range_first, range_last}) {
            if (at < start || at > end || (at == range_last && range_last == range_first)) continue;
            QRectF rect = place(at);
            rect.moveLeft(std::clamp(rect.left(), 2.0, std::max(2.0, width() - 2 - rect.width())));
            painter.drawText(QPointF(rect.left(), baseline), label(double(at)));
            ends.push_back(rect.adjusted(-8, 0, 8, 0));
        }
        for (int64_t at = first_multiple(labelled); at <= end; at += labelled) {
            if (at == range_first || at == range_last) continue;
            const QRectF rect = place(at);
            if (rect.left() < 0 || rect.right() > width()) continue;
            if (std::any_of(ends.begin(), ends.end(),
                            [&](const QRectF &taken) { return taken.intersects(rect); }))
                continue;
            painter.drawText(QPointF(rect.left(), baseline), label(double(at)));
        }
    }
    QString label(double at) const {
        const rust::String text = frame_label(at, rate, timecode);
        return QString::fromUtf8(text.data(), text.size());
    }
    void scrub(double x) {
        const double index = std::clamp(std::floor(offset + x / cell()), 0.0, count() - 1);
        const double frame = first + index;
        if (sent == frame) return;
        sent = frame;
        scrubbed(frame);
    }

    std::function<void(double)> scrubbed;
    double first = 1, last = 1, frame = 1, rate = 24;
    // The view: the frames from the first at its left edge, and across it.
    double offset = 0, span = 1;
    bool timecode = false;
    std::vector<FrameMark> marks;
    std::optional<double> sent;
    // Where a middle drag began, and the view's offset then.
    std::optional<std::pair<double, double>> dragged_from;
};

Timebar::Timebar(rust::Fn<void(TimebarGesture)> gestures, const QStringList &rates,
                 QWidget *parent)
    : QWidget(parent), gestures(gestures) {
    const QColor color = palette().color(QPalette::ButtonText);
    play_backward_icon = glyph_icon(Glyph::PlayBackward, color);
    play_forward_icon = glyph_icon(Glyph::PlayForward, color);
    stop_icon = glyph_icon(Glyph::Stop, color);
    loop_icon = glyph_icon(Glyph::Loop, color);
    once_icon = glyph_icon(Glyph::Once, color);

    auto *layout = new QVBoxLayout(this);
    layout->setContentsMargins(0, 0, 0, 0);
    layout->setSpacing(2);
    auto *ends = new QHBoxLayout;
    ends->setSpacing(4);
    first = field(TimebarGestureKind::FirstTyped, QString());
    ends->addWidget(first);
    bar = new FrameBar([this](double frame) { send(TimebarGestureKind::Scrubbed, frame); });
    ends->addWidget(bar, 1);
    last = field(TimebarGestureKind::LastTyped, QString());
    ends->addWidget(last);
    layout->addLayout(ends);

    // The transport stays centred: the sides take equal shares of the room left.
    auto *controls = new QGridLayout;
    controls->setContentsMargins(0, 0, 0, 0);
    controls->setColumnStretch(0, 1);
    controls->setColumnStretch(2, 1);

    auto *left = new QHBoxLayout;
    rate = new QComboBox;
    rate->setEditable(true);
    rate->setInsertPolicy(QComboBox::NoInsert);
    rate->addItems(rates);
    rate->setFocusPolicy(Qt::ClickFocus);
    rate->setToolTip(QStringLiteral("The frames per second the timeline plays at: type a rate, "
                                    "or choose one"));
    rate->setSizeAdjustPolicy(QComboBox::AdjustToMinimumContentsLengthWithIcon);
    rate->setMinimumContentsLength(5);
    rate->lineEdit()->installEventFilter(this);
    connect(rate, &QComboBox::activated, this,
            [this](int index) { send(TimebarGestureKind::RateTyped, 0, rate->itemText(index)); });
    connect(rate->lineEdit(), &QLineEdit::editingFinished, this, [this] {
        if (rate->lineEdit()->isModified()) {
            rate->lineEdit()->setModified(false);
            send(TimebarGestureKind::RateTyped, 0, rate->currentText());
        }
        rate->clearFocus();
    });
    left->addWidget(rate);
    left->addWidget(new QLabel(QStringLiteral("fps")));
    timecode = button(TimebarGestureKind::ToggleTimecode, QIcon(),
                      QStringLiteral("Frames as timecode, counted from frame 0 at the rate, "
                                     "rather than as numbers. Either can be typed."));
    timecode->setText(QStringLiteral("TC"));
    timecode->setCheckable(true);
    left->addWidget(timecode);
    status = new QLabel;
    status->setSizePolicy(QSizePolicy::Ignored, QSizePolicy::Fixed);
    status->setToolTip(QStringLiteral("Frames shown per second while playing, and in real time "
                                      "those skipped since it began"));
    left->addWidget(status, 1);
    controls->addLayout(left, 0, 0);

    auto *transport = new QHBoxLayout;
    transport->setSpacing(0);
    transport->addWidget(button(TimebarGestureKind::FirstFrame,
                                glyph_icon(Glyph::FirstFrame, color),
                                QStringLiteral("The range's first frame")));
    transport->addWidget(button(TimebarGestureKind::StepBackward,
                                glyph_icon(Glyph::StepBackward, color),
                                QStringLiteral("Back a frame (Left over the viewer)")));
    play_backward = button(TimebarGestureKind::PlayBackward, play_backward_icon, QString());
    transport->addWidget(play_backward);
    frame = field(TimebarGestureKind::FrameTyped,
                  QStringLiteral("The frame viewed: type a number or timecode"));
    frame->setAlignment(Qt::AlignCenter);
    transport->addSpacing(4);
    transport->addWidget(frame);
    transport->addSpacing(4);
    play_forward = button(TimebarGestureKind::PlayForward, play_forward_icon, QString());
    transport->addWidget(play_forward);
    transport->addWidget(button(TimebarGestureKind::StepForward,
                                glyph_icon(Glyph::StepForward, color),
                                QStringLiteral("Forward a frame (Right over the viewer)")));
    transport->addWidget(button(TimebarGestureKind::LastFrame,
                                glyph_icon(Glyph::LastFrame, color),
                                QStringLiteral("The range's last frame")));
    controls->addLayout(transport, 0, 1);

    auto *right = new QHBoxLayout;
    right->addStretch(1);
    looping = button(TimebarGestureKind::ToggleLooping, loop_icon, QString());
    right->addWidget(looping);
    mode = button(TimebarGestureKind::ToggleRealTime, glyph_icon(Glyph::RealTime, color), QString());
    mode->setCheckable(true);
    right->addWidget(mode);
    range = new QComboBox;
    range->addItems({QStringLiteral("Global"), QStringLiteral("Input")});
    range->setFocusPolicy(Qt::NoFocus);
    connect(range, &QComboBox::currentIndexChanged, this, [this](int index) {
        send(index == 1 ? TimebarGestureKind::InputRange : TimebarGestureKind::GlobalRange);
    });
    right->addWidget(range);
    controls->addLayout(right, 0, 2);
    layout->addLayout(controls);
}

QLineEdit *Timebar::field(TimebarGestureKind kind, const QString &tooltip) {
    auto *field = new QLineEdit;
    field->setToolTip(tooltip);
    // Typing takes the keyboard from the panes until Enter or a click elsewhere gives it back.
    field->setFocusPolicy(Qt::ClickFocus);
    field->installEventFilter(this);
    connect(field, &QLineEdit::editingFinished, this, [=, this] {
        if (field->isModified()) {
            field->setModified(false);
            send(kind, 0, field->text());
        }
        field->clearFocus();
    });
    return field;
}

QToolButton *Timebar::button(TimebarGestureKind kind, const QIcon &icon, const QString &tooltip) {
    auto *button = new QToolButton;
    button->setIcon(icon);
    button->setToolTip(tooltip);
    button->setAutoRaise(true);
    button->setFocusPolicy(Qt::NoFocus);
    connect(button, &QToolButton::clicked, this, [=, this] { send(kind); });
    return button;
}

void Timebar::show_timebar(const TimebarData &data) {
    refreshing = true;
    bar->show_frames(data);
    const QFontMetrics metrics = frame->fontMetrics();
    const QString widest =
        data.timecode ? QStringLiteral("-00:00:00:00") : QStringLiteral("-000000");
    const std::pair<QLineEdit *, const QString &> fields[] = {
        {frame, data.frame_text}, {first, data.first_text}, {last, data.last_text}};
    for (const auto &[field, text] : fields) {
        field->setFixedWidth(metrics.horizontalAdvance(widest) + 12);
        // What the artist is typing stays.
        if (!field->hasFocus()) field->setText(text);
    }
    timecode->setChecked(data.timecode);
    if (!rate->hasFocus()) rate->setEditText(data.rate_text);

    const bool backward = data.playing == Playing::Backward;
    const bool forward = data.playing == Playing::Forward;
    play_backward->setIcon(backward ? stop_icon : play_backward_icon);
    play_backward->setToolTip(backward ? QStringLiteral("Stop (K or J over the viewer)")
                                       : QStringLiteral("Play backwards (J over the viewer)"));
    play_forward->setIcon(forward ? stop_icon : play_forward_icon);
    play_forward->setToolTip(forward ? QStringLiteral("Stop (K or L over the viewer)")
                                     : QStringLiteral("Play forwards (L over the viewer)"));

    looping->setIcon(data.looping ? loop_icon : once_icon);
    looping->setToolTip(data.looping
                            ? QStringLiteral("Loop: playing round from the other end at the "
                                             "range's end. Click to stop there instead.")
                            : QStringLiteral("Once: stopping at the range's end. Click to loop."));
    mode->setChecked(data.real_time);
    mode->setToolTip(
        data.real_time
            ? QStringLiteral("Real time: keeping to the rate, skipping frames not ready. Click to "
                             "show every frame instead.")
            : QStringLiteral("Real time is off: showing every frame once ready, no faster than "
                             "the rate. Click to keep to the rate, skipping frames not ready."));

    range->setCurrentIndex(data.input ? 1 : 0);
    range->setToolTip(
        !data.input         ? QStringLiteral("Global: the timeline's frames, set at the bar's ends")
        : data.input_found ? QStringLiteral("Input: the viewed node's sequences' frames")
                           : QStringLiteral("Input: the viewed node reads no sequence, so the "
                                            "timeline's frames"));
    for (QLineEdit *field : {first, last}) field->setEnabled(!data.input);
    first->setToolTip(data.input ? QStringLiteral("The range's first frame")
                                 : QStringLiteral("The timeline's first frame"));
    last->setToolTip(data.input ? QStringLiteral("The range's last frame")
                                : QStringLiteral("The timeline's last frame"));
    status->setText(data.status);
    refreshing = false;
}

bool Timebar::eventFilter(QObject *watched, QEvent *event) {
    if (event->type() == QEvent::KeyPress &&
        static_cast<QKeyEvent *>(event)->key() == Qt::Key_Escape) {
        if (auto *field = qobject_cast<QLineEdit *>(watched)) {
            field->setModified(false);
            field->clearFocus();
            rate->clearFocus();
            send(TimebarGestureKind::Restore);
            return true;
        }
    }
    return QWidget::eventFilter(watched, event);
}

void Timebar::send(TimebarGestureKind kind, double frame, const QString &text) {
    if (refreshing) return;
    gestures(TimebarGesture{kind, frame, text});
}

ViewerStrip::ViewerStrip(double limit, double step, rust::Fn<void(double)> exposure_set,
                         rust::Fn<void(TimebarGesture)> timebar_gestures,
                         const QStringList &rates, QWidget *parent)
    : QWidget(parent) {
    auto *layout = new QVBoxLayout(this);
    layout->setContentsMargins(6, 2, 6, 2);
    layout->setSpacing(2);
    timebar = new Timebar(timebar_gestures, rates, this);
    layout->addWidget(timebar);
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
                                  rust::Fn<void(double)> exposure_set,
                                  rust::Fn<void(TimebarGesture)> timebar_gestures,
                                  const QStringList &rates) {
    auto *widget = new QWidget(dock);
    auto *layout = new QVBoxLayout(widget);
    layout->setContentsMargins(0, 0, 0, 0);
    layout->setSpacing(0);
    layout->addWidget(QWidget::createWindowContainer(pane, widget), 1);
    auto *strip = new ViewerStrip(limit, step, exposure_set, timebar_gestures, rates, widget);
    layout->addWidget(strip);
    dock->setWidget(widget);
    return strip;
}
QLabel *strip_caption(ViewerStrip *strip) { return strip->caption; }
void show_timebar(ViewerStrip *strip, const TimebarData &data) {
    strip->timebar->show_timebar(data);
}
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
        } else if (data.kind == RowKind::Frame) {
            field->setFixedWidth(field->fontMetrics().horizontalAdvance(QStringLiteral("-000.0000")) +
                                 12);
            field->setPlaceholderText(QStringLiteral("Unset"));
            grid->addWidget(field, index, 1);
        } else {
            field->setPlaceholderText(QStringLiteral("No file"));
            auto *browse = new QPushButton(QStringLiteral("Browse…"));
            // From the event loop: the dialog's nested loop may rebuild the rows, deleting the
            // button under its own click.
            connect(browse, &QPushButton::clicked, this, [=, this] {
                QTimer::singleShot(0, this, [=, this] { send(PanelGestureKind::Browse, name); });
            });
            auto *reload = new QPushButton(QStringLiteral("Reload"));
            reload->setToolTip(QStringLiteral("Read the files again, rescanning a sequence"));
            connect(reload, &QPushButton::clicked, this,
                    [=, this] { send(PanelGestureKind::Reload, name); });
            auto *box = new QHBoxLayout;
            box->addWidget(field, 1);
            box->addWidget(browse);
            box->addWidget(reload);
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
