"""Shared presentation styles; no model or training behavior lives here."""
from PySide6.QtCore import Qt, QEasingCurve, QPropertyAnimation
from PySide6.QtWidgets import (
    QLabel, QPushButton, QComboBox, QAbstractSpinBox,
    QGraphicsOpacityEffect, QStackedWidget,
)

BACKGROUND = "#101522"
ACCENT = "#7dd3fc"

WINDOW_STYLE = """
QMainWindow, QStackedWidget { background: #101522; }
QWidget { color: #e2e8f0; font-family: 'Segoe UI', sans-serif; }
QToolTip { background: #253249; color: #f1f5f9; border: 1px solid #536984; padding: 8px; }
QScrollBar:vertical { background: #101522; width: 10px; margin: 2px; }
QScrollBar:horizontal { background: #101522; height: 10px; margin: 2px; }
QScrollBar::handle { background: #3b4a63; border-radius: 4px; min-height: 28px; min-width: 28px; }
QScrollBar::handle:hover { background: #7dd3fc; }
QScrollBar::add-line, QScrollBar::sub-line { width: 0; height: 0; }
QScrollBar::add-page, QScrollBar::sub-page { background: transparent; }
"""

BUTTON_STYLE = """
QPushButton { background: #202d43; color: #e2e8f0; border: 1px solid #3a4e69;
    border-radius: 8px; padding: 9px 14px; font-weight: 600; }
QPushButton:hover { background: #2b3c55; border-color: #7dd3fc; }
QPushButton:pressed { background: #354c69; }
QPushButton:focus { border: 2px solid #7dd3fc; padding: 8px 13px; }
QPushButton:disabled { background: #171f2f; color: #718096; border-color: #29364c; }
QPushButton[primary="true"] { background: #7dd3fc; color: #102033; border-color: #7dd3fc; }
QPushButton[primary="true"]:hover { background: #bae6fd; }
QPushButton[primary="true"]:pressed { background: #38bdf8; }
QPushButton[primary="true"]:disabled { background: #26394b; color: #8298ac; border-color: #34495c; }
"""

INPUT_STYLE = """
QSpinBox, QDoubleSpinBox, QComboBox { background: #182235; color: #f1f5f9;
    border: 1px solid #41536e; border-radius: 7px; padding: 7px 10px; min-width: 90px; }
QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus { border: 2px solid #7dd3fc; }
QSpinBox:hover, QDoubleSpinBox:hover, QComboBox:hover { border-color: #7dd3fc; }
QSpinBox:disabled, QDoubleSpinBox:disabled, QComboBox:disabled { color: #718096; border-color: #29364c; }
QComboBox QAbstractItemView { background: #202d43; color: #f1f5f9;
    selection-background-color: #354c69; selection-color: #ffffff; padding: 6px; }
"""


def polish_page(page):
    """Apply consistent control sizing and focus states to existing pages."""
    page.setObjectName("laboratoryPage")
    page.setStyleSheet("QWidget#laboratoryPage { background: #101522; }")
    for button in page.findChildren(QPushButton):
        button.setCursor(Qt.CursorShape.PointingHandCursor)
        button.setMinimumHeight(38)
        button.setStyleSheet(BUTTON_STYLE)
    for name in ("btn_start", "btn_load", "btn_compute", "btn_load_img"):
        button = getattr(page, name, None)
        if button is not None:
            button.setProperty("primary", True)
            button.style().unpolish(button)
            button.style().polish(button)
    for control in page.findChildren(QComboBox) + page.findChildren(QAbstractSpinBox):
        control.setMinimumHeight(36)
        control.setStyleSheet(INPUT_STYLE)
    status = getattr(page, "status", None)
    if status is not None:
        status.setWordWrap(True)
        status.setStyleSheet("color: #b5c5da; background: #182235; border-radius: 8px; padding: 10px 12px;")


class AnimatedStack(QStackedWidget):
    """Fade a snapshot of the outgoing page, leaving live charts untouched."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self._overlay = QLabel(self)
        self._overlay.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self._overlay.hide()
        self._effect = QGraphicsOpacityEffect(self._overlay)
        self._overlay.setGraphicsEffect(self._effect)
        self._animation = QPropertyAnimation(self._effect, b"opacity", self)
        self._animation.setDuration(160)
        self._animation.setEasingCurve(QEasingCurve.Type.OutCubic)
        self._animation.finished.connect(self._overlay.hide)

    def setCurrentWidget(self, widget):
        if self.currentWidget() is widget:
            return
        self._animation.stop()
        self._overlay.hide()
        snapshot = self.currentWidget().grab() if self.isVisible() and self.currentWidget() else None
        super().setCurrentWidget(widget)
        if snapshot is not None:
            self._overlay.setPixmap(snapshot)
            self._overlay.setGeometry(self.rect())
            self._effect.setOpacity(1.0)
            self._overlay.show()
            self._overlay.raise_()
            self._animation.setStartValue(1.0)
            self._animation.setEndValue(0.0)
            self._animation.start()

    def resizeEvent(self, event):
        self._animation.stop()
        self._overlay.hide()
        super().resizeEvent(event)
