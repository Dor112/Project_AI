from pathlib import Path
from typing import Dict
import torch
from PySide6.QtCore import Qt, QSize
from PySide6.QtGui import QFont, QIcon, QPixmap, QPainter, QColor
from PySide6.QtWidgets import QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, QLabel, QStackedWidget, QFrame, QPushButton, QMessageBox
from app.ml.cnn import build_model, count_parameters
from app.ml.dataset import FASHION_MNIST_CLASSES
from app.visualization.charts import apply_dark_theme
from app.ui.dashboard import DashboardPage
from app.ui.dataset_page import DatasetPage
from app.ui.architecture_page import ArchitecturePage
from app.ui.training_page import TrainingPage
from app.ui.feature_maps_page import FeatureMapsPage
from app.ui.prediction_page import PredictionPage
from app.ui.confusion_matrix_page import ConfusionMatrixPage
from app.ui.whatif_page import WhatIfPage
from app.ui.settings_page import SettingsPage

ASSETS_DIR = Path(__file__).resolve().parent.parent.parent / "assets"
MODELS_DIR = ASSETS_DIR / "models"

def _make_icon(color, letter, size=28):
    pix = QPixmap(size, size)
    pix.fill(Qt.GlobalColor.transparent)
    p = QPainter(pix)
    p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    p.setBrush(QColor(color))
    p.setPen(Qt.PenStyle.NoPen)
    p.drawRoundedRect(2, 2, size - 4, size - 4, 6, 6)
    p.setPen(QColor("#0a0e1a"))
    f = QFont("Segoe UI", int(size * 0.5), QFont.Weight.Bold)
    p.setFont(f)
    p.drawText(pix.rect(), Qt.AlignmentFlag.AlignCenter, letter)
    p.end()
    return QIcon(pix)

class SidebarButton(QPushButton):
    def __init__(self, text, color, letter, parent=None):
        super().__init__(parent)
        self.setText("   " + text)
        self.setIcon(_make_icon(color, letter))
        self.setIconSize(QSize(28, 28))
        self.setCheckable(True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFixedHeight(46)
        self.setFont(QFont("Segoe UI", 11))
        self.setStyleSheet("""
            QPushButton { background-color: transparent; color: #cdd6f4; border: none; border-radius: 10px; text-align: left; padding-left: 10px; }
            QPushButton:hover { background-color: rgba(0, 212, 255, 0.08); color: #00d4ff; }
            QPushButton:checked { background-color: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 rgba(0,212,255,0.25), stop:1 rgba(168,85,247,0.15)); color: #00d4ff; border: 1px solid rgba(0, 212, 255, 0.35); }
        """)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        apply_dark_theme()
        self.setWindowTitle("NeuroLab — AI Laboratory")
        self.resize(1400, 900)
        self.setMinimumSize(1100, 720)
        self.setStyleSheet("""
            QMainWindow { background-color: #0a0e1a; }
            QWidget { color: #cdd6f4; font-family: "Segoe UI", "Inter", sans-serif; }
            QScrollBar:vertical { background: #0a0e1a; width: 10px; margin: 0; }
            QScrollBar::handle:vertical { background: #1e293b; border-radius: 5px; min-height: 20px; }
            QScrollBar::handle:vertical:hover { background: #00d4ff; }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
        """)
        self.device = torch.device("cpu")
        self.model = build_model(num_classes=len(FASHION_MNIST_CLASSES))
        self.num_params = count_parameters(self.model)
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        central = QWidget()
        self.setCentralWidget(central)
        root_layout = QHBoxLayout(central)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)
        sidebar = QFrame()
        sidebar.setFixedWidth(260)
        sidebar.setStyleSheet("""
            QFrame { background-color: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #0b1020, stop:1 #0a0e1a); border-right: 1px solid rgba(0, 212, 255, 0.15); }
        """)
        sidebar_layout = QVBoxLayout(sidebar)
        sidebar_layout.setContentsMargins(16, 20, 16, 20)
        sidebar_layout.setSpacing(8)
        logo = QLabel("◉ NeuroLab")
        logo.setFont(QFont("Segoe UI", 20, QFont.Weight.Bold))
        logo.setStyleSheet("color: #00d4ff; padding: 10px 0 20px 0;")
        sidebar_layout.addWidget(logo)
        subtitle = QLabel("AI Laboratory · Fashion-MNIST")
        subtitle.setFont(QFont("Segoe UI", 9))
        subtitle.setStyleSheet("color: #64748b; padding-bottom: 20px;")
        sidebar_layout.addWidget(subtitle)
        self.stack = QStackedWidget()
        self.buttons = {}
        nav_items = [
            ("Dashboard", "#00d4ff", "D"),
            ("Dataset", "#22d3ee", "◈"),
            ("Architecture", "#a855f7", "▣"),
            ("Training", "#f472b6", "▶"),
            ("Feature Maps", "#f59e0b", "≡"),
            ("Prediction", "#10b981", "✦"),
            ("Confusion Matrix", "#ef4444", "▦"),
            ("What If?", "#8b5cf6", "?"),
            ("Settings", "#64748b", "⚙"),
        ]
        for text, color, letter in nav_items:
            btn = SidebarButton(text, color, letter)
            btn.clicked.connect(lambda checked=False, t=text: self._switch(t))
            sidebar_layout.addWidget(btn)
            self.buttons[text] = btn
        sidebar_layout.addStretch()
        info_frame = QFrame()
        info_frame.setStyleSheet("""
            QFrame { background-color: rgba(0, 212, 255, 0.05); border: 1px solid rgba(0, 212, 255, 0.2); border-radius: 10px; padding: 10px; }
        """)
        info_layout = QVBoxLayout(info_frame)
        info_layout.setContentsMargins(12, 12, 12, 12)
        info_layout.setSpacing(4)
        lbl_dev = QLabel(f"Device: {self.device}")
        lbl_dev.setStyleSheet("color: #94a3b8; font-size: 10pt;")
        info_layout.addWidget(lbl_dev)
        lbl_params = QLabel(f"Params: {self.num_params:,}")
        lbl_params.setStyleSheet("color: #94a3b8; font-size: 10pt;")
        info_layout.addWidget(lbl_params)
        sidebar_layout.addWidget(info_frame)
        root_layout.addWidget(sidebar)
        content = QWidget()
        content.setStyleSheet("background-color: #0a0e1a;")
        content_layout = QVBoxLayout(content)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.addWidget(self.stack)
        root_layout.addWidget(content, 1)
        self.pages = {
            "Dashboard": DashboardPage(self.model, self.device, self.num_params),
            "Dataset": DatasetPage(),
            "Architecture": ArchitecturePage(),
            "Training": TrainingPage(self.model, self.device),
            "Feature Maps": FeatureMapsPage(self.model, self.device),
            "Prediction": PredictionPage(self.model, self.device),
            "Confusion Matrix": ConfusionMatrixPage(self.model, self.device),
            "What If?": WhatIfPage(self.model, self.device),
            "Settings": SettingsPage(),
        }
        for page in self.pages.values():
            self.stack.addWidget(page)
        self._switch("Dashboard")

    def _switch(self, name):
        for key, btn in self.buttons.items():
            btn.setChecked(key == name)
        if name in self.pages:
            self.stack.setCurrentWidget(self.pages[name])
            page = self.pages[name]
            if hasattr(page, "on_activated"):
                try:
                    page.on_activated()
                except Exception as exc:
                    QMessageBox.warning(self, "Page error", str(exc))

    def closeEvent(self, event):
        training_page = self.pages.get("Training")
        if training_page is not None and hasattr(training_page, "request_stop"):
            try:
                training_page.request_stop()
            except Exception:
                pass
        super().closeEvent(event)