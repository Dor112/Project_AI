from pathlib import Path
from typing import Dict
import torch
from PySide6.QtCore import Qt, QSize
from PySide6.QtGui import QFont, QIcon, QPixmap, QPainter, QColor
from PySide6.QtWidgets import QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, QLabel, QStackedWidget, QFrame, QPushButton, QMessageBox, QScrollArea
from app.ml.cnn import build_model, count_parameters
from app.ml.dataset import FASHION_MNIST_CLASSES
from app.visualization.charts import apply_dark_theme
from app.ui.theme import AnimatedStack, WINDOW_STYLE, polish_page
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
    p.setPen(QColor("#101522"))
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
        self.setFixedHeight(42)
        self.setFont(QFont("Segoe UI", 11))
        self.setStyleSheet("""
            QPushButton { background-color: transparent; color: #e2e8f0; border: 1px solid transparent; border-radius: 8px; text-align: left; padding-left: 10px; }
            QPushButton:hover { background-color: rgba(125, 211, 252, 0.08); color: #7dd3fc; }
            QPushButton:focus { border: 1px solid #7dd3fc; }
            QPushButton:pressed { background-color: #29364c; }
            QPushButton:checked { background-color: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 rgba(125,211,252,0.25), stop:1 rgba(184,161,239,0.15)); color: #7dd3fc; border: 1px solid rgba(125, 211, 252, 0.35); }
        """)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        apply_dark_theme()
        self.setWindowTitle("NeuroLab — AI Laboratory")
        self.resize(1400, 900)
        self.setMinimumSize(1100, 720)
        self.setStyleSheet(WINDOW_STYLE)
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
        sidebar.setProperty("panel", True)
        sidebar.setStyleSheet("""
            QFrame[panel="true"] { background-color: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #141c2c, stop:1 #101522); border-right: 1px solid rgba(125, 211, 252, 0.15); }
        """)
        sidebar_layout = QVBoxLayout(sidebar)
        sidebar_layout.setContentsMargins(16, 14, 16, 14)
        sidebar_layout.setSpacing(4)
        logo = QLabel("◉  NeuroLab")
        logo.setFont(QFont("Segoe UI", 20, QFont.Weight.Bold))
        logo.setStyleSheet("color: #7dd3fc; padding: 8px 0 8px 0;")
        sidebar_layout.addWidget(logo)
        subtitle = QLabel("AI Laboratory · Fashion-MNIST")
        subtitle.setFont(QFont("Segoe UI", 9))
        subtitle.setStyleSheet("color: #92a3ba; padding-bottom: 8px;")
        sidebar_layout.addWidget(subtitle)
        self.stack = AnimatedStack()
        self.page_containers = {}
        self.buttons = {}
        nav_items = [
            ("Dashboard", "#7dd3fc", "D"),
            ("Dataset", "#67c9dc", "◈"),
            ("Architecture", "#b8a1ef", "▣"),
            ("Training", "#e5a0bd", "▶"),
            ("Feature Maps", "#e9bd75", "≡"),
            ("Prediction", "#7ad9b1", "✦"),
            ("Confusion Matrix", "#ed9292", "▦"),
            ("What If?", "#b8a1ef", "?"),
            ("Settings", "#92a3ba", "⚙"),
        ]
        groups = {"Dashboard": "ОБЗОР", "Training": "ЭКСПЕРИМЕНТЫ", "Settings": "СИСТЕМА"}
        for index, (text, color, letter) in enumerate(nav_items):
            if text in groups:
                group = QLabel(groups[text])
                group.setStyleSheet("color: #92a3ba; font-size: 8pt; letter-spacing: 2px; padding: 8px 10px 2px;")
                sidebar_layout.addWidget(group)
            btn = SidebarButton(text, color, letter)
            btn.clicked.connect(lambda checked=False, t=text: self._switch(t))
            btn.setShortcut(f"Alt+{index + 1}")
            btn.setToolTip(f"{text} · Alt+{index + 1}")
            sidebar_layout.addWidget(btn)
            self.buttons[text] = btn
        sidebar_layout.addStretch()
        info_frame = QFrame()
        info_frame.setProperty("panel", True)
        info_frame.setStyleSheet("""
            QFrame[panel="true"] { background-color: rgba(125, 211, 252, 0.05); border: 1px solid rgba(125, 211, 252, 0.2); border-radius: 10px; padding: 10px; }
        """)
        info_layout = QVBoxLayout(info_frame)
        info_layout.setContentsMargins(12, 12, 12, 12)
        info_layout.setSpacing(4)
        lbl_dev = QLabel(f"Device: {self.device}")
        lbl_dev.setStyleSheet("color: #b5c5da; font-size: 10pt;")
        info_layout.addWidget(lbl_dev)
        lbl_params = QLabel(f"Params: {self.num_params:,}")
        lbl_params.setStyleSheet("color: #b5c5da; font-size: 10pt;")
        info_layout.addWidget(lbl_params)
        sidebar_layout.addWidget(info_frame)
        root_layout.addWidget(sidebar)
        content = QWidget()
        content.setObjectName("workspaceSurface")
        content.setStyleSheet("QWidget#workspaceSurface { background-color: #101522; }")
        content_layout = QVBoxLayout(content)
        content_layout.setContentsMargins(0, 0, 0, 0)
        topbar = QFrame()
        topbar.setStyleSheet("QFrame { background: #141c2c; border-bottom: 1px solid #29364c; }")
        bar_layout = QHBoxLayout(topbar)
        bar_layout.setContentsMargins(30, 14, 30, 14)
        self.breadcrumb = QLabel("Лаборатория / Dashboard")
        self.breadcrumb.setStyleSheet("color: #b5c5da; font-size: 10pt; border: none;")
        bar_layout.addWidget(self.breadcrumb)
        bar_layout.addStretch()
        badge = QLabel("●  CPU   /   Fashion-MNIST")
        badge.setStyleSheet("color: #7ad9b1; font-size: 9pt; border: none;")
        bar_layout.addWidget(badge)
        content_layout.addWidget(topbar)
        content_layout.addWidget(self.stack, 1)
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
        for name, page in self.pages.items():
            polish_page(page)
            if name == "Training":
                container = QScrollArea()
                container.setWidgetResizable(True)
                container.setFrameShape(QFrame.Shape.NoFrame)
                container.setStyleSheet("QScrollArea { background: #101522; border: none; }")
                container.setWidget(page)
            else:
                container = page
            self.page_containers[name] = container
            self.stack.addWidget(container)
        self._switch("Dashboard")

    def _switch(self, name):
        for key, btn in self.buttons.items():
            btn.setChecked(key == name)
        if name in self.pages:
            self.stack.setCurrentWidget(self.page_containers[name])
            self.breadcrumb.setText(f"Лаборатория / {name}")
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