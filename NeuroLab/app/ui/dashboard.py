import torch
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont
from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QFrame, QGridLayout
from app.ml.cnn import SmallCNN

class StatCard(QFrame):
    def __init__(self, title, value, accent, parent=None):
        super().__init__(parent)
        self.setStyleSheet(f"""
            QFrame {{ background-color: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 rgba(15,23,42,0.9), stop:1 rgba(10,14,26,0.9)); border: 1px solid {accent}40; border-radius: 14px; }}
            QFrame:hover {{ border: 1px solid {accent}aa; }}
        """)
        self.setMinimumHeight(130)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 18, 20, 18)
        title_lbl = QLabel(title.upper())
        title_lbl.setStyleSheet(f"color: {accent}; font-size: 10pt; letter-spacing: 1px;")
        layout.addWidget(title_lbl)
        self.value_lbl = QLabel(value)
        self.value_lbl.setFont(QFont("Segoe UI", 26, QFont.Weight.Bold))
        self.value_lbl.setStyleSheet("color: #e2e8f0;")
        layout.addWidget(self.value_lbl)
        layout.addStretch()

    def set_value(self, value):
        self.value_lbl.setText(value)

class DashboardPage(QWidget):
    def __init__(self, model, device, num_params, parent=None):
        super().__init__(parent)
        self.model = model
        self.device = device
        self.num_params = num_params
        outer = QVBoxLayout(self)
        outer.setContentsMargins(30, 30, 30, 30)
        header = QLabel("Dashboard")
        header.setFont(QFont("Segoe UI", 28, QFont.Weight.Bold))
        header.setStyleSheet("color: #00d4ff;")
        outer.addWidget(header)
        sub = QLabel("Обзор проекта NeuroLab и текущее состояние модели")
        sub.setStyleSheet("color: #94a3b8; font-size: 11pt; margin-bottom: 20px;")
        outer.addWidget(sub)
        grid = QGridLayout()
        grid.setSpacing(16)
        outer.addLayout(grid)
        self.card_model = StatCard("Model", "Small CNN", "#00d4ff")
        self.card_params = StatCard("Parameters", f"{self.num_params:,}", "#a855f7")
        self.card_device = StatCard("Device", str(self.device), "#22d3ee")
        self.card_classes = StatCard("Classes", "10 (Fashion-MNIST)", "#f472b6")
        self.card_layers = StatCard("Conv Layers", "2 blocks", "#f59e0b")
        self.card_status = StatCard("Status", "Ready", "#10b981")
        cards = [self.card_model, self.card_params, self.card_device, self.card_classes, self.card_layers, self.card_status]
        for i, card in enumerate(cards):
            grid.addWidget(card, i // 3, i % 3)
        outer.addStretch()
        info = QFrame()
        info.setStyleSheet("""
            QFrame { background-color: rgba(0, 212, 255, 0.05); border: 1px solid rgba(0, 212, 255, 0.25); border-radius: 14px; padding: 16px; }
        """)
        info_layout = QVBoxLayout(info)
        info_layout.setSpacing(6)
        lines = [
            ("◉  Dataset", "Fashion-MNIST — 70 000 grayscale изображений 28×28 в 10 классах."),
            ("◉  Model", "Маленькая CNN на PyTorch: 2 convolutional блока + FC-классификатор (~213K параметров)."),
            ("◉  Training", "Adam optimizer, CPU-first, быстрое обучение для демонстрации."),
            ("◉  Inference", "Загружайте свои изображения и смотрите предсказания + feature maps."),
        ]
        for title, text in lines:
            row = QHBoxLayout()
            t = QLabel(title)
            t.setFont(QFont("Segoe UI", 11, QFont.Weight.Bold))
            t.setStyleSheet("color: #00d4ff;")
            t.setFixedWidth(140)
            d = QLabel(text)
            d.setStyleSheet("color: #cbd5e1; font-size: 11pt;")
            d.setWordWrap(True)
            row.addWidget(t)
            row.addWidget(d, 1)
            info_layout.addLayout(row)
        outer.addWidget(info)

    def on_activated(self):
        self.card_device.set_value(str(self.device))
        self.card_params.set_value(f"{self.num_params:,}")