import torch
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont
from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QFrame, QGridLayout
from app.ml.hardware import get_cpu_name, get_logical_threads, get_ram_gb, get_gpu_info, get_cuda_available

class SettingsPage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(30, 30, 30, 30)
        header = QLabel("System Information")
        header.setFont(QFont("Segoe UI", 24, QFont.Weight.Bold))
        header.setStyleSheet("color: #7dd3fc;")
        outer.addWidget(header)
        sub = QLabel("Информация о системе и оборудовании")
        sub.setStyleSheet("color: #b5c5da; font-size: 11pt; margin-bottom: 20px;")
        outer.addWidget(sub)
        grid = QGridLayout()
        grid.setSpacing(16)
        outer.addLayout(grid)
        cpu_name = get_cpu_name()
        threads = get_logical_threads()
        ram = get_ram_gb()
        gpu = get_gpu_info()
        cuda = "Available" if get_cuda_available() else "Unavailable"
        device = "CUDA" if torch.cuda.is_available() else "CPU"
        cards = [("CPU", cpu_name, "#7dd3fc"), ("Logical Threads", str(threads), "#b8a1ef"), ("RAM", f"{ram:.1f} GB" if ram > 0 else "Unknown", "#67c9dc"), ("GPU", gpu, "#e5a0bd"), ("CUDA", cuda, "#e9bd75"), ("PyTorch Device", device, "#7ad9b1")]
        for i, (title, value, color) in enumerate(cards):
            card = QFrame()
            card.setProperty("panel", True)
            card.setStyleSheet(f"""
                QFrame[panel="true"] {{ background-color: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 rgba(25,35,55,0.95), stop:1 rgba(25,35,55,0.95)); border: 1px solid #35445e; border-radius: 14px; }}
                QFrame[panel="true"]:hover {{ border: 1px solid #35445e; }}
            """)
            card.setMinimumHeight(130)
            layout = QVBoxLayout(card)
            layout.setContentsMargins(20, 18, 20, 18)
            title_lbl = QLabel(title.upper())
            title_lbl.setStyleSheet(f"color: {color}; font-size: 10pt; letter-spacing: 1px;")
            layout.addWidget(title_lbl)
            value_lbl = QLabel(value)
            value_lbl.setFont(QFont("Segoe UI", 18, QFont.Weight.Bold))
            value_lbl.setStyleSheet("color: #e2e8f0;")
            value_lbl.setWordWrap(True)
            layout.addWidget(value_lbl)
            layout.addStretch()
            grid.addWidget(card, i // 3, i % 3)
        outer.addStretch()
        info = QFrame()
        info.setProperty("panel", True)
        info.setStyleSheet("""
            QFrame[panel="true"] { background-color: rgba(125, 211, 252, 0.05); border: 1px solid rgba(125, 211, 252, 0.25); border-radius: 14px; padding: 16px; }
        """)
        info_layout = QVBoxLayout(info)
        info_layout.setSpacing(6)
        lines = [
            ("◉  CPU Mode", "Приложение оптимизировано для работы на CPU без NVIDIA GPU."),
            ("◉  Performance", "Маленькая CNN (~213K параметров) обеспечивает быстрое обучение."),
            ("◉  Demo Mode", "Используйте Quick Demo для быстрой демонстрации (3 эпохи, 5000 samples)."),
            ("◉  Memory", "Приложение использует ограниченное количество RAM и не загружает весь dataset сразу."),
        ]
        for title, text in lines:
            row = QHBoxLayout()
            t = QLabel(title)
            t.setFont(QFont("Segoe UI", 11, QFont.Weight.Bold))
            t.setStyleSheet("color: #7dd3fc;")
            t.setFixedWidth(140)
            d = QLabel(text)
            d.setStyleSheet("color: #cbd5e1; font-size: 11pt;")
            d.setWordWrap(True)
            row.addWidget(t)
            row.addWidget(d, 1)
            info_layout.addLayout(row)
        outer.addWidget(info)

    def on_activated(self):
        pass