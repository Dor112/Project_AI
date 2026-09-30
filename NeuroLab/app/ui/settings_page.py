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
        header.setFont(QFont("Segoe UI", 28, QFont.Weight.Bold))
        header.setStyleSheet("color: #00d4ff;")
        outer.addWidget(header)
        sub = QLabel("Информация о системе и оборудовании")
        sub.setStyleSheet("color: #94a3b8; font-size: 11pt; margin-bottom: 20px;")
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
        cards = [("CPU", cpu_name, "#00d4ff"), ("Logical Threads", str(threads), "#a855f7"), ("RAM", f"{ram:.1f} GB" if ram > 0 else "Unknown", "#22d3ee"), ("GPU", gpu, "#f472b6"), ("CUDA", cuda, "#f59e0b"), ("PyTorch Device", device, "#10b981")]
        for i, (title, value, color) in enumerate(cards):
            card = QFrame()
            card.setStyleSheet(f"""
                QFrame {{ background-color: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 rgba(15,23,42,0.9), stop:1 rgba(10,14,26,0.9)); border: 1px solid {color}40; border-radius: 14px; }}
                QFrame:hover {{ border: 1px solid {color}aa; }}
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
        info.setStyleSheet("""
            QFrame { background-color: rgba(0, 212, 255, 0.05); border: 1px solid rgba(0, 212, 255, 0.25); border-radius: 14px; padding: 16px; }
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
        pass