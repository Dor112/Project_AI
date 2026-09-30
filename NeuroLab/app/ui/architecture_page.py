from PySide6.QtCore import Qt
from PySide6.QtGui import QFont
from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel, QSplitter, QFrame, QHBoxLayout
from app.visualization.network_view import NetworkView

class ArchitecturePage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(30, 30, 30, 30)
        header = QLabel("Architecture")
        header.setFont(QFont("Segoe UI", 28, QFont.Weight.Bold))
        header.setStyleSheet("color: #00d4ff;")
        outer.addWidget(header)
        sub = QLabel("Визуальное представление слоёв Small CNN")
        sub.setStyleSheet("color: #94a3b8; font-size: 11pt; margin-bottom: 16px;")
        outer.addWidget(sub)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        self.network_view = NetworkView()
        splitter.addWidget(self.network_view)
        desc = QFrame()
        desc.setStyleSheet("""
            QFrame { background-color: rgba(15, 23, 42, 0.6); border: 1px solid rgba(0, 212, 255, 0.2); border-radius: 12px; }
        """)
        desc.setMinimumWidth(320)
        desc_layout = QVBoxLayout(desc)
        desc_layout.setContentsMargins(20, 20, 20, 20)
        desc_layout.setSpacing(10)
        title = QLabel("Layer Legend")
        title.setFont(QFont("Segoe UI", 14, QFont.Weight.Bold))
        title.setStyleSheet("color: #a855f7;")
        desc_layout.addWidget(title)
        legend = [
            ("#0ea5e9", "Input", "Входное изображение 1×28×28 (grayscale)"),
            ("#22d3ee", "Conv2d", "Свёрточный слой (kernel 3×3, padding 1)"),
            ("#a855f7", "ReLU", "Функция активации"),
            ("#f472b6", "MaxPool", "Понижение пространственного разрешения в 2 раза"),
            ("#94a3b8", "Flatten", "Превращение тензора в вектор"),
            ("#f59e0b", "Linear", "Полносвязный слой"),
            ("#10b981", "Output", "Выход: 10 классов Fashion-MNIST"),
        ]
        for color, name, description in legend:
            row = QWidget()
            row_layout = QVBoxLayout(row)
            row_layout.setContentsMargins(0, 4, 0, 4)
            row_layout.setSpacing(2)
            head = QWidget()
            head_layout = QHBoxLayout(head)
            head_layout.setContentsMargins(0, 0, 0, 0)
            head_layout.setSpacing(8)
            swatch = QLabel()
            swatch.setFixedSize(14, 14)
            swatch.setStyleSheet(f"background-color: {color}; border-radius: 3px;")
            head_layout.addWidget(swatch)
            lbl_name = QLabel(name)
            lbl_name.setFont(QFont("Segoe UI", 11, QFont.Weight.Bold))
            lbl_name.setStyleSheet("color: #e2e8f0;")
            head_layout.addWidget(lbl_name)
            head_layout.addStretch()
            row_layout.addWidget(head)
            lbl_desc = QLabel(description)
            lbl_desc.setStyleSheet("color: #94a3b8; font-size: 10pt; padding-left: 22px;")
            lbl_desc.setWordWrap(True)
            row_layout.addWidget(lbl_desc)
            desc_layout.addWidget(row)
        desc_layout.addStretch()
        splitter.addWidget(desc)
        splitter.setSizes([800, 360])
        outer.addWidget(splitter, 1)

    def on_activated(self):
        self.network_view.update()