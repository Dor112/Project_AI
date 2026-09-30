from typing import List
from PySide6.QtCore import Qt, QRectF, QPointF
from PySide6.QtGui import QPainter, QColor, QLinearGradient, QPen, QBrush, QFont
from PySide6.QtWidgets import QWidget

class LayerInfo:
    def __init__(self, name, kind, shape, color):
        self.name = name
        self.kind = kind
        self.shape = shape
        self.color = color

def build_architecture_layers():
    return [
        LayerInfo("Input", "input", "1×28×28", "#0ea5e9"),
        LayerInfo("Conv2d", "conv", "16×28×28", "#22d3ee"),
        LayerInfo("ReLU", "act", "16×28×28", "#a855f7"),
        LayerInfo("MaxPool", "pool", "16×14×14", "#f472b6"),
        LayerInfo("Conv2d", "conv", "32×14×14", "#22d3ee"),
        LayerInfo("ReLU", "act", "32×14×14", "#a855f7"),
        LayerInfo("MaxPool", "pool", "32×7×7", "#f472b6"),
        LayerInfo("Flatten", "flat", "1568", "#94a3b8"),
        LayerInfo("Linear", "fc", "128", "#f59e0b"),
        LayerInfo("ReLU", "act", "128", "#a855f7"),
        LayerInfo("Linear", "fc", "10", "#f59e0b"),
        LayerInfo("Output", "output", "10", "#10b981"),
    ]

class NetworkView(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.layers = build_architecture_layers()
        self.setMinimumSize(600, 600)
        self.setStyleSheet("background-color: #0a0e1a;")

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        w = self.width()
        h = self.height()
        bg = QLinearGradient(0, 0, 0, h)
        bg.setColorAt(0, QColor("#0a0e1a"))
        bg.setColorAt(1, QColor("#0f172a"))
        painter.fillRect(self.rect(), bg)
        painter.setPen(QColor("#cdd6f4"))
        title_font = QFont("Segoe UI", 14, QFont.Weight.Bold)
        painter.setFont(title_font)
        painter.drawText(QRectF(0, 10, w, 30), Qt.AlignmentFlag.AlignHCenter, "Small CNN Architecture")
        margin_top = 60
        margin_bottom = 20
        block_w = min(360, w - 80)
        block_h = 26
        spacing = 10
        total_h = len(self.layers) * (block_h + spacing)
        available_h = h - margin_top - margin_bottom
        if total_h > available_h:
            scale = available_h / total_h
            block_h = max(12, int(block_h * scale))
            spacing = max(2, int(spacing * scale))
            total_h = len(self.layers) * (block_h + spacing)
        start_x = (w - block_w) / 2
        start_y = margin_top + (available_h - total_h) / 2
        pen_arrow = QPen(QColor("#334155"), 1.5)
        painter.setPen(pen_arrow)
        for i in range(len(self.layers) - 1):
            y1 = start_y + i * (block_h + spacing) + block_h
            y2 = start_y + (i + 1) * (block_h + spacing)
            mid = start_x + block_w / 2
            painter.drawLine(QPointF(mid, y1), QPointF(mid, y2))
            painter.drawLine(QPointF(mid - 4, y2 - 4), QPointF(mid, y2))
            painter.drawLine(QPointF(mid + 4, y2 - 4), QPointF(mid, y2))
        font_name = QFont("Segoe UI", 9, QFont.Weight.Bold)
        font_shape = QFont("Consolas", 8)
        for i, layer in enumerate(self.layers):
            y = start_y + i * (block_h + spacing)
            rect = QRectF(start_x, y, block_w, block_h)
            grad = QLinearGradient(rect.topLeft(), rect.bottomRight())
            base = QColor(layer.color)
            grad.setColorAt(0, base)
            grad.setColorAt(1, base.darker(140))
            painter.setBrush(QBrush(grad))
            painter.setPen(QPen(base.lighter(130), 1))
            painter.drawRoundedRect(rect, 6, 6)
            painter.setPen(QColor("#0a0e1a"))
            painter.setFont(font_name)
            painter.drawText(rect.adjusted(10, 0, -120, 0), Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft, layer.name)
            painter.setPen(QColor("#0a0e1a"))
            painter.setFont(font_shape)
            painter.drawText(rect.adjusted(0, 0, -10, 0), Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignRight, layer.shape)
        painter.end()