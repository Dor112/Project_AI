from pathlib import Path
import numpy as np
import torch
from PIL import Image
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont, QPixmap, QImage
from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QFileDialog, QFrame, QMessageBox, QProgressBar
from app.ml.cnn import SmallCNN
from app.ml.inference import predict_pil
from app.ml.dataset import FASHION_MNIST_CLASSES

def _numpy_to_qpixmap_rgb(arr):
    h, w, ch = arr.shape
    if ch == 3:
        qimg = QImage(arr.data, w, h, 3 * w, QImage.Format.Format_RGB888).copy()
    else:
        qimg = QImage(arr.data, w, h, w, QImage.Format.Format_Grayscale8).copy()
    return QPixmap.fromImage(qimg)

class PredictionPage(QWidget):
    def __init__(self, model, device, parent=None):
        super().__init__(parent)
        self.model = model
        self.device = device
        outer = QVBoxLayout(self)
        outer.setContentsMargins(30, 30, 30, 30)
        header = QLabel("Prediction")
        header.setFont(QFont("Segoe UI", 28, QFont.Weight.Bold))
        header.setStyleSheet("color: #00d4ff;")
        outer.addWidget(header)
        sub = QLabel("Загрузите изображение — модель предскажет класс Fashion-MNIST")
        sub.setStyleSheet("color: #94a3b8; font-size: 11pt; margin-bottom: 16px;")
        outer.addWidget(sub)
        ctrl = QHBoxLayout()
        self.btn_load = QPushButton("📁  Load Image")
        self.btn_load.setStyleSheet(self._btn_style("#00d4ff"))
        self.btn_load.clicked.connect(self._load_image)
        ctrl.addWidget(self.btn_load)
        self.btn_demo = QPushButton("🎲  Demo (random noise 28×28)")
        self.btn_demo.setStyleSheet(self._btn_style("#a855f7"))
        self.btn_demo.clicked.connect(self._demo_noise)
        ctrl.addWidget(self.btn_demo)
        ctrl.addStretch()
        outer.addLayout(ctrl)
        body = QHBoxLayout()
        left = QVBoxLayout()
        self.img_lbl = QLabel("Нет изображения")
        self.img_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.img_lbl.setFixedSize(360, 360)
        self.img_lbl.setStyleSheet("""
            QLabel { background: #111827; border: 1px solid #1f2937; border-radius: 16px; color: #64748b; font-size: 12pt; }
        """)
        left.addWidget(self.img_lbl, 0, Qt.AlignmentFlag.AlignHCenter)
        left.addSpacing(10)
        self.file_lbl = QLabel("")
        self.file_lbl.setStyleSheet("color: #94a3b8; font-size: 10pt;")
        self.file_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        left.addWidget(self.file_lbl)
        left.addStretch()
        body.addLayout(left, 1)
        right = QVBoxLayout()
        self.result_frame = QFrame()
        self.result_frame.setStyleSheet("""
            QFrame { background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 rgba(0,212,255,0.1), stop:1 rgba(168,85,247,0.1)); border: 1px solid rgba(0, 212, 255, 0.3); border-radius: 16px; }
        """)
        result_layout = QVBoxLayout(self.result_frame)
        result_layout.setContentsMargins(24, 24, 24, 24)
        title = QLabel("PREDICTION")
        title.setStyleSheet("color: #00d4ff; font-size: 11pt; letter-spacing: 2px;")
        result_layout.addWidget(title)
        self.class_lbl = QLabel("—")
        self.class_lbl.setFont(QFont("Segoe UI", 34, QFont.Weight.Bold))
        self.class_lbl.setStyleSheet("color: #e2e8f0;")
        result_layout.addWidget(self.class_lbl)
        self.prob_lbl = QLabel("")
        self.prob_lbl.setStyleSheet("color: #94a3b8; font-size: 12pt;")
        result_layout.addWidget(self.prob_lbl)
        right.addWidget(self.result_frame)
        topk_title = QLabel("TOP-5 PROBABILITIES")
        topk_title.setStyleSheet("color: #a855f7; font-size: 11pt; letter-spacing: 2px; margin-top: 10px;")
        right.addWidget(topk_title)
        self.bars_widget = QWidget()
        self.bars_layout = QVBoxLayout(self.bars_widget)
        self.bars_layout.setContentsMargins(0, 0, 0, 0)
        self.bars_layout.setSpacing(6)
        right.addWidget(self.bars_widget)
        right.addStretch()
        body.addLayout(right, 1)
        outer.addLayout(body, 1)

    @staticmethod
    def _btn_style(color):
        return f"""
            QPushButton {{ background-color: {color}20; color: {color}; border: 1px solid {color}80; border-radius: 8px; padding: 8px 16px; font-weight: bold; }}
            QPushButton:hover {{ background-color: {color}40; }}
        """

    def set_model(self, model):
        self.model = model

    def _load_image(self):
        path, _ = QFileDialog.getOpenFileName(self, "Выберите изображение", str(Path.home()), "Images (*.png *.jpg *.jpeg *.bmp *.webp)")
        if not path:
            return
        try:
            pil = Image.open(path).convert("L")
            self.file_lbl.setText(Path(path).name)
            self._run_inference(pil)
        except Exception as exc:
            QMessageBox.critical(self, "Ошибка", str(exc))

    def _demo_noise(self):
        arr = np.random.randint(0, 256, (64, 64), dtype=np.uint8)
        pil = Image.fromarray(arr, mode="L")
        self.file_lbl.setText("demo_noise.png")
        self._run_inference(pil)

    def _run_inference(self, pil):
        display = pil.copy()
        display.thumbnail((360, 360), Image.BILINEAR)
        arr = np.array(display)
        if len(arr.shape) == 2:
            arr_rgb = np.stack([arr] * 3, axis=-1)
        else:
            arr_rgb = arr
        pix = _numpy_to_qpixmap_rgb(arr_rgb)
        self.img_lbl.setPixmap(pix)
        try:
            top_class, top_prob, ranking = predict_pil(self.model, pil, self.device, top_k=5)
        except Exception as exc:
            QMessageBox.critical(self, "Ошибка инференса", str(exc))
            return
        self.class_lbl.setText(top_class)
        self.prob_lbl.setText(f"Confidence: {top_prob * 100:.2f}%")
        self._render_bars(ranking)

    def _render_bars(self, ranking):
        while self.bars_layout.count():
            item = self.bars_layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()
        max_prob = max((p for _, p in ranking), default=1.0)
        colors = ["#00d4ff", "#22d3ee", "#a855f7", "#f472b6", "#f59e0b"]
        for i, (cls, prob) in enumerate(ranking):
            row = QWidget()
            row_layout = QHBoxLayout(row)
            row_layout.setContentsMargins(0, 0, 0, 0)
            row_layout.setSpacing(8)
            name = QLabel(cls)
            name.setFixedWidth(110)
            name.setStyleSheet("color: #e2e8f0; font-weight: bold;")
            row_layout.addWidget(name)
            bar = QProgressBar()
            bar.setRange(0, 1000)
            bar.setValue(int(prob / max(max_prob, 1e-6) * 1000))
            bar.setTextVisible(False)
            color = colors[i % len(colors)]
            bar.setStyleSheet(f"""
                QProgressBar {{ background: #111827; border: 1px solid #1f2937; border-radius: 6px; height: 14px; }}
                QProgressBar::chunk {{ background: {color}; border-radius: 5px; }}
            """)
            row_layout.addWidget(bar, 1)
            val = QLabel(f"{prob * 100:.2f}%")
            val.setFixedWidth(70)
            val.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            val.setStyleSheet(f"color: {color}; font-weight: bold;")
            row_layout.addWidget(val)
            self.bars_layout.addWidget(row)

    def on_activated(self):
        pass