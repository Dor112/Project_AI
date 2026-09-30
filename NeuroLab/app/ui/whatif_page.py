from pathlib import Path
import numpy as np
import torch
from PIL import Image, ImageFilter, ImageEnhance, ImageOps
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont, QPixmap, QImage
from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QFileDialog, QFrame, QMessageBox, QScrollArea
from app.ml.cnn import SmallCNN
from app.ml.inference import predict_pil
from app.ml.dataset import FASHION_MNIST_CLASSES, prepare_fashion_mnist

ASSETS_DIR = Path(__file__).resolve().parent.parent.parent / "assets"
DATA_DIR = ASSETS_DIR / "data"

def _numpy_to_qpixmap_gray(arr):
    h, w = arr.shape
    qimg = QImage(arr.data, w, h, w, QImage.Format.Format_Grayscale8).copy()
    return QPixmap.fromImage(qimg)

class WhatIfPage(QWidget):
    def __init__(self, model, device, parent=None):
        super().__init__(parent)
        self.model = model
        self.device = device
        self.original_image = None
        self.dataset = None
        outer = QVBoxLayout(self)
        outer.setContentsMargins(30, 30, 30, 30)
        header = QLabel("What If?")
        header.setFont(QFont("Segoe UI", 28, QFont.Weight.Bold))
        header.setStyleSheet("color: #00d4ff;")
        outer.addWidget(header)
        sub = QLabel("Анализ устойчивости модели к изменениям изображения")
        sub.setStyleSheet("color: #94a3b8; font-size: 11pt; margin-bottom: 16px;")
        outer.addWidget(sub)
        ctrl = QHBoxLayout()
        self.btn_load = QPushButton("📁  Load Image")
        self.btn_load.setStyleSheet(self._btn_style("#00d4ff"))
        self.btn_load.clicked.connect(self._load_image)
        ctrl.addWidget(self.btn_load)
        self.btn_sample = QPushButton("🎲  Random Sample")
        self.btn_sample.setStyleSheet(self._btn_style("#a855f7"))
        self.btn_sample.clicked.connect(self._random_sample)
        ctrl.addWidget(self.btn_sample)
        ctrl.addStretch()
        outer.addLayout(ctrl)
        transforms_layout = QHBoxLayout()
        self.transforms = [("Original", None), ("Blur", "blur"), ("Noise", "noise"), ("Rotate", "rotate"), ("Invert", "invert"), ("Brightness", "brightness"), ("Contrast", "contrast")]
        self.transform_buttons = []
        for name, _ in self.transforms:
            btn = QPushButton(name)
            btn.setStyleSheet(self._btn_style("#f59e0b"))
            btn.clicked.connect(lambda checked, n=name: self._apply_transform(n))
            transforms_layout.addWidget(btn)
            self.transform_buttons.append(btn)
        outer.addLayout(transforms_layout)
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setStyleSheet("QScrollArea { border: none; background: transparent; }")
        self.container = QWidget()
        self.container.setStyleSheet("background: transparent;")
        self.hlayout = QHBoxLayout(self.container)
        self.hlayout.setSpacing(16)
        self.scroll.setWidget(self.container)
        outer.addWidget(self.scroll, 1)
        self.status = QLabel("Загрузите изображение для анализа.")
        self.status.setStyleSheet("color: #94a3b8; padding-top: 8px;")
        outer.addWidget(self.status)

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
            self.original_image = Image.open(path).convert("L")
            self._show_all_transforms()
        except Exception as exc:
            QMessageBox.critical(self, "Ошибка", str(exc))

    def _random_sample(self):
        if self.dataset is None:
            try:
                _, _, _, train_full = prepare_fashion_mnist(root=str(DATA_DIR), batch_size=64, val_fraction=0.1, download=True, num_workers=0)
                self.dataset = train_full
            except Exception as exc:
                QMessageBox.critical(self, "Ошибка", str(exc))
                return
        idx = int(np.random.randint(0, len(self.dataset)))
        img_tensor, label = self.dataset[idx]
        img_arr = (img_tensor.squeeze().numpy() * 127.5 + 127.5).astype(np.uint8)
        self.original_image = Image.fromarray(img_arr, mode="L")
        self._show_all_transforms()

    def _apply_transform(self, name):
        if self.original_image is None:
            QMessageBox.warning(self, "Warning", "Сначала загрузите изображение.")
            return
        self._show_all_transforms()

    def _show_all_transforms(self):
        while self.hlayout.count():
            item = self.hlayout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()
        for name, transform_type in self.transforms:
            if transform_type is None:
                img = self.original_image.copy()
            else:
                img = self._transform_image(self.original_image, transform_type)
            frame = self._create_frame(name, img)
            self.hlayout.addWidget(frame)

    def _transform_image(self, img, transform_type):
        if transform_type == "blur":
            return img.filter(ImageFilter.GaussianBlur(radius=2))
        elif transform_type == "noise":
            arr = np.array(img, dtype=np.float32)
            noise = np.random.normal(0, 30, arr.shape)
            arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
            return Image.fromarray(arr, mode="L")
        elif transform_type == "rotate":
            return img.rotate(15, expand=False)
        elif transform_type == "invert":
            return ImageOps.invert(img)
        elif transform_type == "brightness":
            enhancer = ImageEnhance.Brightness(img)
            return enhancer.enhance(1.5)
        elif transform_type == "contrast":
            enhancer = ImageEnhance.Contrast(img)
            return enhancer.enhance(1.5)
        return img

    def _create_frame(self, name, img):
        frame = QFrame()
        frame.setStyleSheet("""
            QFrame { background: rgba(15, 23, 42, 0.6); border: 1px solid rgba(0, 212, 255, 0.2); border-radius: 12px; }
        """)
        frame.setFixedWidth(280)
        lay = QVBoxLayout(frame)
        lay.setContentsMargins(12, 12, 12, 12)
        lay.setSpacing(8)
        title = QLabel(name)
        title.setStyleSheet("color: #a855f7; font-size: 12pt; font-weight: bold;")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lay.addWidget(title)
        display = img.resize((200, 200), Image.BILINEAR)
        arr = np.array(display)
        pix = _numpy_to_qpixmap_gray(arr)
        img_lbl = QLabel()
        img_lbl.setPixmap(pix)
        img_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lay.addWidget(img_lbl, 0, Qt.AlignmentFlag.AlignHCenter)
        try:
            top_class, top_prob, _ = predict_pil(self.model, img, self.device, top_k=1)
            pred_text = f"{top_class}\n{top_prob * 100:.1f}%"
        except Exception:
            pred_text = "Error"
        pred_lbl = QLabel(pred_text)
        pred_lbl.setStyleSheet("color: #00d4ff; font-size: 11pt; font-weight: bold;")
        pred_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lay.addWidget(pred_lbl)
        return frame

    def on_activated(self):
        pass