from pathlib import Path
from typing import Optional
import numpy as np
import torch
from PIL import Image
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont, QPixmap, QImage
from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QFileDialog, QFrame, QScrollArea, QMessageBox
from app.ml.cnn import SmallCNN
from app.ml.dataset import FASHION_MNIST_CLASSES, prepare_fashion_mnist
from app.visualization.feature_maps import pil_to_tensor_for_model

ASSETS_DIR = Path(__file__).resolve().parent.parent.parent / "assets"
DATA_DIR = ASSETS_DIR / "data"

def _numpy_to_qpixmap_gray(arr):
    h, w = arr.shape
    qimg = QImage(arr.data, w, h, w, QImage.Format.Format_Grayscale8).copy()
    return QPixmap.fromImage(qimg)

def _numpy_to_qpixmap_rgb(arr):
    h, w, ch = arr.shape
    if ch == 3:
        qimg = QImage(arr.data, w, h, 3 * w, QImage.Format.Format_RGB888).copy()
    else:
        qimg = QImage(arr.data, w, h, w, QImage.Format.Format_Grayscale8).copy()
    return QPixmap.fromImage(qimg)

class FeatureMapsPage(QWidget):
    def __init__(self, model, device, parent=None):
        super().__init__(parent)
        self.model = model
        self.device = device
        self.dataset = None
        outer = QVBoxLayout(self)
        outer.setContentsMargins(30, 30, 30, 30)
        header = QLabel("Feature Maps")
        header.setFont(QFont("Segoe UI", 24, QFont.Weight.Bold))
        header.setStyleSheet("color: #7dd3fc;")
        outer.addWidget(header)
        sub = QLabel("Активации промежуточных свёрточных слоёв")
        sub.setStyleSheet("color: #b5c5da; font-size: 11pt; margin-bottom: 16px;")
        outer.addWidget(sub)
        ctrl = QHBoxLayout()
        self.btn_load_img = QPushButton("📁  Load Image")
        self.btn_load_img.setStyleSheet(self._btn_style("#7dd3fc"))
        self.btn_load_img.clicked.connect(self._load_image)
        ctrl.addWidget(self.btn_load_img)
        self.btn_sample = QPushButton("🎲  Random Sample from Fashion-MNIST")
        self.btn_sample.setStyleSheet(self._btn_style("#b8a1ef"))
        self.btn_sample.clicked.connect(self._random_sample)
        ctrl.addWidget(self.btn_sample)
        ctrl.addStretch()
        outer.addLayout(ctrl)
        self.input_lbl = QLabel("Нет изображения")
        self.input_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.input_lbl.setFixedSize(224, 224)
        self.input_lbl.setStyleSheet("""
            QLabel { background: #1a2436; border: 1px solid #35445e; border-radius: 12px; color: #92a3ba; }
        """)
        outer.addWidget(self.input_lbl, 0, Qt.AlignmentFlag.AlignHCenter)
        self.pred_lbl = QLabel("")
        self.pred_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.pred_lbl.setStyleSheet("color: #7dd3fc; font-size: 12pt; font-weight: bold;")
        outer.addWidget(self.pred_lbl)
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setStyleSheet("QScrollArea { border: none; background: transparent; }")
        self.container = QWidget()
        self.container.setStyleSheet("background: transparent;")
        self.vlayout = QVBoxLayout(self.container)
        self.vlayout.setSpacing(16)
        self.scroll.setWidget(self.container)
        outer.addWidget(self.scroll, 1)

    @staticmethod
    def _btn_style(color):
        return f"""
            QPushButton {{ background-color: #35445e; color: {color}; border: 1px solid #35445e; border-radius: 8px; padding: 8px 16px; font-weight: bold; }}
            QPushButton:hover {{ background-color: #35445e; }}
        """

    def set_model(self, model):
        self.model = model

    def _load_image(self):
        path, _ = QFileDialog.getOpenFileName(self, "Выберите изображение", str(Path.home()), "Images (*.png *.jpg *.jpeg *.bmp *.webp)")
        if not path:
            return
        try:
            pil = Image.open(path).convert("L")
            self._process_pil(pil)
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
        pil = Image.fromarray(img_arr, mode="L")
        self._process_pil(pil, forced_label=FASHION_MNIST_CLASSES[label])

    def _process_pil(self, pil, forced_label=None):
        display = pil.resize((224, 224), Image.BILINEAR)
        arr = np.array(display)
        if len(arr.shape) == 2:
            arr_rgb = np.stack([arr] * 3, axis=-1)
        else:
            arr_rgb = arr
        pix = _numpy_to_qpixmap_rgb(arr_rgb)
        self.input_lbl.setPixmap(pix)
        tensor = pil_to_tensor_for_model(pil)
        self.model.eval()
        self.model.clear_feature_maps()
        with torch.no_grad():
            logits = self.model(tensor.to(self.device))
            probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
        pred_idx = int(np.argmax(probs))
        pred_name = FASHION_MNIST_CLASSES[pred_idx]
        prob = float(probs[pred_idx]) * 100.0
        if forced_label is not None:
            self.pred_lbl.setText(f"Ground truth: {forced_label}   |   Predicted: {pred_name} ({prob:.1f}%)")
        else:
            self.pred_lbl.setText(f"Predicted: {pred_name} ({prob:.1f}%)")
        maps = self.model.get_feature_maps()
        self._render_maps(maps)

    def _render_maps(self, maps):
        while self.vlayout.count():
            item = self.vlayout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()
        for name, grid in maps.items():
            frame = QFrame()
            frame.setProperty("panel", True)
            frame.setStyleSheet("""
                QFrame[panel="true"] { background: rgba(25, 35, 55, 0.95); border: 1px solid rgba(125, 211, 252, 0.2); border-radius: 12px; }
            """)
            lay = QVBoxLayout(frame)
            lay.setContentsMargins(16, 12, 16, 12)
            title = QLabel(f"Layer: {name}   (channels grid)")
            title.setStyleSheet("color: #b8a1ef; font-size: 12pt; font-weight: bold;")
            lay.addWidget(title)
            lbl = QLabel()
            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            pix = _numpy_to_qpixmap_gray(grid)
            pix = pix.scaled(640, 240, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
            lbl.setPixmap(pix)
            lay.addWidget(lbl, 0, Qt.AlignmentFlag.AlignHCenter)
            info = QLabel(f"Grid shape: {grid.shape[1]}×{grid.shape[0]} px")
            info.setStyleSheet("color: #b5c5da; font-size: 10pt;")
            info.setAlignment(Qt.AlignmentFlag.AlignCenter)
            lay.addWidget(info)
            self.vlayout.addWidget(frame)

    def on_activated(self):
        pass