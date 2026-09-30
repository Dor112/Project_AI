from pathlib import Path
import numpy as np
from PIL import Image
from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap, QImage, QFont
from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QScrollArea, QFrame, QGridLayout, QMessageBox, QSpinBox, QApplication
from app.ml.dataset import FASHION_MNIST_CLASSES, prepare_fashion_mnist

ASSETS_DIR = Path(__file__).resolve().parent.parent.parent / "assets"
DATA_DIR = ASSETS_DIR / "data"

def _numpy_to_qpixmap_gray(arr):
    h, w = arr.shape
    qimg = QImage(arr.data, w, h, w, QImage.Format.Format_Grayscale8).copy()
    return QPixmap.fromImage(qimg)

class DatasetPage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.dataset = None
        self.current_page = 0
        self.page_size = 60
        outer = QVBoxLayout(self)
        outer.setContentsMargins(30, 30, 30, 30)
        header = QLabel("Dataset Viewer")
        header.setFont(QFont("Segoe UI", 28, QFont.Weight.Bold))
        header.setStyleSheet("color: #00d4ff;")
        outer.addWidget(header)
        sub = QLabel("Просмотр обучающей выборки Fashion-MNIST")
        sub.setStyleSheet("color: #94a3b8; font-size: 11pt; margin-bottom: 16px;")
        outer.addWidget(sub)
        controls = QHBoxLayout()
        self.btn_load = QPushButton("⬇  Load / Download Fashion-MNIST")
        self.btn_load.setStyleSheet(self._btn_style("#00d4ff"))
        self.btn_load.clicked.connect(self._load_dataset)
        controls.addWidget(self.btn_load)
        self.btn_prev = QPushButton("◀  Prev")
        self.btn_prev.setStyleSheet(self._btn_style("#a855f7"))
        self.btn_prev.clicked.connect(self._prev_page)
        controls.addWidget(self.btn_prev)
        self.lbl_page = QLabel("Page 0 / 0")
        self.lbl_page.setStyleSheet("color: #cdd6f4; font-size: 11pt; padding: 0 12px;")
        controls.addWidget(self.lbl_page)
        self.btn_next = QPushButton("Next  ▶")
        self.btn_next.setStyleSheet(self._btn_style("#a855f7"))
        self.btn_next.clicked.connect(self._next_page)
        controls.addWidget(self.btn_next)
        controls.addStretch()
        lbl_size = QLabel("Samples per page:")
        lbl_size.setStyleSheet("color: #94a3b8;")
        controls.addWidget(lbl_size)
        self.spin_size = QSpinBox()
        self.spin_size.setRange(12, 240)
        self.spin_size.setSingleStep(12)
        self.spin_size.setValue(self.page_size)
        self.spin_size.setStyleSheet("color: #cdd6f4; background: #111827; padding: 4px 8px; border-radius: 6px;")
        self.spin_size.valueChanged.connect(self._on_size_changed)
        controls.addWidget(self.spin_size)
        outer.addLayout(controls)
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setStyleSheet("QScrollArea { border: none; background: transparent; }")
        self.grid_container = QWidget()
        self.grid_container.setStyleSheet("background: transparent;")
        self.grid_layout = QGridLayout(self.grid_container)
        self.grid_layout.setSpacing(12)
        self.scroll.setWidget(self.grid_container)
        outer.addWidget(self.scroll, 1)
        self.status = QLabel("Датасет не загружен. Нажмите «Load / Download Fashion-MNIST».")
        self.status.setStyleSheet("color: #94a3b8; padding-top: 8px;")
        outer.addWidget(self.status)

    @staticmethod
    def _btn_style(color):
        return f"""
            QPushButton {{ background-color: {color}20; color: {color}; border: 1px solid {color}80; border-radius: 8px; padding: 8px 16px; font-weight: bold; }}
            QPushButton:hover {{ background-color: {color}40; }}
        """

    def _load_dataset(self):
        try:
            self.status.setText("Загрузка Fashion-MNIST...")
            QApplication.processEvents()
            _, _, _, train_full = prepare_fashion_mnist(root=str(DATA_DIR), batch_size=64, val_fraction=0.1, download=True, num_workers=0)
            self.dataset = train_full
            self.current_page = 0
            self._render_page()
            self.status.setText(f"Загружено: {len(self.dataset)} изображений. Классы: {', '.join(FASHION_MNIST_CLASSES)}.")
        except Exception as exc:
            QMessageBox.critical(self, "Ошибка загрузки", str(exc))
            self.status.setText(f"Ошибка: {exc}")

    def _on_size_changed(self, value):
        self.page_size = value
        self.current_page = 0
        if self.dataset is not None:
            self._render_page()

    def _prev_page(self):
        if self.dataset is None:
            return
        if self.current_page > 0:
            self.current_page -= 1
            self._render_page()

    def _next_page(self):
        if self.dataset is None:
            return
        max_page = (len(self.dataset) - 1) // self.page_size
        if self.current_page < max_page:
            self.current_page += 1
            self._render_page()

    def _render_page(self):
        while self.grid_layout.count():
            item = self.grid_layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()
        cols = 10
        start = self.current_page * self.page_size
        end = min(start + self.page_size, len(self.dataset))
        for idx, real_idx in enumerate(range(start, end)):
            img_tensor, label = self.dataset[real_idx]
            img_arr = (img_tensor.squeeze().numpy() * 127.5 + 127.5).astype(np.uint8)
            frame = QFrame()
            frame.setStyleSheet("""
                QFrame { background-color: #111827; border: 1px solid #1f2937; border-radius: 8px; }
                QFrame:hover { border: 1px solid #00d4ff; }
            """)
            frame.setFixedSize(110, 140)
            layout = QVBoxLayout(frame)
            layout.setContentsMargins(6, 6, 6, 6)
            layout.setSpacing(4)
            lbl_img = QLabel()
            lbl_img.setFixedSize(96, 96)
            lbl_img.setAlignment(Qt.AlignmentFlag.AlignCenter)
            pix = _numpy_to_qpixmap_gray(img_arr).scaled(96, 96, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
            lbl_img.setPixmap(pix)
            layout.addWidget(lbl_img, 0, Qt.AlignmentFlag.AlignHCenter)
            lbl_class = QLabel(FASHION_MNIST_CLASSES[label])
            lbl_class.setAlignment(Qt.AlignmentFlag.AlignCenter)
            lbl_class.setStyleSheet("color: #00d4ff; font-size: 9pt; font-weight: bold;")
            layout.addWidget(lbl_class)
            lbl_idx = QLabel(f"#{real_idx}")
            lbl_idx.setAlignment(Qt.AlignmentFlag.AlignCenter)
            lbl_idx.setStyleSheet("color: #64748b; font-size: 8pt;")
            layout.addWidget(lbl_idx)
            self.grid_layout.addWidget(frame, idx // cols, idx % cols)
        total_pages = max(1, (len(self.dataset) + self.page_size - 1) // self.page_size)
        self.lbl_page.setText(f"Page {self.current_page + 1} / {total_pages}")

    def on_activated(self):
        pass