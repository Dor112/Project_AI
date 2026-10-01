from pathlib import Path
import numpy as np
import torch
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont
from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel, QPushButton, QMessageBox, QApplication
from app.ml.cnn import SmallCNN
from app.ml.dataset import FASHION_MNIST_CLASSES, prepare_fashion_mnist
from app.visualization.charts import ConfusionMatrixWidget

ASSETS_DIR = Path(__file__).resolve().parent.parent.parent / "assets"
DATA_DIR = ASSETS_DIR / "data"

class ConfusionMatrixPage(QWidget):
    def __init__(self, model, device, parent=None):
        super().__init__(parent)
        self.model = model
        self.device = device
        outer = QVBoxLayout(self)
        outer.setContentsMargins(30, 30, 30, 30)
        header = QLabel("Confusion Matrix")
        header.setFont(QFont("Segoe UI", 24, QFont.Weight.Bold))
        header.setStyleSheet("color: #7dd3fc;")
        outer.addWidget(header)
        sub = QLabel("Матрица ошибок на валидационной выборке")
        sub.setStyleSheet("color: #b5c5da; font-size: 11pt; margin-bottom: 16px;")
        outer.addWidget(sub)
        self.btn_compute = QPushButton("🔄  Compute Confusion Matrix")
        self.btn_compute.setStyleSheet(self._btn_style("#b8a1ef"))
        self.btn_compute.clicked.connect(self._compute_matrix)
        outer.addWidget(self.btn_compute, 0, Qt.AlignmentFlag.AlignLeft)
        self.cm_widget = ConfusionMatrixWidget()
        self.cm_widget.setMinimumHeight(500)
        outer.addWidget(self.cm_widget, 1)
        self.status = QLabel("Нажмите кнопку для вычисления матрицы ошибок.")
        self.status.setStyleSheet("color: #b5c5da; padding-top: 8px;")
        outer.addWidget(self.status)

    @staticmethod
    def _btn_style(color):
        return f"""
            QPushButton {{ background-color: #35445e; color: {color}; border: 1px solid #35445e; border-radius: 8px; padding: 8px 16px; font-weight: bold; }}
            QPushButton:hover {{ background-color: #35445e; }}
        """

    def set_model(self, model):
        self.model = model

    def _compute_matrix(self):
        try:
            self.status.setText("Загрузка данных...")
            QApplication.processEvents()
            _, val_loader, _, _ = prepare_fashion_mnist(root=str(DATA_DIR), batch_size=64, val_fraction=0.1, download=True, num_workers=0, test_subset_size=2000)
        except Exception as exc:
            QMessageBox.critical(self, "Ошибка", str(exc))
            self.status.setText(f"Ошибка: {exc}")
            return
        n_classes = len(FASHION_MNIST_CLASSES)
        cm = np.zeros((n_classes, n_classes), dtype=np.int64)
        self.model.eval()
        self.status.setText("Вычисление матрицы ошибок...")
        QApplication.processEvents()
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = inputs.to(self.device, non_blocking=True)
                outputs = self.model(inputs)
                preds = outputs.argmax(dim=1).cpu().numpy()
                tg = targets.numpy()
                for p, t in zip(preds, tg):
                    cm[int(t), int(p)] += 1
        self.cm_widget.set_matrix(cm, list(FASHION_MNIST_CLASSES))
        self.status.setText(f"Матрица ошибок вычислена. Всего samples: {cm.sum()}")

    def on_activated(self):
        pass