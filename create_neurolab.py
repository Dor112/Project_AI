#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
create_neurolab.py
==================
Генератор проекта NeuroLab — учебная desktop-лаборатория CNN,
оптимизированная для CPU (Intel 13th gen, без NVIDIA GPU).

Dataset: Fashion-MNIST (28×28, 10 классов)
Model: Small CNN (~213K parameters)
UI: PySide6, dark futuristic theme
"""

from __future__ import annotations

import os
import sys
import shutil
import importlib
import traceback
from pathlib import Path

# ============================================================================
# Полная структура проекта
# ============================================================================

FILES: dict[str, str] = {}

# ============================================================================
# app/main.py — точка входа
# ============================================================================
FILES["NeuroLab/app/main.py"] = r'''#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Точка входа приложения NeuroLab."""

import os
import sys
import multiprocessing as mp
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if sys.platform in ("win32", "darwin"):
    mp.freeze_support()

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("QT_API", "pyside6")


def main() -> int:
    from PySide6.QtWidgets import QApplication
    from PySide6.QtCore import Qt
    from PySide6.QtGui import QFont

    try:
        QApplication.setHighDpiScaleFactorRoundingPolicy(
            Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
        )
    except AttributeError:
        pass

    app = QApplication(sys.argv)
    app.setApplicationName("NeuroLab")
    app.setApplicationDisplayName("NeuroLab — AI Laboratory")
    app.setOrganizationName("NeuroLab")

    font = QFont("Segoe UI", 10)
    font.setStyleHint(QFont.SansSerif)
    app.setFont(font)

    from app.ui.main_window import MainWindow
    window = MainWindow()
    window.show()

    return app.exec()


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        import traceback
        traceback.print_exc()
        sys.exit(1)
'''

# ============================================================================
# app/ml/cnn.py — маленькая CNN для Fashion-MNIST
# ============================================================================
FILES["NeuroLab/app/ml/cnn.py"] = r'''# -*- coding: utf-8 -*-
"""Маленькая CNN для Fashion-MNIST (28×28 grayscale)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SmallCNN(nn.Module):
    """
    Маленькая учебная CNN для Fashion-MNIST.

    Архитектура:
        Input: 1×28×28
        Conv2d(1, 16, 3, padding=1) -> ReLU -> MaxPool(2) -> 16×14×14
        Conv2d(16, 32, 3, padding=1) -> ReLU -> MaxPool(2) -> 32×7×7
        Flatten -> 32*7*7 = 1568
        Linear(1568, 128) -> ReLU
        Linear(128, 10)

    Параметры: ~213K (очень лёгкая модель)
    """

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()

        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2, 2)

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(128, num_classes)

        self._feature_maps: dict[str, torch.Tensor] = {}
        self._register_hooks()

    def _register_hooks(self) -> None:
        def make_hook(name: str):
            def hook(_module, _input, output):
                self._feature_maps[name] = output.detach()
            return hook

        self.conv1.register_forward_hook(make_hook("conv1"))
        self.conv2.register_forward_hook(make_hook("conv2"))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)

        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.fc2(x)
        return x

    def get_feature_maps(self) -> dict[str, torch.Tensor]:
        return dict(self._feature_maps)

    def clear_feature_maps(self) -> None:
        self._feature_maps.clear()


def build_model(num_classes: int = 10) -> SmallCNN:
    return SmallCNN(num_classes=num_classes)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
'''

# ============================================================================
# app/ml/dataset.py — Fashion-MNIST
# ============================================================================
FILES["NeuroLab/app/ml/dataset.py"] = r'''# -*- coding: utf-8 -*-
"""Fashion-MNIST dataset loader."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple, Optional

import torch
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import datasets, transforms


FASHION_MNIST_CLASSES = (
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
)


def get_transforms() -> transforms.Compose:
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])


def prepare_fashion_mnist(
    root: str | Path,
    batch_size: int = 64,
    val_fraction: float = 0.1,
    download: bool = True,
    num_workers: int = 0,
    train_subset_size: Optional[int] = None,
    test_subset_size: Optional[int] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader, datasets.FashionMNIST]:
    """
    Загружает Fashion-MNIST и создаёт DataLoader'ы.

    Args:
        train_subset_size: если указан, использует только часть training data
        test_subset_size: если указан, использует только часть test data
    """
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)

    transform = get_transforms()

    train_full = datasets.FashionMNIST(
        root=str(root),
        train=True,
        download=download,
        transform=transform,
    )

    test_full = datasets.FashionMNIST(
        root=str(root),
        train=False,
        download=download,
        transform=transform,
    )

    # Subsets для Quick Demo
    if train_subset_size is not None and train_subset_size < len(train_full):
        indices = list(range(train_subset_size))
        train_full = Subset(train_full, indices)

    if test_subset_size is not None and test_subset_size < len(test_full):
        indices = list(range(test_subset_size))
        test_full = Subset(test_full, indices)

    # Split на train/val
    n_total = len(train_full)
    n_val = max(1, int(n_total * val_fraction))
    n_train = n_total - n_val

    train_set, val_set = random_split(
        train_full,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(42),
    )

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
    )

    test_loader = DataLoader(
        test_full,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
    )

    return train_loader, val_loader, test_loader, train_full
'''

# ============================================================================
# app/ml/trainer.py — логика обучения
# ============================================================================
FILES["NeuroLab/app/ml/trainer.py"] = r'''# -*- coding: utf-8 -*-
"""Логика обучения CNN."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from app.ml.cnn import SmallCNN


@dataclass
class TrainingStats:
    epoch: int = 0
    train_loss: float = 0.0
    train_acc: float = 0.0
    val_loss: float = 0.0
    val_acc: float = 0.0
    elapsed: float = 0.0
    history: dict = field(default_factory=lambda: {
        "train_loss": [], "train_acc": [],
        "val_loss": [], "val_acc": [],
    })


class Trainer:
    """Тренеровщик CNN с поддержкой паузы/остановки."""

    def __init__(
        self,
        model: SmallCNN,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
    ) -> None:
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(
            self.model.parameters(), lr=learning_rate, weight_decay=weight_decay,
        )
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.1)
        self.stats = TrainingStats()

    def train_epoch(
        self,
        step_callback: Optional[Callable[[TrainingStats, int, int], None]] = None,
        should_stop: Optional[Callable[[], bool]] = None,
        should_pause: Optional[Callable[[], bool]] = None,
    ) -> TrainingStats:
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        n_batches = len(self.train_loader)

        t0 = time.time()
        for batch_idx, (inputs, targets) in enumerate(self.train_loader):
            if should_stop and should_stop():
                break
            while should_pause and should_pause():
                time.sleep(0.1)
                if should_stop and should_stop():
                    break

            inputs = inputs.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)

            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            if step_callback is not None and (batch_idx % max(1, n_batches // 10) == 0 or batch_idx == n_batches - 1):
                self.stats.train_loss = running_loss / max(1, total)
                self.stats.train_acc = 100.0 * correct / max(1, total)
                step_callback(self.stats, batch_idx + 1, n_batches)

        self.stats.train_loss = running_loss / max(1, total)
        self.stats.train_acc = 100.0 * correct / max(1, total)
        self.stats.elapsed = time.time() - t0
        return self.stats

    def validate(self) -> TrainingStats:
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, targets in self.val_loader:
                inputs = inputs.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                running_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        self.stats.val_loss = running_loss / max(1, total)
        self.stats.val_acc = 100.0 * correct / max(1, total)
        return self.stats

    def step_scheduler(self) -> None:
        self.scheduler.step()

    def save_checkpoint(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "stats": {
                "epoch": self.stats.epoch,
                "train_loss": self.stats.train_loss,
                "train_acc": self.stats.train_acc,
                "val_loss": self.stats.val_loss,
                "val_acc": self.stats.val_acc,
            },
        }, path)

    def load_checkpoint(self, path: str | Path) -> bool:
        path = Path(path)
        if not path.exists():
            return False
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["model_state_dict"])
        if "optimizer_state_dict" in ckpt:
            try:
                self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            except Exception:
                pass
        if "stats" in ckpt:
            for k, v in ckpt["stats"].items():
                setattr(self.stats, k, v)
        return True
'''

# ============================================================================
# app/ml/inference.py — инференс
# ============================================================================
FILES["NeuroLab/app/ml/inference.py"] = r'''# -*- coding: utf-8 -*-
"""Инференс: предсказание класса для произвольного изображения."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple, List

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from app.ml.cnn import SmallCNN
from app.ml.dataset import FASHION_MNIST_CLASSES


def load_image_as_tensor(path: str | Path, size: int = 28) -> torch.Tensor:
    """Загружает изображение, приводит к 28×28 grayscale и нормализует."""
    img = Image.open(path).convert("L")  # Grayscale
    img = img.resize((size, size), Image.BILINEAR)
    arr = np.array(img, dtype=np.float32) / 255.0
    # Normalize to [-1, 1]
    arr = (arr - 0.5) / 0.5
    return torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)  # (1, 1, 28, 28)


def predict(
    model: SmallCNN,
    image_tensor: torch.Tensor,
    device: torch.device,
    top_k: int = 5,
) -> Tuple[str, float, List[Tuple[str, float]]]:
    """Возвращает (top_class, top_prob, [(class, prob), ...])."""
    model.eval()
    with torch.no_grad():
        x = image_tensor.to(device)
        logits = model(x)
        probs = F.softmax(logits, dim=1)[0].cpu().numpy()
    top_k = max(1, min(top_k, len(probs)))
    idxs = np.argsort(probs)[::-1][:top_k]
    top_class = FASHION_MNIST_CLASSES[idxs[0]]
    top_prob = float(probs[idxs[0]])
    ranking = [(FASHION_MNIST_CLASSES[i], float(probs[i])) for i in idxs]
    return top_class, top_prob, ranking


def predict_pil(
    model: SmallCNN,
    pil_image: Image.Image,
    device: torch.device,
    top_k: int = 5,
) -> Tuple[str, float, List[Tuple[str, float]]]:
    """Инференс из PIL.Image."""
    size = 28
    img = pil_image.convert("L").resize((size, size), Image.BILINEAR)
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = (arr - 0.5) / 0.5
    tensor = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)
    return predict(model, tensor, device, top_k=top_k)
'''

# ============================================================================
# app/ml/hardware.py — определение оборудования
# ============================================================================
FILES["NeuroLab/app/ml/hardware.py"] = r'''# -*- coding: utf-8 -*-
"""Определение характеристик системы."""

from __future__ import annotations

import os
import platform
import torch


def get_cpu_name() -> str:
    """Получает название CPU."""
    try:
        import cpuinfo
        info = cpuinfo.get_cpu_info()
        return info.get("brand_raw", "Unknown CPU")
    except ImportError:
        return platform.processor() or "Unknown CPU"


def get_logical_threads() -> int:
    """Получает количество логических потоков CPU."""
    return os.cpu_count() or 1


def get_ram_gb() -> float:
    """Получает объём RAM в ГБ."""
    try:
        import psutil
        return psutil.virtual_memory().total / (1024**3)
    except ImportError:
        return 0.0


def get_gpu_info() -> str:
    """Получает информацию о GPU."""
    if torch.cuda.is_available():
        return torch.cuda.get_device_name(0)
    return "None (CPU only)"


def get_cuda_available() -> bool:
    """Проверяет доступность CUDA."""
    return torch.cuda.is_available()


def get_pytorch_device() -> str:
    """Возвращает устройство PyTorch."""
    return "CUDA" if torch.cuda.is_available() else "CPU"


def get_optimal_threads() -> int:
    """Возвращает оптимальное количество потоков для обучения."""
    total = get_logical_threads()
    # Оставляем 2 потока для системы
    return max(2, total - 2)
'''

# ============================================================================
# app/visualization/charts.py — графики
# ============================================================================
FILES["NeuroLab/app/visualization/charts.py"] = r'''# -*- coding: utf-8 -*-
"""Графики на базе pyqtgraph."""

from __future__ import annotations

from typing import List

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt


def apply_dark_theme() -> None:
    pg.setConfigOptions(antialias=True, background="#0a0e1a", foreground="#cdd6f4")


class LossChart(pg.PlotWidget):
    """График loss: train и val."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setTitle("Loss", color="#cdd6f4", size="11pt")
        self.setLabel("bottom", "Epoch")
        self.setLabel("left", "Loss")
        self.showGrid(x=True, y=True, alpha=0.15)
        self.addLegend(offset=(10, 10))
        pen_train = pg.mkPen(color="#00d4ff", width=2)
        pen_val = pg.mkPen(color="#f472b6", width=2)
        self.train_curve = self.plot([], [], pen=pen_train, name="Train")
        self.val_curve = self.plot([], [], pen=pen_val, name="Val")

    def update_data(self, train_loss: List[float], val_loss: List[float]) -> None:
        x = list(range(1, len(train_loss) + 1))
        self.train_curve.setData(x, train_loss)
        xv = list(range(1, len(val_loss) + 1))
        self.val_curve.setData(xv, val_loss)
        if train_loss or val_loss:
            self.enableAutoRange()


class AccuracyChart(pg.PlotWidget):
    """График accuracy: train и val."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setTitle("Accuracy, %", color="#cdd6f4", size="11pt")
        self.setLabel("bottom", "Epoch")
        self.setLabel("left", "Accuracy")
        self.showGrid(x=True, y=True, alpha=0.15)
        self.addLegend(offset=(10, 10))
        pen_train = pg.mkPen(color="#22d3ee", width=2)
        pen_val = pg.mkPen(color="#a855f7", width=2)
        self.train_curve = self.plot([], [], pen=pen_train, name="Train")
        self.val_curve = self.plot([], [], pen=pen_val, name="Val")

    def update_data(self, train_acc: List[float], val_acc: List[float]) -> None:
        x = list(range(1, len(train_acc) + 1))
        self.train_curve.setData(x, train_acc)
        xv = list(range(1, len(val_acc) + 1))
        self.val_curve.setData(xv, val_acc)
        if train_acc or val_acc:
            self.enableAutoRange()


class ConfusionMatrixWidget(pg.GraphicsLayoutWidget):
    """Confusion matrix как heatmap."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.view = self.addViewBox()
        self.view.setAspectLocked(True)
        self.image_item = pg.ImageItem()
        self.view.addItem(self.image_item)
        self.text_items: List[pg.TextItem] = []

    def set_matrix(self, matrix: np.ndarray, labels: List[str]) -> None:
        for t in self.text_items:
            self.view.removeItem(t)
        self.text_items.clear()

        row_sums = matrix.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        norm = matrix / row_sums

        lut = np.zeros((256, 4), dtype=np.ubyte)
        for i in range(256):
            t = i / 255.0
            r = int(10 + t * (0 - 10))
            g = int(14 + t * (212 - 14))
            b = int(26 + t * (255 - 26))
            lut[i] = (max(0, min(255, r)), max(0, min(255, g)), max(0, min(255, b)), 255)
        self.image_item.setLookupTable(lut)

        self.image_item.setImage(norm.T, autoLevels=False, levels=(0.0, 1.0))
        n = len(labels)
        self.image_item.setRect(0, 0, n, n)
        self.view.setRange(xRange=(0, n), yRange=(0, n))

        for i, lab in enumerate(labels):
            t_bottom = pg.TextItem(lab, color="#cdd6f4", anchor=(1, 0.5))
            t_bottom.setPos(i + 0.5, n + 0.2)
            self.view.addItem(t_bottom)
            self.text_items.append(t_bottom)

            t_left = pg.TextItem(lab, color="#cdd6f4", anchor=(1, 0.5))
            t_left.setPos(-0.2, i + 0.5)
            self.view.addItem(t_left)
            self.text_items.append(t_left)

        for i in range(n):
            for j in range(n):
                val = float(matrix[i, j])
                txt = pg.TextItem(str(int(val)) if val >= 1 else "", color="#0a0e1a", anchor=(0.5, 0.5))
                txt.setPos(j + 0.5, i + 0.5)
                self.view.addItem(txt)
                self.text_items.append(txt)
'''

# ============================================================================
# app/visualization/feature_maps.py
# ============================================================================
FILES["NeuroLab/app/visualization/feature_maps.py"] = r'''# -*- coding: utf-8 -*-
"""Утилиты для извлечения и отображения feature maps."""

from __future__ import annotations

from typing import Dict

import numpy as np
import torch
from PIL import Image

from app.ml.cnn import SmallCNN


def activation_to_grid(activation: torch.Tensor, max_channels: int = 16) -> np.ndarray:
    """Преобразует тензор активаций (1, C, H, W) в сетку изображений."""
    act = activation[0].detach().cpu()
    c, h, w = act.shape
    c = min(c, max_channels)
    act = act[:c]

    mins = act.flatten(1).min(dim=1).values.view(-1, 1, 1)
    maxs = act.flatten(1).max(dim=1).values.view(-1, 1, 1)
    denom = (maxs - mins).clamp(min=1e-6)
    act = (act - mins) / denom

    cols = int(np.ceil(np.sqrt(c)))
    rows = int(np.ceil(c / cols))
    pad = 1
    grid_h = rows * (h + pad) + pad
    grid_w = cols * (w + pad) + pad
    grid = np.zeros((grid_h, grid_w), dtype=np.float32)

    arr = act.numpy()
    for i in range(c):
        r = i // cols
        cc = i % cols
        y0 = r * (h + pad) + pad
        x0 = cc * (w + pad) + pad
        grid[y0:y0 + h, x0:x0 + w] = arr[i]

    return (grid * 255.0).astype(np.uint8)


def get_feature_maps_for_image(
    model: SmallCNN,
    image_tensor: torch.Tensor,
    device: torch.device,
) -> Dict[str, np.ndarray]:
    """Прогоняет одно изображение через сеть и возвращает feature maps."""
    model.eval()
    model.clear_feature_maps()
    with torch.no_grad():
        _ = model(image_tensor.to(device))
    maps = model.get_feature_maps()
    return {name: activation_to_grid(act) for name, act in maps.items()}


def pil_to_tensor_for_model(pil_image: Image.Image) -> torch.Tensor:
    """PIL -> нормализованный тензор (1, 1, 28, 28)."""
    img = pil_image.convert("L").resize((28, 28), Image.BILINEAR)
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = (arr - 0.5) / 0.5
    return torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)
'''

# ============================================================================
# app/visualization/network_view.py — визуализация архитектуры
# ============================================================================
FILES["NeuroLab/app/visualization/network_view.py"] = r'''# -*- coding: utf-8 -*-
"""Визуализация архитектуры CNN в виде графа слоёв."""

from __future__ import annotations

from typing import List

from PySide6.QtCore import Qt, QRectF, QPointF
from PySide6.QtGui import (
    QPainter, QColor, QLinearGradient, QPen, QBrush, QFont,
)
from PySide6.QtWidgets import QWidget


class LayerInfo:
    def __init__(self, name: str, kind: str, shape: str, color: str) -> None:
        self.name = name
        self.kind = kind
        self.shape = shape
        self.color = color


def build_architecture_layers() -> List[LayerInfo]:
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
    """Отображает архитектуру сети в виде вертикального графа слоёв."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.layers = build_architecture_layers()
        self.setMinimumSize(600, 600)
        self.setStyleSheet("background-color: #0a0e1a;")

    def paintEvent(self, event) -> None:
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
'''
# ============================================================================
# app/ui/main_window.py — главное окно с sidebar
# ============================================================================
FILES["NeuroLab/app/ui/main_window.py"] = r'''# -*- coding: utf-8 -*-
"""Главное окно приложения."""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import torch
from PySide6.QtCore import Qt, QSize
from PySide6.QtGui import QFont, QIcon, QPixmap, QPainter, QColor
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, QLabel,
    QStackedWidget, QFrame, QPushButton, QApplication, QMessageBox,
)

from app.ml.cnn import build_model, count_parameters
from app.ml.dataset import FASHION_MNIST_CLASSES
from app.visualization.charts import apply_dark_theme

from app.ui.dashboard import DashboardPage
from app.ui.dataset_page import DatasetPage
from app.ui.architecture_page import ArchitecturePage
from app.ui.training_page import TrainingPage
from app.ui.feature_maps_page import FeatureMapsPage
from app.ui.prediction_page import PredictionPage
from app.ui.confusion_matrix_page import ConfusionMatrixPage
from app.ui.whatif_page import WhatIfPage
from app.ui.settings_page import SettingsPage


ASSETS_DIR = Path(__file__).resolve().parent.parent.parent / "assets"
MODELS_DIR = ASSETS_DIR / "models"


def _make_icon(color: str, letter: str, size: int = 28) -> QIcon:
    pix = QPixmap(size, size)
    pix.fill(Qt.GlobalColor.transparent)
    p = QPainter(pix)
    p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    p.setBrush(QColor(color))
    p.setPen(Qt.PenStyle.NoPen)
    p.drawRoundedRect(2, 2, size - 4, size - 4, 6, 6)
    p.setPen(QColor("#0a0e1a"))
    f = QFont("Segoe UI", int(size * 0.5), QFont.Weight.Bold)
    p.setFont(f)
    p.drawText(pix.rect(), Qt.AlignmentFlag.AlignCenter, letter)
    p.end()
    return QIcon(pix)


class SidebarButton(QPushButton):
    def __init__(self, text: str, color: str, letter: str, parent=None) -> None:
        super().__init__(parent)
        self.setText("   " + text)
        self.setIcon(_make_icon(color, letter))
        self.setIconSize(QSize(28, 28))
        self.setCheckable(True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFixedHeight(46)
        self.setFont(QFont("Segoe UI", 11))
        self.setStyleSheet("""
            QPushButton {
                background-color: transparent;
                color: #cdd6f4;
                border: none;
                border-radius: 10px;
                text-align: left;
                padding-left: 10px;
            }
            QPushButton:hover {
                background-color: rgba(0, 212, 255, 0.08);
                color: #00d4ff;
            }
            QPushButton:checked {
                background-color: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 rgba(0,212,255,0.25), stop:1 rgba(168,85,247,0.15));
                color: #00d4ff;
                border: 1px solid rgba(0, 212, 255, 0.35);
            }
        """)


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        apply_dark_theme()

        self.setWindowTitle("NeuroLab — AI Laboratory")
        self.resize(1400, 900)
        self.setMinimumSize(1100, 720)

        self.setStyleSheet("""
            QMainWindow { background-color: #0a0e1a; }
            QWidget { color: #cdd6f4; font-family: "Segoe UI", "Inter", sans-serif; }
            QScrollBar:vertical {
                background: #0a0e1a; width: 10px; margin: 0;
            }
            QScrollBar::handle:vertical {
                background: #1e293b; border-radius: 5px; min-height: 20px;
            }
            QScrollBar::handle:vertical:hover { background: #00d4ff; }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
        """)

        self.device = torch.device("cpu")
        self.model = build_model(num_classes=len(FASHION_MNIST_CLASSES))
        self.num_params = count_parameters(self.model)
        MODELS_DIR.mkdir(parents=True, exist_ok=True)

        central = QWidget()
        self.setCentralWidget(central)
        root_layout = QHBoxLayout(central)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        sidebar = QFrame()
        sidebar.setFixedWidth(260)
        sidebar.setStyleSheet("""
            QFrame {
                background-color: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #0b1020, stop:1 #0a0e1a);
                border-right: 1px solid rgba(0, 212, 255, 0.15);
            }
        """)
        sidebar_layout = QVBoxLayout(sidebar)
        sidebar_layout.setContentsMargins(16, 20, 16, 20)
        sidebar_layout.setSpacing(8)

        logo = QLabel("◉ NeuroLab")
        logo.setFont(QFont("Segoe UI", 20, QFont.Weight.Bold))
        logo.setStyleSheet("color: #00d4ff; padding: 10px 0 20px 0;")
        sidebar_layout.addWidget(logo)

        subtitle = QLabel("AI Laboratory · Fashion-MNIST")
        subtitle.setFont(QFont("Segoe UI", 9))
        subtitle.setStyleSheet("color: #64748b; padding-bottom: 20px;")
        sidebar_layout.addWidget(subtitle)

        self.stack = QStackedWidget()
        self.buttons: Dict[str, SidebarButton] = {}

        nav_items = [
            ("Dashboard", "#00d4ff", "D"),
            ("Dataset", "#22d3ee", "◈"),
            ("Architecture", "#a855f7", "▣"),
            ("Training", "#f472b6", "▶"),
            ("Feature Maps", "#f59e0b", "≡"),
            ("Prediction", "#10b981", "✦"),
            ("Confusion Matrix", "#ef4444", "▦"),
            ("What If?", "#8b5cf6", "?"),
            ("Settings", "#64748b", "⚙"),
        ]

        for text, color, letter in nav_items:
            btn = SidebarButton(text, color, letter)
            btn.clicked.connect(lambda _=False, t=text: self._switch(t))
            sidebar_layout.addWidget(btn)
            self.buttons[text] = btn

        sidebar_layout.addStretch()

        info_frame = QFrame()
        info_frame.setStyleSheet("""
            QFrame {
                background-color: rgba(0, 212, 255, 0.05);
                border: 1px solid rgba(0, 212, 255, 0.2);
                border-radius: 10px;
                padding: 10px;
            }
        """)
        info_layout = QVBoxLayout(info_frame)
        info_layout.setContentsMargins(12, 12, 12, 12)
        info_layout.setSpacing(4)

        lbl_dev = QLabel(f"Device: {self.device}")
        lbl_dev.setStyleSheet("color: #94a3b8; font-size: 10pt;")
        info_layout.addWidget(lbl_dev)

        lbl_params = QLabel(f"Params: {self.num_params:,}")
        lbl_params.setStyleSheet("color: #94a3b8; font-size: 10pt;")
        info_layout.addWidget(lbl_params)

        sidebar_layout.addWidget(info_frame)

        root_layout.addWidget(sidebar)

        content = QWidget()
        content.setStyleSheet("background-color: #0a0e1a;")
        content_layout = QVBoxLayout(content)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.addWidget(self.stack)
        root_layout.addWidget(content, 1)

        self.pages = {
            "Dashboard": DashboardPage(self.model, self.device, self.num_params),
            "Dataset": DatasetPage(),
            "Architecture": ArchitecturePage(),
            "Training": TrainingPage(self.model, self.device),
            "Feature Maps": FeatureMapsPage(self.model, self.device),
            "Prediction": PredictionPage(self.model, self.device),
            "Confusion Matrix": ConfusionMatrixPage(self.model, self.device),
            "What If?": WhatIfPage(self.model, self.device),
            "Settings": SettingsPage(),
        }
        for page in self.pages.values():
            self.stack.addWidget(page)

        self._switch("Dashboard")

    def _switch(self, name: str) -> None:
        for key, btn in self.buttons.items():
            btn.setChecked(key == name)
        if name in self.pages:
            self.stack.setCurrentWidget(self.pages[name])
            page = self.pages[name]
            if hasattr(page, "on_activated"):
                try:
                    page.on_activated()
                except Exception as exc:
                    QMessageBox.warning(self, "Page error", str(exc))

    def closeEvent(self, event) -> None:
        training_page = self.pages.get("Training")
        if training_page is not None and hasattr(training_page, "request_stop"):
            try:
                training_page.request_stop()
            except Exception:
                pass
        super().closeEvent(event)
'''

# ============================================================================
# app/ui/dashboard.py
# ============================================================================
FILES["NeuroLab/app/ui/dashboard.py"] = r'''# -*- coding: utf-8 -*-
"""Dashboard — обзорная страница."""

from __future__ import annotations

import torch
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QFrame, QGridLayout,
)

from app.ml.cnn import SmallCNN
from app.ml.hardware import get_cpu_name, get_logical_threads, get_ram_gb, get_gpu_info, get_cuda_available


class StatCard(QFrame):
    def __init__(self, title: str, value: str, accent: str, parent=None) -> None:
        super().__init__(parent)
        self.setStyleSheet(f"""
            QFrame {{
                background-color: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 rgba(15,23,42,0.9), stop:1 rgba(10,14,26,0.9));
                border: 1px solid {accent}40;
                border-radius: 14px;
            }}
            QFrame:hover {{
                border: 1px solid {accent}aa;
            }}
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

    def set_value(self, value: str) -> None:
        self.value_lbl.setText(value)


class DashboardPage(QWidget):
    def __init__(self, model: SmallCNN, device: torch.device, num_params: int, parent=None) -> None:
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

        cards = [
            self.card_model, self.card_params, self.card_device,
            self.card_classes, self.card_layers, self.card_status,
        ]
        for i, card in enumerate(cards):
            grid.addWidget(card, i // 3, i % 3)

        outer.addStretch()

        info = QFrame()
        info.setStyleSheet("""
            QFrame {
                background-color: rgba(0, 212, 255, 0.05);
                border: 1px solid rgba(0, 212, 255, 0.25);
                border-radius: 14px;
                padding: 16px;
            }
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

    def on_activated(self) -> None:
        self.card_device.set_value(str(self.device))
        self.card_params.set_value(f"{self.num_params:,}")
'''

# ============================================================================
# app/ui/dataset_page.py
# ============================================================================
FILES["NeuroLab/app/ui/dataset_page.py"] = r'''# -*- coding: utf-8 -*-
"""Dataset viewer — просмотр Fashion-MNIST."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image
from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap, QImage, QFont
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QScrollArea, QFrame, QGridLayout, QMessageBox, QSpinBox, QApplication,
)

from app.ml.dataset import FASHION_MNIST_CLASSES, prepare_fashion_mnist


ASSETS_DIR = Path(__file__).resolve().parent.parent.parent / "assets"
DATA_DIR = ASSETS_DIR / "data"


def _numpy_to_qpixmap_gray(arr: np.ndarray) -> QPixmap:
    h, w = arr.shape
    qimg = QImage(arr.data, w, h, w, QImage.Format.Format_Grayscale8).copy()
    return QPixmap.fromImage(qimg)


class DatasetPage(QWidget):
    def __init__(self, parent=None) -> None:
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
    def _btn_style(color: str) -> str:
        return f"""
            QPushButton {{
                background-color: {color}20;
                color: {color};
                border: 1px solid {color}80;
                border-radius: 8px;
                padding: 8px 16px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background-color: {color}40;
            }}
        """

    def _load_dataset(self) -> None:
        try:
            self.status.setText("Загрузка Fashion-MNIST (это может занять некоторое время при первом запуске)...")
            QApplication.processEvents()
            _, _, _, train_full = prepare_fashion_mnist(
                root=str(DATA_DIR), batch_size=64, val_fraction=0.1,
                download=True, num_workers=0,
            )
            self.dataset = train_full
            self.current_page = 0
            self._render_page()
            self.status.setText(f"Загружено: {len(self.dataset)} изображений. Классы: {', '.join(FASHION_MNIST_CLASSES)}.")
        except Exception as exc:
            QMessageBox.critical(self, "Ошибка загрузки", str(exc))
            self.status.setText(f"Ошибка: {exc}")

    def _on_size_changed(self, value: int) -> None:
        self.page_size = value
        self.current_page = 0
        if self.dataset is not None:
            self._render_page()

    def _prev_page(self) -> None:
        if self.dataset is None:
            return
        if self.current_page > 0:
            self.current_page -= 1
            self._render_page()

    def _next_page(self) -> None:
        if self.dataset is None:
            return
        max_page = (len(self.dataset) - 1) // self.page_size
        if self.current_page < max_page:
            self.current_page += 1
            self._render_page()

    def _render_page(self) -> None:
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
                QFrame {
                    background-color: #111827;
                    border: 1px solid #1f2937;
                    border-radius: 8px;
                }
                QFrame:hover { border: 1px solid #00d4ff; }
            """)
            frame.setFixedSize(110, 140)

            layout = QVBoxLayout(frame)
            layout.setContentsMargins(6, 6, 6, 6)
            layout.setSpacing(4)

            lbl_img = QLabel()
            lbl_img.setFixedSize(96, 96)
            lbl_img.setAlignment(Qt.AlignmentFlag.AlignCenter)
            pix = _numpy_to_qpixmap_gray(img_arr).scaled(
                96, 96, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation,
            )
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

    def on_activated(self) -> None:
        pass
'''

# ============================================================================
# app/ui/architecture_page.py
# ============================================================================
FILES["NeuroLab/app/ui/architecture_page.py"] = r'''# -*- coding: utf-8 -*-
"""Страница визуализации архитектуры сети."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QFont
from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel, QSplitter, QFrame, QHBoxLayout

from app.visualization.network_view import NetworkView


class ArchitecturePage(QWidget):
    def __init__(self, parent=None) -> None:
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
            QFrame {
                background-color: rgba(15, 23, 42, 0.6);
                border: 1px solid rgba(0, 212, 255, 0.2);
                border-radius: 12px;
            }
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

    def on_activated(self) -> None:
        self.network_view.update()
'''

# ============================================================================
# app/ui/training_page.py
# ============================================================================
FILES["NeuroLab/app/ui/training_page.py"] = r'''# -*- coding: utf-8 -*-
"""Страница обучения модели."""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PySide6.QtCore import Qt, QObject, Signal, QThread
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QFrame,
    QSpinBox, QDoubleSpinBox, QProgressBar, QMessageBox, QGridLayout,
    QApplication, QComboBox,
)

from app.ml.cnn import SmallCNN
from app.ml.dataset import prepare_fashion_mnist, FASHION_MNIST_CLASSES
from app.ml.trainer import Trainer, TrainingStats
from app.visualization.charts import LossChart, AccuracyChart, ConfusionMatrixWidget

ASSETS_DIR = Path(__file__).resolve().parent.parent.parent / "assets"
DATA_DIR = ASSETS_DIR / "data"
MODELS_DIR = ASSETS_DIR / "models"


class TrainingSignals(QObject):
    step = Signal(object, int, int)
    epoch_done = Signal(object)
    finished = Signal(str)
    error = Signal(str)


class TrainingWorker(QThread):
    def __init__(
        self,
        trainer: Trainer,
        epochs: int,
        signals: TrainingSignals,
    ) -> None:
        super().__init__()
        self.trainer = trainer
        self.epochs = epochs
        self.signals = signals
        self._stop_flag = threading.Event()
        self._pause_flag = threading.Event()

    def request_stop(self) -> None:
        self._stop_flag.set()
        self._pause_flag.clear()

    def toggle_pause(self) -> None:
        if self._pause_flag.is_set():
            self._pause_flag.clear()
        else:
            self._pause_flag.set()

    def is_paused(self) -> bool:
        return self._pause_flag.is_set()

    def is_stopped(self) -> bool:
        return self._stop_flag.is_set()

    def run(self) -> None:
        try:
            for epoch in range(1, self.epochs + 1):
                if self._stop_flag.is_set():
                    break
                self.trainer.stats.epoch = epoch
                self.trainer.train_epoch(
                    step_callback=lambda s, b, n: self.signals.step.emit(s, b, n),
                    should_stop=lambda: self._stop_flag.is_set(),
                    should_pause=lambda: self._pause_flag.is_set(),
                )
                if self._stop_flag.is_set():
                    break
                self.trainer.validate()
                self.trainer.step_scheduler()
                self.trainer.stats.history["train_loss"].append(self.trainer.stats.train_loss)
                self.trainer.stats.history["train_acc"].append(self.trainer.stats.train_acc)
                self.trainer.stats.history["val_loss"].append(self.trainer.stats.val_loss)
                self.trainer.stats.history["val_acc"].append(self.trainer.stats.val_acc)
                self.signals.epoch_done.emit(self.trainer.stats)

            save_path = MODELS_DIR / "neurolab_cnn.pth"
            self.trainer.save_checkpoint(save_path)
            self.signals.finished.emit(f"Обучение завершено. Модель сохранена: {save_path}")
        except Exception as exc:
            self.signals.error.emit(str(exc))


class TrainingPage(QWidget):
    def __init__(self, model: SmallCNN, device: torch.device, parent=None) -> None:
        super().__init__(parent)
        self.model = model
        self.device = device
        self.trainer: Optional[Trainer] = None
        self.worker: Optional[TrainingWorker] = None
        self.thread: Optional[QThread] = None
        self.signals = TrainingSignals()
        self.signals.step.connect(self._on_step)
        self.signals.epoch_done.connect(self._on_epoch_done)
        self.signals.finished.connect(self._on_finished)
        self.signals.error.connect(self._on_error)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(30, 30, 30, 30)

        header = QLabel("Training")
        header.setFont(QFont("Segoe UI", 28, QFont.Weight.Bold))
        header.setStyleSheet("color: #00d4ff;")
        outer.addWidget(header)

        sub = QLabel("Настоящее обучение CNN на Fashion-MNIST с визуализацией метрик")
        sub.setStyleSheet("color: #94a3b8; font-size: 11pt; margin-bottom: 16px;")
        outer.addWidget(sub)

        ctrl_frame = QFrame()
        ctrl_frame.setStyleSheet("""
            QFrame {
                background-color: rgba(15, 23, 42, 0.6);
                border: 1px solid rgba(0, 212, 255, 0.2);
                border-radius: 12px;
            }
        """)
        ctrl_layout = QGridLayout(ctrl_frame)
        ctrl_layout.setContentsMargins(20, 20, 20, 20)
        ctrl_layout.setHorizontalSpacing(20)
        ctrl_layout.setVerticalSpacing(12)

        self.combo_mode = QComboBox()
        self.combo_mode.addItems(["Quick Demo", "Normal", "Full"])
        self.combo_mode.setCurrentIndex(0)
        self.combo_mode.setStyleSheet(self._combo_style())
        self.combo_mode.currentIndexChanged.connect(self._on_mode_changed)

        self.spin_epochs = QSpinBox()
        self.spin_epochs.setRange(1, 200)
        self.spin_epochs.setValue(3)
        self.spin_epochs.setStyleSheet(self._spin_style())

        self.spin_batch = QSpinBox()
        self.spin_batch.setRange(16, 512)
        self.spin_batch.setSingleStep(16)
        self.spin_batch.setValue(64)
        self.spin_batch.setStyleSheet(self._spin_style())

        self.spin_lr = QDoubleSpinBox()
        self.spin_lr.setDecimals(5)
        self.spin_lr.setRange(1e-6, 1e-1)
        self.spin_lr.setValue(1e-3)
        self.spin_lr.setSingleStep(1e-4)
        self.spin_lr.setStyleSheet(self._spin_style())

        ctrl_layout.addWidget(self._labeled("Mode", self.combo_mode), 0, 0)
        ctrl_layout.addWidget(self._labeled("Epochs", self.spin_epochs), 0, 1)
        ctrl_layout.addWidget(self._labeled("Batch size", self.spin_batch), 0, 2)
        ctrl_layout.addWidget(self._labeled("Learning rate", self.spin_lr), 0, 3)

        btn_row = QHBoxLayout()
        self.btn_start = QPushButton("▶  Start Training")
        self.btn_start.setStyleSheet(self._btn_style("#10b981"))
        self.btn_start.clicked.connect(self._start_training)
        btn_row.addWidget(self.btn_start)

        self.btn_pause = QPushButton("⏸  Pause")
        self.btn_pause.setStyleSheet(self._btn_style("#f59e0b"))
        self.btn_pause.setEnabled(False)
        self.btn_pause.clicked.connect(self._toggle_pause)
        btn_row.addWidget(self.btn_pause)

        self.btn_stop = QPushButton("■  Stop")
        self.btn_stop.setStyleSheet(self._btn_style("#ef4444"))
        self.btn_stop.setEnabled(False)
        self.btn_stop.clicked.connect(self._stop_training)
        btn_row.addWidget(self.btn_stop)

        self.btn_reset = QPushButton("↻  Reset Model")
        self.btn_reset.setStyleSheet(self._btn_style("#a855f7"))
        self.btn_reset.clicked.connect(self._reset_model)
        btn_row.addWidget(self.btn_reset)

        ctrl_layout.addLayout(btn_row, 1, 0, 1, 4)
        outer.addWidget(ctrl_frame)

        self.progress = QProgressBar()
        self.progress.setRange(0, 1000)
        self.progress.setValue(0)
        self.progress.setTextVisible(True)
        self.progress.setStyleSheet("""
            QProgressBar {
                background: #111827;
                border: 1px solid #1f2937;
                border-radius: 8px;
                text-align: center;
                color: #cdd6f4;
                height: 24px;
            }
            QProgressBar::chunk {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #00d4ff, stop:1 #a855f7);
                border-radius: 7px;
            }
        """)
        outer.addWidget(self.progress)

        metrics = QHBoxLayout()
        self.lbl_epoch = self._metric("Epoch", "—")
        self.lbl_train_loss = self._metric("Train Loss", "—")
        self.lbl_train_acc = self._metric("Train Acc", "—")
        self.lbl_val_loss = self._metric("Val Loss", "—")
        self.lbl_val_acc = self._metric("Val Acc", "—")
        self.lbl_time = self._metric("Time/Epoch", "—")
        for w in (self.lbl_epoch, self.lbl_train_loss, self.lbl_train_acc,
                  self.lbl_val_loss, self.lbl_val_acc, self.lbl_time):
            metrics.addWidget(w)
        outer.addLayout(metrics)

        charts_row = QHBoxLayout()
        self.loss_chart = LossChart()
        self.loss_chart.setMinimumHeight(240)
        self.acc_chart = AccuracyChart()
        self.acc_chart.setMinimumHeight(240)
        charts_row.addWidget(self.loss_chart, 1)
        charts_row.addWidget(self.acc_chart, 1)
        outer.addLayout(charts_row, 1)

        cm_label = QLabel("Confusion Matrix (validation, last epoch)")
        cm_label.setStyleSheet("color: #a855f7; font-size: 12pt; font-weight: bold; margin-top: 8px;")
        outer.addWidget(cm_label)
        self.cm_widget = ConfusionMatrixWidget()
        self.cm_widget.setMinimumHeight(320)
        outer.addWidget(self.cm_widget, 1)

        self.status = QLabel("Готов к обучению.")
        self.status.setStyleSheet("color: #94a3b8; padding-top: 6px;")
        outer.addWidget(self.status)

        self._on_mode_changed(0)

    @staticmethod
    def _spin_style() -> str:
        return """
            QSpinBox, QDoubleSpinBox {
                background: #111827;
                color: #cdd6f4;
                border: 1px solid #1f2937;
                border-radius: 6px;
                padding: 4px 8px;
                min-width: 100px;
            }
            QSpinBox:hover, QDoubleSpinBox:hover { border: 1px solid #00d4ff; }
        """

    @staticmethod
    def _combo_style() -> str:
        return """
            QComboBox {
                background: #111827;
                color: #cdd6f4;
                border: 1px solid #1f2937;
                border-radius: 6px;
                padding: 4px 8px;
                min-width: 120px;
            }
            QComboBox:hover { border: 1px solid #00d4ff; }
            QComboBox::drop-down { border: none; }
        """

    @staticmethod
    def _btn_style(color: str) -> str:
        return f"""
            QPushButton {{
                background-color: {color}20;
                color: {color};
                border: 1px solid {color}80;
                border-radius: 8px;
                padding: 8px 16px;
                font-weight: bold;
            }}
            QPushButton:hover {{ background-color: {color}40; }}
            QPushButton:disabled {{ color: #475569; border: 1px solid #334155; background: transparent; }}
        """

    @staticmethod
    def _labeled(text: str, widget: QWidget) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(4)
        lbl = QLabel(text)
        lbl.setStyleSheet("color: #94a3b8; font-size: 10pt;")
        lay.addWidget(lbl)
        lay.addWidget(widget)
        return w

    @staticmethod
    def _metric(title: str, value: str) -> QFrame:
        f = QFrame()
        f.setStyleSheet("""
            QFrame {
                background: rgba(0, 212, 255, 0.05);
                border: 1px solid rgba(0, 212, 255, 0.2);
                border-radius: 10px;
            }
        """)
        lay = QVBoxLayout(f)
        lay.setContentsMargins(12, 8, 12, 8)
        t = QLabel(title.upper())
        t.setStyleSheet("color: #00d4ff; font-size: 9pt; letter-spacing: 1px;")
        lay.addWidget(t)
        v = QLabel(value)
        v.setFont(QFont("Segoe UI", 14, QFont.Weight.Bold))
        v.setStyleSheet("color: #e2e8f0;")
        v.setObjectName("value")
        lay.addWidget(v)
        f._value_label = v
        return f

    def _set_metric(self, frame: QFrame, value: str) -> None:
        frame._value_label.setText(value)

    def _on_mode_changed(self, index: int) -> None:
        if index == 0:  # Quick Demo
            self.spin_epochs.setValue(3)
            self.spin_batch.setValue(64)
        elif index == 1:  # Normal
            self.spin_epochs.setValue(5)
            self.spin_batch.setValue(128)
        elif index == 2:  # Full
            self.spin_epochs.setValue(10)
            self.spin_batch.setValue(128)

    def _start_training(self) -> None:
        if self.worker is not None and self.worker.isRunning():
            return

        mode = self.combo_mode.currentIndex()
        train_subset = None
        test_subset = None

        if mode == 0:  # Quick Demo
            train_subset = 5000
            test_subset = 1000
        elif mode == 1:  # Normal
            train_subset = 20000
            test_subset = 5000
        # Full: None (весь датасет)

        try:
            self.status.setText("Подготовка данных...")
            QApplication.processEvents()
            train_loader, val_loader, _, _ = prepare_fashion_mnist(
                root=str(DATA_DIR),
                batch_size=self.spin_batch.value(),
                val_fraction=0.1,
                download=True,
                num_workers=0,
                train_subset_size=train_subset,
                test_subset_size=test_subset,
            )
        except Exception as exc:
            QMessageBox.critical(self, "Ошибка", f"Не удалось загрузить данные: {exc}")
            self.status.setText(f"Ошибка: {exc}")
            return

        self.trainer = Trainer(
            model=self.model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=self.device,
            learning_rate=self.spin_lr.value(),
        )

        self.worker = TrainingWorker(self.trainer, self.spin_epochs.value(), self.signals)
        self.thread = self.worker
        self.worker.start()

        self.btn_start.setEnabled(False)
        self.btn_pause.setEnabled(True)
        self.btn_stop.setEnabled(True)
        self.combo_mode.setEnabled(False)
        self.spin_epochs.setEnabled(False)
        self.spin_batch.setEnabled(False)
        self.spin_lr.setEnabled(False)
        self.status.setText("Обучение запущено...")

    def _toggle_pause(self) -> None:
        if self.worker is None:
            return
        self.worker.toggle_pause()
        if self.worker.is_paused():
            self.btn_pause.setText("▶  Resume")
            self.status.setText("Обучение приостановлено.")
        else:
            self.btn_pause.setText("⏸  Pause")
            self.status.setText("Обучение возобновлено.")

    def request_stop(self) -> None:
        if self.worker is not None and self.worker.isRunning():
            self.worker.request_stop()
            self.worker.wait(5000)

    def _stop_training(self) -> None:
        if self.worker is None:
            return
        self.worker.request_stop()
        self.status.setText("Остановка...")

    def _reset_model(self) -> None:
        from app.ml.cnn import build_model
        self.model = build_model(num_classes=len(FASHION_MNIST_CLASSES))
        main = self.window()
        if hasattr(main, "model"):
            main.model = self.model
            for name in ("Feature Maps", "Prediction", "Confusion Matrix", "What If?"):
                p = main.pages.get(name)
                if p is not None and hasattr(p, "set_model"):
                    p.set_model(self.model)
        QMessageBox.information(self, "Reset", "Модель сброшена к начальным весам.")

    def _on_step(self, stats: TrainingStats, batch_idx: int, n_batches: int) -> None:
        frac = batch_idx / max(1, n_batches)
        epoch_frac = (stats.epoch - 1 + frac) / max(1, self.spin_epochs.value())
        self.progress.setValue(int(epoch_frac * 1000))
        self._set_metric(self.lbl_epoch, f"{stats.epoch} / {self.spin_epochs.value()}")
        self._set_metric(self.lbl_train_loss, f"{stats.train_loss:.4f}")
        self._set_metric(self.lbl_train_acc, f"{stats.train_acc:.2f}%")

    def _on_epoch_done(self, stats: TrainingStats) -> None:
        self._set_metric(self.lbl_epoch, f"{stats.epoch} / {self.spin_epochs.value()}")
        self._set_metric(self.lbl_train_loss, f"{stats.train_loss:.4f}")
        self._set_metric(self.lbl_train_acc, f"{stats.train_acc:.2f}%")
        self._set_metric(self.lbl_val_loss, f"{stats.val_loss:.4f}")
        self._set_metric(self.lbl_val_acc, f"{stats.val_acc:.2f}%")
        self._set_metric(self.lbl_time, f"{stats.elapsed:.1f}s")

        h = stats.history
        self.loss_chart.update_data(h["train_loss"], h["val_loss"])
        self.acc_chart.update_data(h["train_acc"], h["val_acc"])

        self._compute_confusion_matrix()

        frac = stats.epoch / max(1, self.spin_epochs.value())
        self.progress.setValue(int(frac * 1000))
        self.status.setText(f"Эпоха {stats.epoch} завершена. Val Acc: {stats.val_acc:.2f}%")

    def _compute_confusion_matrix(self) -> None:
        if self.trainer is None:
            return
        n_classes = len(FASHION_MNIST_CLASSES)
        cm = np.zeros((n_classes, n_classes), dtype=np.int64)
        self.trainer.model.eval()
        with torch.no_grad():
            for inputs, targets in self.trainer.val_loader:
                inputs = inputs.to(self.device, non_blocking=True)
                outputs = self.trainer.model(inputs)
                preds = outputs.argmax(dim=1).cpu().numpy()
                tg = targets.numpy()
                for p, t in zip(preds, tg):
                    cm[int(t), int(p)] += 1
        self.cm_widget.set_matrix(cm, list(FASHION_MNIST_CLASSES))

    def _on_finished(self, message: str) -> None:
        self.status.setText(message)
        self.progress.setValue(1000)
        self._cleanup_worker()

    def _on_error(self, message: str) -> None:
        QMessageBox.critical(self, "Ошибка обучения", message)
        self.status.setText(f"Ошибка: {message}")
        self._cleanup_worker()

    def _cleanup_worker(self) -> None:
        self.btn_start.setEnabled(True)
        self.btn_pause.setEnabled(False)
        self.btn_stop.setEnabled(False)
        self.combo_mode.setEnabled(True)
        self.spin_epochs.setEnabled(True)
        self.spin_batch.setEnabled(True)
        self.spin_lr.setEnabled(True)
        self.btn_pause.setText("⏸  Pause")
        if self.worker is not None:
            self.worker.wait(2000)
        self.worker = None
        self.thread = None
'''

# ============================================================================
# app/ui/feature_maps_page.py
# ============================================================================
FILES["NeuroLab/app/ui/feature_maps_page.py"] = r'''# -*- coding: utf-8 -*-
"""Страница визуализации feature maps."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont, QPixmap, QImage
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QFileDialog, QFrame, QScrollArea, QMessageBox,
)

from app.ml.cnn import SmallCNN
from app.ml.dataset import FASHION_MNIST_CLASSES, prepare_fashion_mnist
from app.visualization.feature_maps import pil_to_tensor_for_model


ASSETS_DIR = Path(__file__).resolve().parent.parent.parent / "assets"
DATA_DIR = ASSETS_DIR / "data"


def _numpy_to_qpixmap_gray(arr: np.ndarray) -> QPixmap:
    h, w = arr.shape
    qimg = QImage(arr.data, w, h, w, QImage.Format.Format_Grayscale8).copy()
    return QPixmap.fromImage(qimg)


def _numpy_to_qpixmap_rgb(arr: np.ndarray) -> QPixmap:
    h, w, ch = arr.shape
    if ch == 3:
        qimg = QImage(arr.data, w, h, 3 * w, QImage.Format.Format_RGB888).copy()
    else:
        qimg = QImage(arr.data, w, h, w, QImage.Format.Format_Grayscale8).copy()
    return QPixmap.fromImage(qimg)


class FeatureMapsPage(QWidget):
    def __init__(self, model: SmallCNN, device: torch.device, parent=None) -> None:
        super().__init__(parent)
        self.model = model
        self.device = device
        self.dataset = None

        outer = QVBoxLayout(self)
        outer.setContentsMargins(30, 30, 30, 30)

        header = QLabel("Feature Maps")
        header.setFont(QFont("Segoe UI", 28, QFont.Weight.Bold))
        header.setStyleSheet("color: #00d4ff;")
        outer.addWidget(header)

        sub = QLabel("Активации промежуточных свёрточных слоёв")
        sub.setStyleSheet("color: #94a3b8; font-size: 11pt; margin-bottom: 16px;")
        outer.addWidget(sub)

        ctrl = QHBoxLayout()
        self.btn_load_img = QPushButton("📁  Load Image")
        self.btn_load_img.setStyleSheet(self._btn_style("#00d4ff"))
        self.btn_load_img.clicked.connect(self._load_image)
        ctrl.addWidget(self.btn_load_img)

        self.btn_sample = QPushButton("🎲  Random Sample from Fashion-MNIST")
        self.btn_sample.setStyleSheet(self._btn_style("#a855f7"))
        self.btn_sample.clicked.connect(self._random_sample)
        ctrl.addWidget(self.btn_sample)

        ctrl.addStretch()
        outer.addLayout(ctrl)

        self.input_lbl = QLabel("Нет изображения")
        self.input_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.input_lbl.setFixedSize(224, 224)
        self.input_lbl.setStyleSheet("""
            QLabel {
                background: #111827;
                border: 1px solid #1f2937;
                border-radius: 12px;
                color: #64748b;
            }
        """)
        outer.addWidget(self.input_lbl, 0, Qt.AlignmentFlag.AlignHCenter)

        self.pred_lbl = QLabel("")
        self.pred_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.pred_lbl.setStyleSheet("color: #00d4ff; font-size: 12pt; font-weight: bold;")
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
    def _btn_style(color: str) -> str:
        return f"""
            QPushButton {{
                background-color: {color}20;
                color: {color};
                border: 1px solid {color}80;
                border-radius: 8px;
                padding: 8px 16px;
                font-weight: bold;
            }}
            QPushButton:hover {{ background-color: {color}40; }}
        """

    def set_model(self, model: SmallCNN) -> None:
        self.model = model

    def _load_image(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Выберите изображение",
            str(Path.home()),
            "Images (*.png *.jpg *.jpeg *.bmp *.webp)",
        )
        if not path:
            return
        try:
            pil = Image.open(path).convert("L")
            self._process_pil(pil)
        except Exception as exc:
            QMessageBox.critical(self, "Ошибка", str(exc))

    def _random_sample(self) -> None:
        if self.dataset is None:
            try:
                _, _, _, train_full = prepare_fashion_mnist(
                    root=str(DATA_DIR), batch_size=64, val_fraction=0.1,
                    download=True, num_workers=0,
                )
                self.dataset = train_full
            except Exception as exc:
                QMessageBox.critical(self, "Ошибка", str(exc))
                return
        idx = int(np.random.randint(0, len(self.dataset)))
        img_tensor, label = self.dataset[idx]
        img_arr = (img_tensor.squeeze().numpy() * 127.5 + 127.5).astype(np.uint8)
        pil = Image.fromarray(img_arr, mode="L")
        self._process_pil(pil, forced_label=FASHION_MNIST_CLASSES[label])

    def _process_pil(self, pil: Image.Image, forced_label: Optional[str] = None) -> None:
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

    def _render_maps(self, maps: dict) -> None:
        while self.vlayout.count():
            item = self.vlayout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()

        for name, grid in maps.items():
            frame = QFrame()
            frame.setStyleSheet("""
                QFrame {
                    background: rgba(15, 23, 42, 0.6);
                    border: 1px solid rgba(0, 212, 255, 0.2);
                    border-radius: 12px;
                }
            """)
            lay = QVBoxLayout(frame)
            lay.setContentsMargins(16, 12, 16, 12)

            title = QLabel(f"Layer: {name}   (channels grid)")
            title.setStyleSheet("color: #a855f7; font-size: 12pt; font-weight: bold;")
            lay.addWidget(title)

            lbl = QLabel()
            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            pix = _numpy_to_qpixmap_gray(grid)
            pix = pix.scaled(
                640, 240,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            lbl.setPixmap(pix)
            lay.addWidget(lbl, 0, Qt.AlignmentFlag.AlignHCenter)

            info = QLabel(f"Grid shape: {grid.shape[1]}×{grid.shape[0]} px")
            info.setStyleSheet("color: #94a3b8; font-size: 10pt;")
            info.setAlignment(Qt.AlignmentFlag.AlignCenter)
            lay.addWidget(info)

            self.vlayout.addWidget(frame)

    def on_activated(self) -> None:
        pass
'''

# ============================================================================
# app/ui/prediction_page.py
# ============================================================================
FILES["NeuroLab/app/ui/prediction_page.py"] = r'''# -*- coding: utf-8 -*-
"""Страница инференса: загрузка изображения и предсказание."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont, QPixmap, QImage
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QFileDialog, QFrame, QMessageBox, QProgressBar,
)

from app.ml.cnn import SmallCNN
from app.ml.inference import predict_pil
from app.ml.dataset import FASHION_MNIST_CLASSES


def _numpy_to_qpixmap_rgb(arr: np.ndarray) -> QPixmap:
    h, w, ch = arr.shape
    if ch == 3:
        qimg = QImage(arr.data, w, h, 3 * w, QImage.Format.Format_RGB888).copy()
    else:
        qimg = QImage(arr.data, w, h, w, QImage.Format.Format_Grayscale8).copy()
    return QPixmap.fromImage(qimg)


class PredictionPage(QWidget):
    def __init__(self, model: SmallCNN, device: torch.device, parent=None) -> None:
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
            QLabel {
                background: #111827;
                border: 1px solid #1f2937;
                border-radius: 16px;
                color: #64748b;
                font-size: 12pt;
            }
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
            QFrame {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 rgba(0,212,255,0.1), stop:1 rgba(168,85,247,0.1));
                border: 1px solid rgba(0, 212, 255, 0.3);
                border-radius: 16px;
            }
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
    def _btn_style(color: str) -> str:
        return f"""
            QPushButton {{
                background-color: {color}20;
                color: {color};
                border: 1px solid {color}80;
                border-radius: 8px;
                padding: 8px 16px;
                font-weight: bold;
            }}
            QPushButton:hover {{ background-color: {color}40; }}
        """

    def set_model(self, model: SmallCNN) -> None:
        self.model = model

    def _load_image(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Выберите изображение",
            str(Path.home()),
            "Images (*.png *.jpg *.jpeg *.bmp *.webp)",
        )
        if not path:
            return
        try:
            pil = Image.open(path).convert("L")
            self.file_lbl.setText(Path(path).name)
            self._run_inference(pil)
        except Exception as exc:
            QMessageBox.critical(self, "Ошибка", str(exc))

    def _demo_noise(self) -> None:
        arr = np.random.randint(0, 256, (64, 64), dtype=np.uint8)
        pil = Image.fromarray(arr, mode="L")
        self.file_lbl.setText("demo_noise.png")
        self._run_inference(pil)

    def _run_inference(self, pil: Image.Image) -> None:
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

    def _render_bars(self, ranking) -> None:
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
                QProgressBar {{
                    background: #111827;
                    border: 1px solid #1f2937;
                    border-radius: 6px;
                    height: 14px;
                }}
                QProgressBar::chunk {{
                    background: {color};
                    border-radius: 5px;
                }}
            """)
            row_layout.addWidget(bar, 1)

            val = QLabel(f"{prob * 100:.2f}%")
            val.setFixedWidth(70)
            val.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            val.setStyleSheet(f"color: {color}; font-weight: bold;")
            row_layout.addWidget(val)

            self.bars_layout.addWidget(row)

    def on_activated(self) -> None:
        pass
'''

# ============================================================================
# app/ui/confusion_matrix_page.py
# ============================================================================
FILES["NeuroLab/app/ui/confusion_matrix_page.py"] = r'''# -*- coding: utf-8 -*-
"""Страница confusion matrix."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QPushButton, QMessageBox, QApplication,
)

from app.ml.cnn import SmallCNN
from app.ml.dataset import FASHION_MNIST_CLASSES, prepare_fashion_mnist
from app.visualization.charts import ConfusionMatrixWidget

ASSETS_DIR = Path(__file__).resolve().parent.parent.parent / "assets"
DATA_DIR = ASSETS_DIR / "data"


class ConfusionMatrixPage(QWidget):
    def __init__(self, model: SmallCNN, device: torch.device, parent=None) -> None:
        super().__init__(parent)
        self.model = model
        self.device = device

        outer = QVBoxLayout(self)
        outer.setContentsMargins(30, 30, 30, 30)

        header = QLabel("Confusion Matrix")
        header.setFont(QFont("Segoe UI", 28, QFont.Weight.Bold))
        header.setStyleSheet("color: #00d4ff;")
        outer.addWidget(header)

        sub = QLabel("Матрица ошибок на валидационной выборке")
        sub.setStyleSheet("color: #94a3b8; font-size: 11pt; margin-bottom: 16px;")
        outer.addWidget(sub)

        self.btn_compute = QPushButton("🔄  Compute Confusion Matrix")
        self.btn_compute.setStyleSheet(self._btn_style("#a855f7"))
        self.btn_compute.clicked.connect(self._compute_matrix)
        outer.addWidget(self.btn_compute, 0, Qt.AlignmentFlag.AlignLeft)

        self.cm_widget = ConfusionMatrixWidget()
        self.cm_widget.setMinimumHeight(500)
        outer.addWidget(self.cm_widget, 1)

        self.status = QLabel("Нажмите кнопку для вычисления матрицы ошибок.")
        self.status.setStyleSheet("color: #94a3b8; padding-top: 8px;")
        outer.addWidget(self.status)

    @staticmethod
    def _btn_style(color: str) -> str:
        return f"""
            QPushButton {{
                background-color: {color}20;
                color: {color};
                border: 1px solid {color}80;
                border-radius: 8px;
                padding: 8px 16px;
                font-weight: bold;
            }}
            QPushButton:hover {{ background-color: {color}40; }}
        """

    def set_model(self, model: SmallCNN) -> None:
        self.model = model

    def _compute_matrix(self) -> None:
        try:
            self.status.setText("Загрузка данных...")
            QApplication.processEvents()
            _, val_loader, _, _ = prepare_fashion_mnist(
                root=str(DATA_DIR), batch_size=64, val_fraction=0.1,
                download=True, num_workers=0, test_subset_size=2000,
            )
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

    def on_activated(self) -> None:
        pass
'''

# ============================================================================
# app/ui/whatif_page.py
# ============================================================================
FILES["NeuroLab/app/ui/whatif_page.py"] = r'''# -*- coding: utf-8 -*-
"""Страница What-If analysis."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageFilter, ImageEnhance, ImageOps
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont, QPixmap, QImage
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QFileDialog, QFrame, QMessageBox, QScrollArea,
)

from app.ml.cnn import SmallCNN
from app.ml.inference import predict_pil
from app.ml.dataset import FASHION_MNIST_CLASSES, prepare_fashion_mnist

ASSETS_DIR = Path(__file__).resolve().parent.parent.parent / "assets"
DATA_DIR = ASSETS_DIR / "data"


def _numpy_to_qpixmap_gray(arr: np.ndarray) -> QPixmap:
    h, w = arr.shape
    qimg = QImage(arr.data, w, h, w, QImage.Format.Format_Grayscale8).copy()
    return QPixmap.fromImage(qimg)


class WhatIfPage(QWidget):
    def __init__(self, model: SmallCNN, device: torch.device, parent=None) -> None:
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
        self.transforms = [
            ("Original", None),
            ("Blur", "blur"),
            ("Noise", "noise"),
            ("Rotate", "rotate"),
            ("Invert", "invert"),
            ("Brightness", "brightness"),
            ("Contrast", "contrast"),
        ]
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
    def _btn_style(color: str) -> str:
        return f"""
            QPushButton {{
                background-color: {color}20;
                color: {color};
                border: 1px solid {color}80;
                border-radius: 8px;
                padding: 8px 16px;
                font-weight: bold;
            }}
            QPushButton:hover {{ background-color: {color}40; }}
        """

    def set_model(self, model: SmallCNN) -> None:
        self.model = model

    def _load_image(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Выберите изображение",
            str(Path.home()),
            "Images (*.png *.jpg *.jpeg *.bmp *.webp)",
        )
        if not path:
            return
        try:
            self.original_image = Image.open(path).convert("L")
            self._show_all_transforms()
        except Exception as exc:
            QMessageBox.critical(self, "Ошибка", str(exc))

    def _random_sample(self) -> None:
        if self.dataset is None:
            try:
                _, _, _, train_full = prepare_fashion_mnist(
                    root=str(DATA_DIR), batch_size=64, val_fraction=0.1,
                    download=True, num_workers=0,
                )
                self.dataset = train_full
            except Exception as exc:
                QMessageBox.critical(self, "Ошибка", str(exc))
                return
        idx = int(np.random.randint(0, len(self.dataset)))
        img_tensor, label = self.dataset[idx]
        img_arr = (img_tensor.squeeze().numpy() * 127.5 + 127.5).astype(np.uint8)
        self.original_image = Image.fromarray(img_arr, mode="L")
        self._show_all_transforms()

    def _apply_transform(self, name: str) -> None:
        if self.original_image is None:
            QMessageBox.warning(self, "Warning", "Сначала загрузите изображение.")
            return
        self._show_all_transforms()

    def _show_all_transforms(self) -> None:
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

    def _create_frame(self, name: str, img) -> QFrame:
        frame = QFrame()
        frame.setStyleSheet("""
            QFrame {
                background: rgba(15, 23, 42, 0.6);
                border: 1px solid rgba(0, 212, 255, 0.2);
                border-radius: 12px;
            }
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

    def on_activated(self) -> None:
        pass
'''

# ============================================================================
# app/ui/settings_page.py
# ============================================================================
FILES["NeuroLab/app/ui/settings_page.py"] = r'''# -*- coding: utf-8 -*-
"""Страница настроек и информации о системе."""

from __future__ import annotations

import torch
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QFrame, QGridLayout,
)

from app.ml.hardware import get_cpu_name, get_logical_threads, get_ram_gb, get_gpu_info, get_cuda_available


class SettingsPage(QWidget):
    def __init__(self, parent=None) -> None:
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

        cards = [
            ("CPU", cpu_name, "#00d4ff"),
            ("Logical Threads", str(threads), "#a855f7"),
            ("RAM", f"{ram:.1f} GB" if ram > 0 else "Unknown", "#22d3ee"),
            ("GPU", gpu, "#f472b6"),
            ("CUDA", cuda, "#f59e0b"),
            ("PyTorch Device", device, "#10b981"),
        ]

        for i, (title, value, color) in enumerate(cards):
            card = QFrame()
            card.setStyleSheet(f"""
                QFrame {{
                    background-color: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                        stop:0 rgba(15,23,42,0.9), stop:1 rgba(10,14,26,0.9));
                    border: 1px solid {color}40;
                    border-radius: 14px;
                }}
                QFrame:hover {{
                    border: 1px solid {color}aa;
                }}
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
            QFrame {
                background-color: rgba(0, 212, 255, 0.05);
                border: 1px solid rgba(0, 212, 255, 0.25);
                border-radius: 14px;
                padding: 16px;
            }
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

    def on_activated(self) -> None:
        pass
'''

# ============================================================================
# requirements.txt
# ============================================================================
FILES["NeuroLab/requirements.txt"] = r'''PySide6>=6.6.0
torch>=2.2.0
torchvision>=0.17.0
numpy>=1.26.0
Pillow>=10.0.0
pyqtgraph>=0.13.0
py-cpuinfo>=9.0.0
psutil>=5.9.0
'''
