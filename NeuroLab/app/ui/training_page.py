import threading
from pathlib import Path
from typing import Optional
import numpy as np
import torch
from PySide6.QtCore import Qt, QObject, Signal, QThread
from PySide6.QtGui import QFont
from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QFrame, QSpinBox, QDoubleSpinBox, QProgressBar, QMessageBox, QGridLayout, QApplication, QComboBox
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
    def __init__(self, trainer, epochs, signals):
        super().__init__()
        self.trainer = trainer
        self.epochs = epochs
        self.signals = signals
        self._stop_flag = threading.Event()
        self._pause_flag = threading.Event()

    def request_stop(self):
        self._stop_flag.set()
        self._pause_flag.clear()

    def toggle_pause(self):
        if self._pause_flag.is_set():
            self._pause_flag.clear()
        else:
            self._pause_flag.set()

    def is_paused(self):
        return self._pause_flag.is_set()

    def is_stopped(self):
        return self._stop_flag.is_set()

    def run(self):
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
    def __init__(self, model, device, parent=None):
        super().__init__(parent)
        self.model = model
        self.device = device
        self.trainer = None
        self.worker = None
        self.thread = None
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
            QFrame { background-color: rgba(15, 23, 42, 0.6); border: 1px solid rgba(0, 212, 255, 0.2); border-radius: 12px; }
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
            QProgressBar { background: #111827; border: 1px solid #1f2937; border-radius: 8px; text-align: center; color: #cdd6f4; height: 24px; }
            QProgressBar::chunk { background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #00d4ff, stop:1 #a855f7); border-radius: 7px; }
        """)
        outer.addWidget(self.progress)
        metrics = QHBoxLayout()
        self.lbl_epoch = self._metric("Epoch", "—")
        self.lbl_train_loss = self._metric("Train Loss", "—")
        self.lbl_train_acc = self._metric("Train Acc", "—")
        self.lbl_val_loss = self._metric("Val Loss", "—")
        self.lbl_val_acc = self._metric("Val Acc", "—")
        self.lbl_time = self._metric("Time/Epoch", "—")
        for w in (self.lbl_epoch, self.lbl_train_loss, self.lbl_train_acc, self.lbl_val_loss, self.lbl_val_acc, self.lbl_time):
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
    def _spin_style():
        return """
            QSpinBox, QDoubleSpinBox { background: #111827; color: #cdd6f4; border: 1px solid #1f2937; border-radius: 6px; padding: 4px 8px; min-width: 100px; }
            QSpinBox:hover, QDoubleSpinBox:hover { border: 1px solid #00d4ff; }
        """

    @staticmethod
    def _combo_style():
        return """
            QComboBox { background: #111827; color: #cdd6f4; border: 1px solid #1f2937; border-radius: 6px; padding: 4px 8px; min-width: 120px; }
            QComboBox:hover { border: 1px solid #00d4ff; }
            QComboBox::drop-down { border: none; }
        """

    @staticmethod
    def _btn_style(color):
        return f"""
            QPushButton {{ background-color: {color}20; color: {color}; border: 1px solid {color}80; border-radius: 8px; padding: 8px 16px; font-weight: bold; }}
            QPushButton:hover {{ background-color: {color}40; }}
            QPushButton:disabled {{ color: #475569; border: 1px solid #334155; background: transparent; }}
        """

    @staticmethod
    def _labeled(text, widget):
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
    def _metric(title, value):
        f = QFrame()
        f.setStyleSheet("""
            QFrame { background: rgba(0, 212, 255, 0.05); border: 1px solid rgba(0, 212, 255, 0.2); border-radius: 10px; }
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

    def _set_metric(self, frame, value):
        frame._value_label.setText(value)

    def _on_mode_changed(self, index):
        if index == 0:
            self.spin_epochs.setValue(3)
            self.spin_batch.setValue(64)
        elif index == 1:
            self.spin_epochs.setValue(5)
            self.spin_batch.setValue(128)
        elif index == 2:
            self.spin_epochs.setValue(10)
            self.spin_batch.setValue(128)

    def _start_training(self):
        if self.worker is not None and self.worker.isRunning():
            return
        mode = self.combo_mode.currentIndex()
        train_subset = None
        test_subset = None
        if mode == 0:
            train_subset = 5000
            test_subset = 1000
        elif mode == 1:
            train_subset = 20000
            test_subset = 5000
        try:
            self.status.setText("Подготовка данных...")
            QApplication.processEvents()
            train_loader, val_loader, _, _ = prepare_fashion_mnist(
                root=str(DATA_DIR), batch_size=self.spin_batch.value(), val_fraction=0.1,
                download=True, num_workers=0, train_subset_size=train_subset, test_subset_size=test_subset,
            )
        except Exception as exc:
            QMessageBox.critical(self, "Ошибка", f"Не удалось загрузить данные: {exc}")
            self.status.setText(f"Ошибка: {exc}")
            return
        self.trainer = Trainer(
            model=self.model, train_loader=train_loader, val_loader=val_loader,
            device=self.device, learning_rate=self.spin_lr.value(),
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

    def _toggle_pause(self):
        if self.worker is None:
            return
        self.worker.toggle_pause()
        if self.worker.is_paused():
            self.btn_pause.setText("▶  Resume")
            self.status.setText("Обучение приостановлено.")
        else:
            self.btn_pause.setText("⏸  Pause")
            self.status.setText("Обучение возобновлено.")

    def request_stop(self):
        if self.worker is not None and self.worker.isRunning():
            self.worker.request_stop()
            self.worker.wait(5000)

    def _stop_training(self):
        if self.worker is None:
            return
        self.worker.request_stop()
        self.status.setText("Остановка...")

    def _reset_model(self):
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

    def _on_step(self, stats, batch_idx, n_batches):
        frac = batch_idx / max(1, n_batches)
        epoch_frac = (stats.epoch - 1 + frac) / max(1, self.spin_epochs.value())
        self.progress.setValue(int(epoch_frac * 1000))
        self._set_metric(self.lbl_epoch, f"{stats.epoch} / {self.spin_epochs.value()}")
        self._set_metric(self.lbl_train_loss, f"{stats.train_loss:.4f}")
        self._set_metric(self.lbl_train_acc, f"{stats.train_acc:.2f}%")

    def _on_epoch_done(self, stats):
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

    def _compute_confusion_matrix(self):
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

    def _on_finished(self, message):
        self.status.setText(message)
        self.progress.setValue(1000)
        self._cleanup_worker()

    def _on_error(self, message):
        QMessageBox.critical(self, "Ошибка обучения", message)
        self.status.setText(f"Ошибка: {message}")
        self._cleanup_worker()

    def _cleanup_worker(self):
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