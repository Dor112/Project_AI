from typing import List
import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt

def apply_dark_theme():
    pg.setConfigOptions(antialias=True, background="#0a0e1a", foreground="#cdd6f4")

class LossChart(pg.PlotWidget):
    def __init__(self, parent=None):
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

    def update_data(self, train_loss, val_loss):
        x = list(range(1, len(train_loss) + 1))
        self.train_curve.setData(x, train_loss)
        xv = list(range(1, len(val_loss) + 1))
        self.val_curve.setData(xv, val_loss)
        if train_loss or val_loss:
            self.enableAutoRange()

class AccuracyChart(pg.PlotWidget):
    def __init__(self, parent=None):
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

    def update_data(self, train_acc, val_acc):
        x = list(range(1, len(train_acc) + 1))
        self.train_curve.setData(x, train_acc)
        xv = list(range(1, len(val_acc) + 1))
        self.val_curve.setData(xv, val_acc)
        if train_acc or val_acc:
            self.enableAutoRange()

class ConfusionMatrixWidget(pg.GraphicsLayoutWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.view = self.addViewBox()
        self.view.setAspectLocked(True)
        self.image_item = pg.ImageItem()
        self.view.addItem(self.image_item)
        self.text_items = []

    def set_matrix(self, matrix, labels):
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