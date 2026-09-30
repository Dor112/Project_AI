import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional
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
    history: dict = field(default_factory=lambda: {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []})

class Trainer:
    def __init__(self, model, train_loader, val_loader, device, learning_rate=1e-3):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.stats = TrainingStats()

    def train_epoch(self, step_callback=None, should_stop=None, should_pause=None):
        self.model.train()
        running_loss, correct, total = 0.0, 0, 0
        n_batches = len(self.train_loader)
        t0 = time.time()
        for batch_idx, (inputs, targets) in enumerate(self.train_loader):
            if should_stop and should_stop():
                break
            while should_pause and should_pause():
                time.sleep(0.1)
                if should_stop and should_stop():
                    break
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            if step_callback and (batch_idx % max(1, n_batches // 10) == 0 or batch_idx == n_batches - 1):
                self.stats.train_loss = running_loss / max(1, total)
                self.stats.train_acc = 100.0 * correct / max(1, total)
                step_callback(self.stats, batch_idx + 1, n_batches)
        self.stats.train_loss = running_loss / max(1, total)
        self.stats.train_acc = 100.0 * correct / max(1, total)
        self.stats.elapsed = time.time() - t0
        return self.stats

    def validate(self):
        self.model.eval()
        running_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for inputs, targets in self.val_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                running_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        self.stats.val_loss = running_loss / max(1, total)
        self.stats.val_acc = 100.0 * correct / max(1, total)
        return self.stats

    def save_checkpoint(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model_state_dict": self.model.state_dict()}, path)