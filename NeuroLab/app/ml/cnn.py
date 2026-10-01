import torch
import torch.nn as nn

class SmallCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(32*7*7, 128)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(128, num_classes)
        self._feature_maps = {}
        self._register_hooks()

    def _register_hooks(self):
        def hook(name):
            def fn(m, i, o):
                self._feature_maps[name] = o.detach()
            return fn
        self.conv1.register_forward_hook(hook("conv1"))
        self.conv2.register_forward_hook(hook("conv2"))

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.flatten(x)
        x = self.relu3(self.fc1(x))
        return self.fc2(x)

    def get_feature_maps(self):
        return dict(self._feature_maps)

    def clear_feature_maps(self):
        self._feature_maps.clear()

def build_model(num_classes=10):
    return SmallCNN(num_classes)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)