from typing import Dict
import numpy as np
import torch
from PIL import Image
from app.ml.cnn import SmallCNN

def activation_to_grid(activation, max_channels=16):
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

def pil_to_tensor_for_model(pil_image):
    img = pil_image.convert("L").resize((28, 28), Image.BILINEAR)
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = (arr - 0.5) / 0.5
    return torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)