from typing import Tuple, List
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from app.ml.cnn import SmallCNN
from app.ml.dataset import FASHION_MNIST_CLASSES

def predict_pil(model, pil_image, device, top_k=5):
    img = pil_image.convert("L").resize((28, 28), Image.BILINEAR)
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = (arr - 0.5) / 0.5
    tensor = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)
    model.eval()
    with torch.no_grad():
        logits = model(tensor.to(device))
        probs = F.softmax(logits, dim=1)[0].cpu().numpy()
    idxs = np.argsort(probs)[::-1][:top_k]
    top_class = FASHION_MNIST_CLASSES[idxs[0]]
    top_prob = float(probs[idxs[0]])
    ranking = [(FASHION_MNIST_CLASSES[i], float(probs[i])) for i in idxs]
    return top_class, top_prob, ranking