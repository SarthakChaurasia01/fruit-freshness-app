# utils.py
import json
import os
from typing import Dict

import torch
import timm
from torchvision import transforms
from PIL import Image

# Default image transforms (resize to 256, center-crop to 224, normalize with ImageNet stats)
def get_transforms():
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

def load_labels(labels_path: str = "labels.json") -> Dict[int, str]:
    with open(labels_path, "r") as f:
        data = json.load(f)
    # ensure keys are ints
    return {int(k): v for k, v in data.items()}

def build_model(num_classes: int = 6, checkpoint_path: str = "models/model.pth", device: str = None):
    """
    Build EfficientNet-B2 and load weights from checkpoint_path.
    Uses timm to create the architecture.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = timm.create_model("efficientnet_b2", pretrained=False, num_classes=num_classes)
    model.to(device)

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

    # load state dict (allowing possible strict=False if naming mismatch)
    sd = torch.load(checkpoint_path, map_location=device)
    # If checkpoint saved as {'model_state_dict': ...} handle that:
    if isinstance(sd, dict) and ("model_state_dict" in sd or "state_dict" in sd):
        key = "model_state_dict" if "model_state_dict" in sd else "state_dict"
        sd = sd[key]

    try:
        model.load_state_dict(sd)
    except Exception as e:
        # try non-strict load (useful if you saved differently)
        model.load_state_dict(sd, strict=False)

    model.eval()
    return model, device

def predict_image(model, device, pil_image: Image.Image, transforms, labels: Dict[int, str], topk: int = 3):
    """
    Returns list of (label, prob) sorted descending.
    """
    x = transforms(pil_image).unsqueeze(0).to(device)  # shape: (1,C,H,W)
    with torch.no_grad():
        out = model(x)
        probs = torch.nn.functional.softmax(out, dim=1)[0]
        topk_vals, topk_idx = torch.topk(probs, k=topk)
        results = []
        for val, idx in zip(topk_vals.cpu().numpy(), topk_idx.cpu().numpy()):
            label = labels.get(int(idx), str(idx))
            results.append((label, float(val)))
    return results
