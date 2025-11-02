"""
Helpers for loading model and preprocessing images.

IMPORTANT:
If your saved checkpoint is a state_dict from a custom model (not ResNet18),
edit build_model() to recreate your original architecture exactly and then
load the state_dict into it.

Expected model file: model.pth at repo root.
"""

import torch
import torchvision.transforms as T
from torchvision import models
from pathlib import Path
import json

# labels file location
LABELS_PATH = "labels.json"

def build_model(num_classes=6):
    """
    Fallback model architecture used if loading a state_dict fails to directly
    load a saved model object. This uses a ResNet18 backbone with a custom head.
    If your model was different, replace this function accordingly.
    """
    # Create a ResNet18 and replace final fc to match num_classes
    model = models.resnet18(pretrained=False)
    in_features = model.fc.in_features
    model.fc = torch.nn.Linear(in_features, num_classes)
    return model

def load_model(model_path="model.pth", device="cpu"):
    """
    Attempts to load a model saved in either form:
     - torch.save(model)  ---> loads full object with torch.load
     - torch.save(model.state_dict()) ---> loads state_dict into fallback model

    Returns: model (on device), device
    """
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found at {model_path.resolve()} - please upload model.pth in repo root.")

    device = torch.device(device)
    try:
        # Try to load full model object
        loaded = torch.load(str(model_path), map_location=device)
        if isinstance(loaded, dict) and "state_dict" in loaded and len(loaded) == 1:
            # Some frameworks wrap state_dict in dict
            state = loaded["state_dict"]
            model = build_model(num_classes=6)
            model.load_state_dict(state)
        elif isinstance(loaded, dict) and not any(hasattr(v, "__call__") for v in loaded.values()):
            # Heuristic: it's a state_dict (mapping of tensors)
            state = loaded
            model = build_model(num_classes=6)
            model.load_state_dict(state)
        elif hasattr(loaded, "eval") and hasattr(loaded, "state_dict"):
            # It's likely an actual nn.Module object
            model = loaded
        else:
            # Fallback: assume state_dict-like
            try:
                model = build_model(num_classes=6)
                model.load_state_dict(loaded)
            except Exception as e:
                raise RuntimeError("Couldn't interpret the saved model file. If you saved only the state_dict with a custom model, modify build_model() to recreate that architecture and try again.") from e
    except RuntimeError as e:
        # Try loading as state_dict into known architecture
        state = torch.load(str(model_path), map_location=device)
        model = build_model(num_classes=6)
        model.load_state_dict(state)
    except Exception as e:
        # Final attempt: try torch.jit or other
        try:
            model = torch.jit.load(str(model_path), map_location=device)
        except Exception as e2:
            raise RuntimeError(f"Failed to load model.pth: {e}\nAlso failed to load with torch.jit: {e2}\nIf your model was saved as state_dict of a custom architecture, edit build_model() to match your model class.") from e2

    model.to(device)
    model.eval()
    return model, device

# Preprocessing: adapt to model's expected size. Using common 224x224 + ImageNet norm.
def preprocess_image(pil_image, size=224):
    transforms = T.Compose([
        T.Resize((size, size)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
    ])
    return transforms(pil_image)
