import torch
import timm
import os

CLASS_MAP = {
    0: "Apple (Fresh)",
    1: "Apple (Rotten)",
    2: "Banana (Fresh)",
    3: "Banana (Rotten)",
    4: "Orange (Fresh)",
    5: "Orange (Rotten)"
}

def build_effnet_b2(num_classes=6, pretrained_backbone=False):
    """Build EfficientNet-B2 model."""
    model = timm.create_model('efficientnet_b2', pretrained=pretrained_backbone, num_classes=num_classes)
    return model

def load_model(path, num_classes=6, device="cpu"):
    """Load model from .pth file."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found at: {path}")

    model = build_effnet_b2(num_classes=num_classes, pretrained_backbone=False)
    model.to(device)

    state = torch.load(path, map_location=device)
    if isinstance(state, dict):
        if "state_dict" in state:
            state = state["state_dict"]
        elif "model_state_dict" in state:
            state = state["model_state_dict"]

    # Clean keys if saved with DataParallel
    new_state = {}
    for k, v in state.items():
        new_state[k.replace("module.", "")] = v

    model.load_state_dict(new_state, strict=False)
    model.eval()
    return model
