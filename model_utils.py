# model_utils.py (diagnostic version)
import torch
from torchvision import models
from pathlib import Path
import traceback
import sys

LABELS_PATH = "labels.json"

def build_model(num_classes=6):
    model = models.resnet18(pretrained=False)
    in_features = model.fc.in_features
    model.fc = torch.nn.Linear(in_features, num_classes)
    return model

def load_model(model_path="models/model.pth", device="cpu"):
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found at {model_path.resolve()} - please upload model.pth at this path.")

    device = torch.device(device)
    last_exc = None

    # Try loading with torch.load first
    try:
        loaded = torch.load(str(model_path), map_location=device)
        print("INFO: torch.load succeeded. Loaded type:", type(loaded))
        # If it's an nn.Module instance
        if hasattr(loaded, "eval") and hasattr(loaded, "state_dict"):
            print("INFO: loaded object looks like an nn.Module. Using it directly.")
            model = loaded
            model.to(device)
            model.eval()
            return model, device

        # If it's a dict / state_dict
        if isinstance(loaded, dict):
            print("INFO: loaded object is dict. Keys:", list(loaded.keys())[:50])
            # common wrappers:
            if "state_dict" in loaded:
                state = loaded["state_dict"]
                print("INFO: found 'state_dict' key. Using that as state dict.")
                model = build_model(num_classes=6)
                model.load_state_dict(state)
                model.to(device)
                model.eval()
                return model, device

            # lightning style often has "state_dict" or keys with "model." prefix
            # Heuristic: check if values are tensors
            sample_val = next(iter(loaded.values()))
            if hasattr(sample_val, "shape"):
                print("INFO: top-level dict appears to be a state_dict (tensor values). Loading into fallback model.")
                model = build_model(num_classes=6)
                try:
                    model.load_state_dict(loaded)
                except RuntimeError as e:
                    print("WARNING: load_state_dict raised RuntimeError:", e)
                    # Try removing possible 'model.' prefixes
                    new_state = {}
                    for k, v in loaded.items():
                        new_key = k.replace("model.", "") if isinstance(k, str) else k
                        new_state[new_key] = v
                    try:
                        model.load_state_dict(new_state)
                    except Exception as e2:
                        print("ERROR: still failed after stripping 'model.' prefixes:", e2)
                        raise
                model.to(device)
                model.eval()
                return model, device

        # If we reach here, try torch.jit
        last_exc = RuntimeError("Unable to interpret loaded object as module or state_dict.")
    except Exception as e:
        last_exc = e
        print("ERROR: torch.load failed or yielded unexpected object. Exception:")
        traceback.print_exc(file=sys.stdout)

    # Try torch.jit.load
    try:
        print("INFO: Attempting torch.jit.load as fallback...")
        model = torch.jit.load(str(model_path), map_location=device)
        print("INFO: torch.jit.load succeeded. Type:", type(model))
        model.to(device)
        model.eval()
        return model, device
    except Exception as e2:
        print("ERROR: torch.jit.load failed. Exception:")
        traceback.print_exc(file=sys.stdout)
        # raise a helpful error containing both exceptions
        raise RuntimeError(f"Failed to load model.pth. torch.load error: {last_exc}\n torch.jit.load error: {e2}") from e2
