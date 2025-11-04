# app.py
import streamlit as st
from PIL import Image
import io
import torch
import torchvision.transforms as T
from torchvision.transforms import InterpolationMode
import numpy as np

# --------- CONFIG ---------
MODEL_PATH = "model.pth"   # or change to a download path
DEVICE = torch.device("cpu")

CLASS_MAP = {
    0: "freshapples",
    1: "freshbanana",
    2: "freshoranges",
    3: "rottenapples",
    4: "rottenbanana",
    5: "rottenoranges",
}

# ImageTransform settings you gave:
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
resize_size = 256
crop_size = 224
interpolation = InterpolationMode.BICUBIC

transform = T.Compose([
    T.Resize(resize_size, interpolation=interpolation),
    T.CenterCrop(crop_size),
    T.ToTensor(),
    T.Normalize(mean=mean, std=std),
])

# --------- MODEL LOADING ---------
@st.cache_resource(show_spinner=True)
def load_model(path: str):
    # If you saved state_dict:
    # from your_model_file import YourModelClass
    # model = YourModelClass(num_classes=6)
    # model.load_state_dict(torch.load(path, map_location="cpu"))
    # model.eval()
    #
    # If you saved entire model (torch.save(model)), use torch.load directly.
    try:
        # Try loading state_dict into a generic torch.nn.Module if user included architecture
        checkpoint = torch.load(path, map_location="cpu")
        # If checkpoint is a dict with 'state_dict', extract it
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
            # You must have the model class available. Placeholder:
            st.error("Model was saved with state_dict. Make sure to define model architecture in this file.")
            return None
        # If it's a whole model object (rare), return it
        if isinstance(checkpoint, torch.nn.Module):
            checkpoint.eval()
            return checkpoint
        # fallback: maybe this is a raw state_dict without wrapper
        # We'll try to load it into a simple torchvision model for illustration (replace with your model).
        st.error("Unable to auto-load model. See app README: you must define model architecture.")
        return None
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# If you have a custom model class, define/import it here and uncomment load logic above.

# --------- PREDICTION UTILITIES ---------
def predict_image(model, pil_image: Image.Image):
    img = pil_image.convert("RGB")
    x = transform(img).unsqueeze(0).to(DEVICE)
    model.eval()
    with torch.no_grad():
        out = model(x)
        probs = torch.nn.functional.softmax(out, dim=1)
        pred = int(probs.argmax(dim=1).item())
        confidence = float(probs.max().item())
    return CLASS_MAP[pred], confidence

# --------- STREAMLIT UI ---------
st.set_page_config(page_title="Fruit Freshness Classifier", layout="centered")
st.title("🍎 Fruit Freshness Classifier")

st.markdown("Upload an image of *apples / bananas / oranges* and the model will predict fresh vs rotten.")

uploaded = st.file_uploader("Upload image", type=["png", "jpg", "jpeg"])

# Optionally, show a button to load the model only once
if st.button("Load model"):
    model = load_model(MODEL_PATH)
    if model is not None:
        st.success("Model loaded successfully.")
    else:
        st.warning("Model load returned None. Check logs / model file and architecture.")

# If a model is available in the environment cache, try to use it:
# NOTE: you probably want to call load_model immediately for demo; adjust as needed.
try:
    model = load_model(MODEL_PATH)
except Exception:
    model = None

if uploaded is not None:
    img = Image.open(io.BytesIO(uploaded.read()))
    st.image(img, caption="Input image", use_column_width=True)
    if model is None:
        st.warning("Model is not loaded. Click 'Load model' or check model path.")
    else:
        label, conf = predict_image(model, img)
        st.success(f"Prediction: **{label}** (confidence {conf:.2%})")

st.caption("Note: Make sure the model architecture in this file matches the saved weights/state_dict.")
