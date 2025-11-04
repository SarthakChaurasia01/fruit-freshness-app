# app.py
import streamlit as st
from PIL import Image
import io
import torch
import torch.nn.functional as F
from torchvision import transforms, models
import torchvision.transforms.functional as TF

st.set_page_config(page_title="Fruit Freshness Classifier", layout="centered")

# --- CLASSES / LABELS ---
LABELS = {
    0: "freshapples",
    1: "freshbanana",
    2: "freshoranges",
    3: "rottenapples",
    4: "rottenbanana",
    5: "rottenoranges"
}

MODEL_PATH = "model.pth"  # make sure this file is in the repo / same folder as app.py

@st.cache_resource(show_spinner=False)
def load_model(path: str):
    """
    Load a torch model saved with torch.save(model.state_dict(), path) or torch.save(model, path).
    This function tries state_dict load first (recommended).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Build a model architecture matching your training model.
    # Below uses EfficientNet_B2 from torchvision as an example; adjust if your arch differs.
    try:
        # If you trained with an EfficientNet B2 feature extractor head:
        model = models.efficientnet_b2(weights=None)  # no pretrained weights here
        # Replace final classifier to match 6 classes (modify if yours differs)
        in_features = model.classifier[1].in_features if hasattr(model, "classifier") else model.classifier.in_features
        model.classifier[1] = torch.nn.Linear(in_features, 6)
    except Exception:
        # fallback simple model stub (safer than crash)
        model = models.resnet18(weights=None)
        model.fc = torch.nn.Linear(model.fc.in_features, 6)

    try:
        # try loading state_dict (most common)
        state = torch.load(path, map_location=device)
        if isinstance(state, dict) and not any(k.startswith("_") for k in state.keys()):
            model.load_state_dict(state)
        else:
            # maybe saved whole model
            model = state
    except Exception as e:
        st.error(f"Error loading model from {path}: {e}")
    model.to(device)
    model.eval()
    return model, device

@st.cache_data
def get_transforms():
    # Use the same transforms you provided:
    return transforms.Compose([
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

def predict_image(model, device, pil_img):
    tf = get_transforms()
    img_t = tf(pil_img).unsqueeze(0).to(device)  # 1 x C x H x W
    with torch.no_grad():
        out = model(img_t)
        probs = F.softmax(out, dim=1).cpu().numpy()[0]
        top_idx = int(probs.argmax())
        return LABELS[top_idx], float(probs[top_idx]), probs

# UI
st.title("🍎 Fruit Freshness Classifier")
st.write("Upload an image of an apple, banana, or orange. The model predicts `fresh` or `rotten`.")

uploaded = st.file_uploader("Choose an image...", type=["jpg","jpeg","png"])
col1, col2 = st.columns([2,1])

# Load model once
with st.spinner("Loading model..."):
    model, device = load_model(MODEL_PATH)

if uploaded:
    img = Image.open(io.BytesIO(uploaded.read())).convert("RGB")
    col1.image(img, caption="Input image", use_column_width=True)
    if st.button("Predict"):
        with st.spinner("Predicting..."):
            label, prob, probs = predict_image(model, device, img)
        st.success(f"Prediction: **{label}** — confidence: **{prob*100:.2f}%**")
        # show class probabilities table
        import pandas as pd
        df = pd.DataFrame({
            "class": [LABELS[i] for i in range(len(probs))],
            "probability": [float(p) for p in probs]
        }).sort_values("probability", ascending=False)
        col2.table(df.reset_index(drop=True))

st.markdown("---")
st.caption("Note: The app expects `model.pth` in the same folder as `app.py` (or edit MODEL_PATH).")
