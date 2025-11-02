import streamlit as st
from PIL import Image
import io
import torch
import torch.nn.functional as F
import json
from model_utils import load_model, preprocess_image, LABELS_PATH

st.set_page_config(page_title="Fruit Freshness Detector", page_icon="🍎")

st.title("🍏 Fruit Freshness Detector")
st.write("Upload an image of an apple / banana / orange and the model will predict whether it's fresh or rotten.")

# Load labels and model (cached)
@st.cache_resource
@st.cache_resource
def load_resources(model_path="models/model.pth", device="cpu"):

    with open(LABELS_PATH, "r") as f:
        labels = json.load(f)
    model, device = load_model(model_path, device=device)
    return model, device, labels

# Try to use GPU if available (Streamlit Cloud usually provides CPU)
device = "cuda" if torch.cuda.is_available() else "cpu"
model, device, labels = load_resources(device=device)

st.sidebar.markdown("### Options")
show_probs = st.sidebar.checkbox("Show probabilities", value=True)
st.sidebar.markdown("Model file: `model.pth` (place at repo root)")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
camera = st.camera_input("Or take a photo (mobile)")

image_bytes = None
if uploaded_file is not None:
    image_bytes = uploaded_file.read()
elif camera is not None:
    image_bytes = camera.read()

if image_bytes:
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    st.image(image, caption="Input image", use_column_width=True)
    st.write("")
    st.write("Detecting...")

    # Preprocess
    input_tensor = preprocess_image(image).unsqueeze(0).to(device)  # shape: [1, C, H, W]

    model.eval()
    with torch.inference_mode():
        outputs = model(input_tensor)  # raw logits or probabilities

        # Convert logits → probs if needed
        if outputs.ndim == 1 or outputs.shape[1] == 1:
            # single output or binary? treat as logits
            probs = torch.sigmoid(outputs)
            probs = probs.detach().cpu().numpy().squeeze()
        else:
            probs = F.softmax(outputs, dim=1).detach().cpu().numpy().squeeze()

    # Get predicted class
    if probs.ndim == 0:
        # scalar
        pred_idx = int(probs >= 0.5)
    else:
        pred_idx = int(probs.argmax())

    # Map label
    label_text = labels.get(str(pred_idx), f"Class {pred_idx}")

    st.subheader(f"Prediction: **{label_text}**")

    if show_probs:
        st.write("Confidence:")
        if hasattr(probs, "tolist"):
            probs_list = probs.tolist() if isinstance(probs, (list, tuple,)) or probs.ndim > 0 else [float(probs)]
        else:
            probs_list = [float(probs)]

        # If multi-class
        if len(probs_list) > 1:
            # show each class with probability (sorted)
            pairs = [(int(i), float(p)) for i, p in enumerate(probs_list)]
            pairs_sorted = sorted(pairs, key=lambda x: x[1], reverse=True)
            for idx, p in pairs_sorted:
                st.write(f"- {labels.get(str(idx), str(idx))}: {p*100:.2f}%")
        else:
            # binary/single value
            st.write(f"- {label_text}: {probs_list[0]*100:.2f}%")

    st.success("Done!")
else:
    st.info("Upload an image of apples / banana / oranges (fresh or rotten).")

st.markdown("---")
st.markdown("**Mapping used** (id → label):")
st.json(labels)
