# app.py
import streamlit as st
from PIL import Image
import io
import os
import json

from utils import get_transforms, load_labels, build_model, predict_image

st.set_page_config(page_title="Fruit Freshness Classifier", layout="centered")

st.title("🍎 Fruit Freshness Classifier (EfficientNet-B2)")
st.markdown("Upload an image of an apple / banana / orange — the model predicts fresh vs rotten.")

# Sidebar
st.sidebar.header("Model")
model_path = st.sidebar.text_input("Model path", value="models/model.pth")
labels_path = st.sidebar.text_input("Labels JSON", value="labels.json")
top_k = st.sidebar.slider("Top K predictions", 1, 6, 3)

# Load labels and model once (cache)
@st.cache_resource(show_spinner=False)
def load_resources(model_path, labels_path):
    labels = load_labels(labels_path)
    model, device = build_model(num_classes=len(labels), checkpoint_path=model_path)
    transforms = get_transforms()
    return labels, model, device, transforms

try:
    labels, model, device, transforms = load_resources(model_path, labels_path)
except Exception as e:
    st.error(f"Error loading model or labels: {e}")
    st.stop()

uploaded = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
if uploaded:
    try:
        img = Image.open(io.BytesIO(uploaded.read())).convert("RGB")
    except Exception as e:
        st.error("Invalid image uploaded.")
        st.stop()

    st.image(img, caption="Input image", use_column_width=True)

    if st.button("Predict"):
        with st.spinner("Predicting..."):
            try:
                results = predict_image(model, device, img, transforms, labels, topk=top_k)
            except Exception as e:
                st.error(f"Prediction error: {e}")
                st.stop()

        st.subheader("Top predictions")
        for label, prob in results:
            st.write(f"**{label}** — {prob*100:.2f}%")
else:
    st.info("Upload an image to get a prediction.")

st.write("---")
st.write("Model info:")
st.write(f"- Device: `{device}`")
st.write(f"- Classes: {json.dumps(labels, indent=2)}")
