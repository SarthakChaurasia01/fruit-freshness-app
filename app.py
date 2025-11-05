import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import json
import io

# ----------------------------
# Page config
# ----------------------------
st.set_page_config(page_title="🍎 Fresh or Rotten Fruit Classifier", layout="centered")
st.title("🍎 Fresh vs Rotten Fruit Classifier")
st.write("Upload an image of a fruit to check whether it's **Fresh** or **Rotten**.")

# ----------------------------
# Load labels
# ----------------------------
@st.cache_resource
def load_labels():
    with open("labels.json", "r") as f:
        labels = json.load(f)
    return labels

# ----------------------------
# Load model
# ----------------------------
@st.cache_resource
def load_model():
    model = models.efficientnet_b2(weights=None)
    num_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_features, 6)  # 6 classes
    model.load_state_dict(torch.load("model.pth", map_location=torch.device("cpu")))
    model.eval()
    return model

# ----------------------------
# Image Preprocessing
# ----------------------------
def transform_image(image):
    transform = transforms.Compose([
        transforms.Resize((260, 260)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)

# ----------------------------
# Prediction function
# ----------------------------
def predict(model, image_tensor, labels):
    with torch.no_grad():
        outputs = model(image_tensor)
        probs = torch.softmax(outputs, dim=1)
        top_prob, top_class = probs.topk(1, dim=1)
        pred_label = labels[str(top_class.item())]
        return pred_label, float(top_prob.item()), probs.squeeze().tolist()

# ----------------------------
# App Interface
# ----------------------------
labels = load_labels()
model = load_model()

uploaded_file = st.file_uploader("📤 Upload a fruit image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(io.BytesIO(uploaded_file.read())).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("🔍 Predict"):
        with st.spinner("Classifying..."):
            tensor = transform_image(image)
            label, prob, all_probs = predict(model, tensor, labels)
            st.success(f"✅ Prediction: **{label}** ({prob*100:.2f}% confidence)")

            if st.checkbox("Show all class probabilities"):
                st.json({labels[str(i)]: round(p*100, 2) for i, p in enumerate(all_probs)})
