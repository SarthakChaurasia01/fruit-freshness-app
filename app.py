import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import json
import io
from pathlib import Path

# ----------------------------
# Page Config
# ----------------------------
st.set_page_config(page_title="🍎 Fresh or Rotten Fruit Classifier", layout="centered")
st.title("🍎 Fresh vs Rotten Fruit Classifier")
st.write("Upload an image of a fruit to check whether it's **Fresh** or **Rotten**.")

# ----------------------------
# Paths
# ----------------------------
MODEL_PATH = Path("models/model.pth")
LABELS_PATH = Path("labels.json")

# ----------------------------
# Load labels
# ----------------------------
@st.cache_resource
def load_labels():
    if LABELS_PATH.exists():
        with open(LABELS_PATH, "r") as f:
            labels = json.load(f)
        return labels
    else:
        st.warning("⚠️ labels.json not found. Using default labels.")
        return {
            "0": "freshapples",
            "1": "freshbanana",
            "2": "freshoranges",
            "3": "rottenapples",
            "4": "rottenbanana",
            "5": "rottenoranges"
        }

# ----------------------------
# Load model
# ----------------------------
@st.cache_resource
def load_model():
    """
    Robust loader that tries several common .pth formats:
      1) pure state_dict saved with torch.save(model.state_dict())
      2) checkpoint dict with keys like 'model_state_dict' or 'state_dict'
      3) a pickled nn.Module (if loading succeeds)
    If loading fails due to UnpicklingError the function raises a helpful error.
    """
    if not MODEL_PATH.exists():
        st.error(f"Model file not found at {MODEL_PATH}. Please check your folder structure.")
        st.stop()

    try:
        loaded_obj = torch.load(MODEL_PATH, map_location=torch.device("cpu"))
    except Exception as e:
        # Catch pickling/unpickle errors and show user-friendly advice
        st.error("Failed to load the model file with torch.load().")
        st.error(f"Error: {e.__class__.__name__}: {str(e)}")
        st.markdown(
            """
            **Likely causes / next steps**
            - The `.pth` file was saved as a pickled `nn.Module` on a different Python/PyTorch version or used custom classes.
            - If you have access to the training environment, re-save the model as a `state_dict`:
              ```py
              torch.save(model.state_dict(), "model_state_dict.pth")
              ```
              Or export as TorchScript:
              ```py
              scripted = torch.jit.trace(model, dummy_input)
              scripted.save("model_scripted.pt")
              ```
            - Ensure the same PyTorch version is used on deployment as on training machine.
            """
        )
        st.stop()

    # If load returned a dict-like object, try to extract a state_dict
    if isinstance(loaded_obj, dict):
        # common keys in checkpoints
        for key in ("model_state_dict", "state_dict", "model"):
            if key in loaded_obj:
                state_dict = loaded_obj[key]
                break
        else:
            state_dict = loaded_obj  # maybe a plain state_dict

        # instantiate architecture: best-effort with torchvision efficientnet_b2
        try:
            model = models.efficientnet_b2(weights=None)
            # adapt classifier head — adjust output dim if needed
            num_features = model.classifier[1].in_features
            model.classifier[1] = nn.Linear(num_features, 6)
        except Exception:
            st.error("Failed to create EfficientNet-B2 architecture. If your training used a different EfficientNet implementation (timm or custom), you must instantiate the same model here.")
            st.stop()

        try:
            model.load_state_dict(state_dict, strict=False)
            model.eval()
            return model
        except Exception as e:
            st.error("Loaded object looks like a state-dict but failed to load into the torchvision EfficientNet-B2 architecture.")
            st.error(f"load_state_dict error: {e}")
            st.markdown("If you used a different model or custom head when training, recreate the exact architecture here before loading.")
            st.stop()

    # If load returned an nn.Module (the whole model was pickled)
    if isinstance(loaded_obj, nn.Module):
        loaded_obj.eval()
        return loaded_obj

    # Unknown format
    st.error("The loaded .pth file has an unrecognized format. It is not a dict-like state_dict nor a pickled nn.Module.")
    st.stop()


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
