import streamlit as st
elif model_path_input.strip() != "":
model_path_to_load = model_path_input.strip()
else:
model_path_to_load = DEFAULT_MODEL_PATH if use_default else None


if st.sidebar.button("Load model"):
if not model_path_to_load:
st.sidebar.error("No model path provided.")
else:
try:
with st.spinner("Loading model..."):
st.session_state.model = load_model_from_path(model_path_to_load, num_classes=NUM_CLASSES, device=device)
st.session_state.model_path = model_path_to_load
st.sidebar.success(f"Model loaded from: {model_path_to_load}")
except Exception as e:
st.sidebar.error(f"Failed to load model: {e}")


# Auto-load if file exists and model not loaded
if st.session_state.model is None and os.path.exists(DEFAULT_MODEL_PATH):
try:
with st.spinner("Auto-loading default model..."):
st.session_state.model = load_model_from_path(DEFAULT_MODEL_PATH, num_classes=NUM_CLASSES, device=device)
st.session_state.model_path = DEFAULT_MODEL_PATH
st.sidebar.success(f"Auto-loaded default model from {DEFAULT_MODEL_PATH}")
except Exception:
pass


st.write("---")


uploaded_image = st.file_uploader("Upload image (jpg/png)", type=["jpg", "jpeg", "png"])
if uploaded_image is not None:
try:
img = Image.open(uploaded_image).convert("RGB")
except Exception as e:
st.error(f"Can't open image: {e}")
img = None


if img is not None:
st.image(img, caption="Uploaded image", use_column_width=True)
if st.session_state.model is None:
st.warning("Model not loaded. Load a model from the sidebar to enable predictions.")
else:
topk = st.slider("Top K", min_value=1, max_value=min(6, NUM_CLASSES), value=1)
if st.button("Predict"):
# Preprocess
x = test_transform(img).unsqueeze(0).to(device)
with torch.no_grad():
out = st.session_state.model(x)
probs = torch.nn.functional.softmax(out, dim=1)
top_probs, top_idxs = probs.topk(topk, dim=1)
top_probs = top_probs.cpu().numpy().squeeze()
top_idxs = top_idxs.cpu().numpy().squeeze()
if topk == 1:
top_probs = [float(top_probs)]
top_idxs = [int(top_idxs)]
st.success("Prediction")
for idx, p in zip(top_idxs, top_probs):
label = CLASS_MAP.get(int(idx), f"class_{idx}")
st.write(f"**{label}** — {p*100:.2f}%")


st.write("---")
st.subheader("Class mapping")
for k, v in CLASS_MAP.items():
st.write(f"{k} → {v}")


st.caption("Model expects RGB images resized to {}x{} during preprocessing.".format(IMG_SIZE, IMG_SIZE))
