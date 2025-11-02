import gradio as gr
import torch
from PIL import Image
from model_utils import load_model, CLASS_MAP
from transforms import test_transform

# Constants
MODEL_PATH = "model.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Load model
model = load_model(MODEL_PATH, num_classes=len(CLASS_MAP), device=DEVICE)

def predict(image):
    """Predict fruit freshness from an uploaded image."""
    try:
        img = image.convert("RGB")
        x = test_transform(img).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            preds = torch.nn.functional.softmax(model(x), dim=1)
        top_probs, top_idxs = preds.topk(3, dim=1)
        top_probs = top_probs.cpu().numpy().squeeze()
        top_idxs = top_idxs.cpu().numpy().squeeze()

        results = {CLASS_MAP[int(i)]: float(p) for i, p in zip(top_idxs, top_probs)}
        return results
    except Exception as e:
        return {"Error": str(e)}

# Build Gradio UI
demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="Upload Fruit Image"),
    outputs=gr.Label(num_top_classes=3),
    title="🍎 Fruit Freshness Classifier",
    description="Upload an image of a fruit (Apple, Banana, or Orange) to check if it’s fresh or rotten.",
    examples=[
        ["examples/apple_fresh.jpg"],
        ["examples/banana_rotten.jpg"],
        ["examples/orange_fresh.jpg"]
    ],
)

if __name__ == "__main__":
    demo.launch()
