# Fruit Freshness Classifier (Streamlit)


This repository contains a Streamlit app that loads a PyTorch EfficientNet-B2 model and predicts whether an uploaded fruit image (apple/banana/orange) is fresh or rotten.


## Files
- `app.py` — Streamlit app (UI + inference)
- `model_loader.py` — helper to build and load the model
- `transforms.py` — train & test transforms (you provided these)
- `requirements.txt` — dependencies
- `models/model.pth` — put your trained model here (or upload via UI)


## Run locally
1. Create a virtual environment and install dependencies:


```bash
python -m venv venv
source venv/bin/activate # on Windows: venv\Scripts\activate
pip install -r requirements.txt
