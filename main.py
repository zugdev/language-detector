import joblib
import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import gdown

app = FastAPI()

# Enable CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class TextRequest(BaseModel):
    text: str

# Load from drive when the server starts
print("Downloading model from Google Drive...")
drive_id = '1ruTX5DbM2wPoBU8KShiRoTL2OUSAuTA4'
output = 'language_detection_model.pkl'
gdown.download(f'https://drive.google.com/uc?id={drive_id}&confirm=t', output, quiet=False)
model = joblib.load(output)
print("Model loaded successfully!")

@app.post("/predict")
async def predict_language(request: TextRequest):
    text = request.text
    
    probabilities = model.predict_proba([text])[0]

    langs = model.classes_

    lang_probs = [(langs[i], probabilities[i]) for i in range(len(langs))]

    top_5_langs = sorted(lang_probs, key=lambda x: x[1], reverse=True)[:5]

    return {"top_5_languages": [{"language": lang, "probability": prob} for lang, prob in top_5_langs]}