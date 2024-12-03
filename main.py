import joblib
import uvicorn
import os
import zipfile
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

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

def unzip_model():
    zip_file = 'language_detection_model.zip'
    model_file = '/tmp/language_detection_model.pkl'  # Use /tmp for writable file system
    
    if os.path.exists(zip_file) and not os.path.exists(model_file):
        print(f"Unzipping model from {zip_file}...")
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            zip_ref.extractall('/tmp')  # Extract to /tmp directory
        print("Model unzipped successfully.")
    else:
        if os.path.exists(model_file):
            print("Model file already exists. Skipping unzip.")
        else:
            print("Model zip file not found!")

# Unzip the model if it's not already extracted
unzip_model()

# Load the model
model = joblib.load('/tmp/language_detection_model.pkl')
print("Model loaded successfully!")

@app.post("/predict")
async def predict_language(request: TextRequest):
    text = request.text
    
    probabilities = model.predict_proba([text])[0]

    langs = model.classes_

    lang_probs = [(langs[i], probabilities[i]) for i in range(len(langs))]

    top_5_langs = sorted(lang_probs, key=lambda x: x[1], reverse=True)[:5]

    return {"top_5_languages": [{"language": lang, "probability": prob} for lang, prob in top_5_langs]}

if __name__ == "__main__":
    # Unzip the model if necessary when starting the server
    unzip_model()

    # Load the model
    model = joblib.load('/tmp/language_detection_model.pkl')
    print("Model loaded successfully!")
    
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)