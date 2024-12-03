import joblib
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI()

# Enable CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://lang-scan.vercel.app, https://localhost:8080"],
    allow_credentials=True,
    allow_methods=["POST"],
    allow_headers=["*"],
)

class TextRequest(BaseModel):
    text: str

model = joblib.load('language_detection_model.pkl')

@app.post("/predict")
async def predict_language(request: TextRequest):
    text = request.text
    
    probabilities = model.predict_proba([text])[0]

    langs = model.classes_

    lang_probs = [(langs[i], probabilities[i]) for i in range(len(langs))]

    top_5_langs = sorted(lang_probs, key=lambda x: x[1], reverse=True)[:5]

    return {"top_5_languages": [{"language": lang, "probability": prob} for lang, prob in top_5_langs]}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)