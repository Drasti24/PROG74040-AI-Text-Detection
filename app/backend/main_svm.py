from pathlib import Path

import joblib
import numpy as np

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field


app = FastAPI(
    title="AI Text Detection API",
    description="Detect whether text is human-written or AI-generated.",
    version="1.0"
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = (
    BASE_DIR
    / "deploy_models"
    / "tfidf_svm_model.joblib"
)

VECTORIZER_PATH = (
    BASE_DIR
    / "deploy_models"
    / "tfidf_svm_vectorizer.joblib"
)


print("Loading TF-IDF vectorizer and SVM model...")

vectorizer = joblib.load(VECTORIZER_PATH)
model = joblib.load(MODEL_PATH)

print("SVM model loaded successfully.")


class TextRequest(BaseModel):
    text: str = Field(
        ...,
        min_length=1,
        description="Text to classify as human-written or AI-generated."
    )


@app.get("/")
def root():
    return {
        "message": "AI Text Detection API is running",
        "model": "TF-IDF + SVM"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model": "TF-IDF + SVM"
    }


@app.post("/predict")
def predict(request: TextRequest):

    text = request.text.strip()

    if not text:
        raise HTTPException(
            status_code=400,
            detail="Text cannot be empty."
        )

    transformed_text = vectorizer.transform([text])

    prediction = int(
        model.predict(transformed_text)[0]
    )

    decision_score = float(
        model.decision_function(transformed_text)[0]
    )

    # Convert decision score into a simple 0–1 display score.
    # This is not a calibrated probability.
    confidence_score = 1 / (
        1 + np.exp(-abs(decision_score))
    )

    if prediction == 1:
        prediction_label = "AI Generated"
    else:
        prediction_label = "Human Written"

    return {
        "prediction": prediction_label,
        "confidence": round(confidence_score, 4),
        "decision_score": round(decision_score, 4)
    }