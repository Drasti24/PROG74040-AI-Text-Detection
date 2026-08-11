from pathlib import Path
from fastapi.middleware.cors import CORSMiddleware
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)


# --------------------------------------------------
# FastAPI app
# --------------------------------------------------

app = FastAPI(
    title="AI Text Detection API",
    description="Detect whether text is human-written or AI-generated.",
    version="1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------------------------------------------------
# Model path
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = PROJECT_ROOT / "models" / "roberta_final"

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"RoBERTa model not found at: {MODEL_PATH}"
    )


# --------------------------------------------------
# Load tokenizer and model
# --------------------------------------------------

print("Loading RoBERTa model...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH
)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_PATH
)

model.eval()

print("RoBERTa model loaded successfully.")


# --------------------------------------------------
# Request model
# --------------------------------------------------

class TextRequest(BaseModel):
    text: str = Field(
        ...,
        min_length=1,
        description="Text to classify as Human or AI-generated."
    )


# --------------------------------------------------
# Root endpoint
# --------------------------------------------------

@app.get("/")
def root():
    return {
        "message": "AI Text Detection API is running"
    }


# --------------------------------------------------
# Prediction endpoint
# --------------------------------------------------

@app.post("/predict")
def predict(request: TextRequest):

    text = request.text.strip()

    if not text:
        raise HTTPException(
            status_code=400,
            detail="Text cannot be empty."
        )

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=256
    )

    with torch.inference_mode():
        outputs = model(**inputs)

        probabilities = torch.softmax(
            outputs.logits,
            dim=-1
        )[0]

    human_probability = probabilities[0].item()
    ai_probability = probabilities[1].item()

    predicted_class = int(
        torch.argmax(probabilities).item()
    )

    if predicted_class == 1:
        prediction = "AI Generated"
        confidence = ai_probability
    else:
        prediction = "Human Written"
        confidence = human_probability

    return {
        "prediction": prediction,
        "confidence": round(confidence, 4),
        "human_probability": round(human_probability, 4),
        "ai_probability": round(ai_probability, 4)
    }