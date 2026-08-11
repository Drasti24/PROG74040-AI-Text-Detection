import torch

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
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


# --------------------------------------------------
# CORS
# --------------------------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------------------
# Hugging Face model
# --------------------------------------------------

MODEL_ID = "drasti02/ai-text-detector-roberta"

print("Loading RoBERTa model from Hugging Face Hub...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_ID
)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_ID
)

model.eval()

print("RoBERTa model loaded successfully.")


# --------------------------------------------------
# Request schema
# --------------------------------------------------

class TextRequest(BaseModel):
    text: str = Field(
        ...,
        min_length=1,
        description="Text to classify as human-written or AI-generated."
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
# Health endpoint
# --------------------------------------------------

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model": "RoBERTa",
        "model_source": "Hugging Face Hub"
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