from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import pandas as pd
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="PREDIQT API",
    description="Telecom Churn Prediction API — MLOps Platform",
    version="1.0.0"
)

# Load model at startup
MODEL_PATH = "data/models/xgboost_churn_model.pkl"
model = None


@app.on_event("startup")
def load_model():
    global model
    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
        logger.info("✅ Model loaded successfully")
    else:
        logger.error("❌ Model file not found")


class CustomerFeatures(BaseModel):
    """Input schema matching the 19 preprocessed features."""
    gender: int = Field(..., ge=0, le=1, example=0)
    SeniorCitizen: int = Field(..., ge=0, le=1, example=0)
    Partner: int = Field(..., ge=0, le=1, example=1)
    Dependents: int = Field(..., ge=0, le=1, example=0)
    tenure: int = Field(..., example=12)
    PhoneService: int = Field(..., ge=0, le=1, example=1)
    MultipleLines: int = Field(..., example=0)
    InternetService: int = Field(..., example=0)
    OnlineSecurity: int = Field(..., example=0)
    OnlineBackup: int = Field(..., example=2)
    DeviceProtection: int = Field(..., example=0)
    TechSupport: int = Field(..., example=0)
    StreamingTV: int = Field(..., example=0)
    StreamingMovies: int = Field(..., example=0)
    Contract: int = Field(..., example=0)
    PaperlessBilling: int = Field(..., ge=0, le=1, example=1)
    PaymentMethod: int = Field(..., example=2)
    MonthlyCharges: float = Field(..., example=65.5)
    TotalCharges: float = Field(..., example=786.0)


class PredictionResponse(BaseModel):
    churn_prediction: int
    churn_probability: float
    risk_level: str


@app.get("/")
def root():
    return {
        "service": "PREDIQT — Telecom Churn Prediction API",
        "status": "running",
        "version": "1.0.0"
    }


@app.get("/health")
def health_check():
    return {
        "status": "healthy" if model is not None else "model not loaded",
        "model_loaded": model is not None
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(customer: CustomerFeatures):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        # Convert to DataFrame
        data = pd.DataFrame([customer.dict()])

        # Predict
        prediction = int(model.predict(data)[0])
        probability = float(model.predict_proba(data)[0][1])

        # Risk level
        if probability < 0.3:
            risk = "low"
        elif probability < 0.6:
            risk = "medium"
        else:
            risk = "high"

        logger.info(f"Prediction made: churn={prediction}, prob={probability:.2f}")

        return PredictionResponse(
            churn_prediction=prediction,
            churn_probability=round(probability, 4),
            risk_level=risk
        )

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)