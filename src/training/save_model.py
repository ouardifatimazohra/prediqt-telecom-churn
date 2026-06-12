import joblib
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from src.training.train import train_model

if __name__ == "__main__":
    model, metrics = train_model()

    os.makedirs("data/models", exist_ok=True)
    joblib.dump(model, "data/models/xgboost_churn_model.pkl")

    print("\n✅ Modèle sauvegardé : data/models/xgboost_churn_model.pkl")