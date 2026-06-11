import mlflow
import mlflow.xgboost
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, f1_score,
                             roc_auc_score, precision_score)
import pandas as pd
import logging
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from src.features.preprocessing import run_preprocessing

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def train_model(filepath: str = "data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv"):
    """Train XGBoost model with MLflow tracking."""

    # Preprocessing
    logger.info("Starting preprocessing...")
    X, y = run_preprocessing(filepath)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    logger.info(f"Train size: {X_train.shape[0]} | Test size: {X_test.shape[0]}")

    # MLflow experiment
    mlflow.set_experiment("prediqt-churn-prediction")

    with mlflow.start_run(run_name="xgboost-baseline"):

        # Hyperparameters
        params = {
            "n_estimators": 200,
            "max_depth": 6,
            "learning_rate": 0.1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": 42,
            "eval_metric": "logloss"
        }

        # Log hyperparameters
        mlflow.log_params(params)

        # Train
        logger.info("Training XGBoost model...")
        model = xgb.XGBClassifier(**params)
        model.fit(X_train, y_train)

        # Evaluate
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        metrics = {
            "accuracy": round(accuracy_score(y_test, y_pred), 4),
            "f1_score": round(f1_score(y_test, y_pred), 4),
            "roc_auc": round(roc_auc_score(y_test, y_prob), 4),
            "precision": round(precision_score(y_test, y_pred), 4)
        }

        # Log metrics
        mlflow.log_metrics(metrics)

        # Log model
        mlflow.xgboost.log_model(model, "xgboost-churn-model")

        # Print results
        print("\n" + "="*45)
        print("   PREDIQT — Model Training Results")
        print("="*45)
        for metric, value in metrics.items():
            print(f"  ✅ {metric:<12} : {value}")
        print("="*45)
        print(f"\n📊 MLflow Run ID : {mlflow.active_run().info.run_id}")
        print("🚀 Model logged to MLflow registry\n")

        return model, metrics


if __name__ == "__main__":
    train_model()