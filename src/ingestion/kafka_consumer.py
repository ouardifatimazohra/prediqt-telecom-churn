import json
import psycopg2
import pandas as pd
from kafka import KafkaConsumer
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib
import logging
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DB_CONFIG = {
    "host": "localhost",
    "port": 5432,
    "dbname": "prediqt",
    "user": "prediqt",
    "password": "prediqt123"
}

MODEL_PATH = "data/models/xgboost_churn_model.pkl"


def create_db_table(conn):
    """Create predictions table if not exists."""
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS churn_predictions (
            id SERIAL PRIMARY KEY,
            customer_id VARCHAR(50),
            churn_prediction INTEGER,
            churn_probability FLOAT,
            risk_level VARCHAR(10),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    cursor.close()
    logger.info("✅ Table churn_predictions ready")


def preprocess_customer(customer: dict) -> pd.DataFrame:
    """Preprocess a single customer record."""
    df = pd.DataFrame([customer])

    # Drop customerID
    customer_id = df.get('customerID', ['unknown'])[0] if 'customerID' in df else 'unknown'
    df.drop(columns=['customerID'], errors='ignore', inplace=True)
    df.drop(columns=['Churn'], errors='ignore', inplace=True)

    # Fix TotalCharges
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'].fillna(0, inplace=True)

    # Encode
    binary_cols = ['gender', 'Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']
    for col in binary_cols:
        if col in df.columns:
            df[col] = df[col].map({'Yes': 1, 'No': 0, 'Male': 1, 'Female': 0})

    cat_cols = df.select_dtypes(include='object').columns
    le = LabelEncoder()
    for col in cat_cols:
        df[col] = le.fit_transform(df[col].astype(str))

    return df, customer_id


def consume_and_predict():
    """Consume Kafka events and make predictions."""
    # Load model
    model = joblib.load(MODEL_PATH)
    logger.info("✅ Model loaded")

    # Connect to PostgreSQL
    conn = psycopg2.connect(**DB_CONFIG)
    create_db_table(conn)

    # Create consumer
    consumer = KafkaConsumer(
        'churn-events',
        bootstrap_servers=['localhost:9092'],
        value_deserializer=lambda m: json.loads(m.decode('utf-8')),
        auto_offset_reset='earliest',
        group_id='prediqt-consumer'
    )

    logger.info("🎧 Listening to churn-events topic...")

    for message in consumer:
        customer = message.value

        try:
            df, customer_id = preprocess_customer(customer)
            prediction = int(model.predict(df)[0])
            probability = float(model.predict_proba(df)[0][1])
            risk = "high" if probability > 0.6 else "medium" if probability > 0.3 else "low"

            # Save to PostgreSQL
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO churn_predictions
                (customer_id, churn_prediction, churn_probability, risk_level)
                VALUES (%s, %s, %s, %s)
            """, (customer_id, prediction, probability, risk))
            conn.commit()
            cursor.close()

            logger.info(f"✅ {customer_id} → churn={prediction} | prob={probability:.2f} | risk={risk}")

        except Exception as e:
            logger.error(f"❌ Error processing message: {e}")


if __name__ == "__main__":
    consume_and_predict()