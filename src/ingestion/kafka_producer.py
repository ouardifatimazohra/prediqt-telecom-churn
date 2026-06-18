import json
import time
import random
import pandas as pd
from kafka import KafkaProducer
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_producer():
    """Create Kafka producer."""
    return KafkaProducer(
        bootstrap_servers=['localhost:9092'],
        value_serializer=lambda v: json.dumps(v).encode('utf-8')
    )


def load_sample_customers(filepath: str) -> list:
    """Load customers from dataset."""
    df = pd.read_csv(filepath)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)
    return df.to_dict(orient='records')


def stream_customers(filepath: str, interval: float = 1.0):
    """Stream customer events to Kafka topic."""
    producer = create_producer()
    customers = load_sample_customers(filepath)

    logger.info(f"Starting stream — {len(customers)} customers loaded")
    logger.info("Sending events to topic 'churn-events'...")

    count = 0
    while True:
        customer = random.choice(customers)
        producer.send('churn-events', value=customer)
        count += 1
        logger.info(f"Event #{count} sent — customerID: {customer.get('customerID', 'N/A')}")
        time.sleep(interval)


if __name__ == "__main__":
    stream_customers(
        filepath="data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv",
        interval=1.0
    )