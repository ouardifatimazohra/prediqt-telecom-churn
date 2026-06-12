# PREDIQT — Telecom Churn Prediction Platform

![Python](https://img.shields.io/badge/Python-3.14-blue?style=flat-square&logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?style=flat-square&logo=fastapi)
![MLflow](https://img.shields.io/badge/MLflow-2.x-0194E2?style=flat-square&logo=mlflow)
![Docker](https://img.shields.io/badge/Docker-Containerized-2496ED?style=flat-square&logo=docker)
![Kafka](https://img.shields.io/badge/Kafka-Streaming-231F20?style=flat-square&logo=apachekafka)
![Airflow](https://img.shields.io/badge/Airflow-Orchestration-017CEE?style=flat-square&logo=apacheairflow)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

> End-to-end MLOps platform for real-time telecom customer churn prediction.  
> Built with production-grade architecture : streaming ingestion, automated training pipeline, REST API serving, and live monitoring.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        PREDIQT Platform                         │
│                                                                 │
│  ┌──────────┐     ┌──────────┐    ┌──────────┐     ┌──────────┐ │
│  │  Kafka   │───▶│ Airflow  │───▶│ XGBoost  │───▶│  MLflow  │ │
│  │ Streaming│     │ Pipeline │    │  Model   │     │ Registry │ │
│  └──────────┘     └──────────┘    └──────────┘     └──────────┘ │
│                                        │                        │
│                                        ▼                        │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐                   │
│  │ Grafana  │◀───│PostgreSQL│◀───│ FastAPI  │◀── HTTP Request │
│  │Dashboard │    │ Storage  │    │ Serving  │                   │
│  └──────────┘    └──────────┘    └──────────┘                   │
└─────────────────────────────────────────────────────────────────┘
```

## Tech Stack

| Layer | Technology | Role |
|-------|-----------|------|
| Ingestion | Apache Kafka | Real-time event streaming |
| Orchestration | Apache Airflow | Pipeline automation |
| Feature Engineering | Pandas · Scikit-learn | Data preprocessing |
| ML Model | XGBoost | Churn prediction |
| Experiment Tracking | MLflow | Model versioning & registry |
| Serving | FastAPI | REST API endpoint |
| Storage | PostgreSQL | Predictions & features store |
| Monitoring | Grafana · Prometheus | Live metrics dashboard |
| Containerization | Docker · Docker Compose | Full stack deployment |

## Project Structure

```
prediqt-telecom-churn/
├── src/
│   ├── ingestion/        # Kafka consumer & producer
│   ├── features/         # Feature engineering pipeline
│   ├── training/         # XGBoost training & MLflow tracking
│   ├── serving/          # FastAPI prediction API
│   └── monitoring/       # Drift detection & metrics
├── tests/                # Unit & integration tests
├── data/
│   ├── raw/              # Raw telecom dataset
│   ├── processed/        # Engineered features
│   └── models/           # Serialized models
├── docs/                 # Architecture diagrams
├── .github/workflows/    # CI/CD pipelines
├── docker-compose.yml    # Full stack deployment
├── Makefile              # Dev shortcuts
└── requirements.txt      # Dependencies
```

## Quick Start

```bash
# Clone the repository
git clone https://github.com/ouardifatimazohra/prediqt-telecom-churn.git
cd prediqt-telecom-churn

# Start the full stack
docker-compose up -d

# Check API health
curl http://localhost:8000/health

# Make a prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"tenure": 12, "monthly_charges": 65.5, "contract": "Month-to-month"}'
```

## API Endpoints

| Method | Endpoint | Description                 |
|--------|----------|-----------------------------|
| GET    | `/`      | API status                  |
| GET    | `/health`| Health check + model status |
| POST   |`/predict`| Real-time churn prediction  |

**Example response:**
```json
{
  "churn_prediction": 0,
  "churn_probability": 0.0644,
  "risk_level": "low"
}
```

Interactive docs available at `/docs` (Swagger UI).

## Dataset

- **Source :** [IBM Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- **Size :** 7,043 customers · 21 features
- **Target :** Binary churn prediction (Yes / No)
- **Domain :** Moroccan telecom market adaptation


## Model Performance

| Metric    | Score  |
|-----------|--------|
| Accuracy  | 79.49% |
| F1-Score  | 58.89% |
| ROC-AUC   | 83.06% |
| Precision | 62.92% |

*Baseline XGBoost model — tracked with MLflow.*

## Roadmap

- [x] Project structure & architecture design
- [ ] Kafka ingestion pipeline
- [x] Feature engineering module
- [x] XGBoost training with MLflow tracking
- [x] FastAPI serving endpoint
- [ ] Docker containerization
- [ ] CI/CD with GitHub Actions
- [ ] Grafana monitoring dashboard

## Author

**Fatimazohra Ouardi**  
Data & MLOps Engineering Student — Université Privée de Fès  
[GitHub](https://github.com/ouardifatimazohra) · [LinkedIn](https://www.linkedin.com/in/ouardi-fatima-zohra-34857931b/)

---
*Built as part of an MLOps learning journey — 2025/2026*