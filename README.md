# MLOps Assignment 2 — End-to-End ML Pipeline
### Titanic Survival Prediction using Apache Airflow + MLflow

**Student:** Maryam Khalid  
**ID:** i221917  
**Course:** MLOps — BS Data Science  

---

## Overview

This project implements an end-to-end Machine Learning pipeline that:
- Uses **Apache Airflow** to orchestrate a DAG with 11 tasks
- Uses **MLflow** to track experiments, log metrics, and register models
- Predicts **Titanic survival** using RandomForest and LogisticRegression
- Demonstrates **parallel processing**, **retry logic**, and **branching**

---

## Project Structure

```
├── mlops_airflow_mlflow_pipeline.py   # Main Airflow DAG file
├── titanic.csv                        # Titanic dataset
├── requirements.txt                   # Python dependencies
├── docker-compose.yml                 # Docker setup for Airflow + MLflow
└── README.md
```

---

## Requirements

- Docker Desktop (running)
- Python 3.8+
- 8GB RAM minimum

---

## Setup Instructions

### Step 1 — Clone the repository
```bash
git clone https://github.com/maryamss-hub/mlops-assignment2
cd mlops-assignment2
```

### Step 2 — Create required folders
```bash
mkdir -p logs plugins data
```

### Step 3 — Add the dataset
Place `titanic.csv` inside the `data/` folder.

### Step 4 — Set environment variable (Windows)
Create a `.env` file in the project root:
```
AIRFLOW_UID=50000
```

### Step 5 — Initialize Airflow
```bash
docker compose up airflow-init
```

### Step 6 — Start all services
```bash
docker compose up -d
```

### Step 7 — Access the UIs
| Service | URL | Login |
|---------|-----|-------|
| Airflow | http://localhost:8080 | admin / admin |
| MLflow  | http://127.0.0.1:5000 | — |

---

## Running the Pipeline

1. Open Airflow at http://localhost:8080
2. Find the DAG: `mlops_airflow_mlflow_pipeline`
3. Toggle it **ON**
4. Click **Trigger DAG** (▶ button)
5. Watch tasks execute in the Graph view

To run with different hyperparameters, edit these lines in the DAG file:
```python
MODEL_TYPE = "RandomForest"       # or "LogisticRegression"
HYPERPARAMS = {
    "n_estimators": 100,          # change this
    "max_depth": 5,               # change this
    "C": 1.0,
    "max_iter": 200,
}
ACCURACY_THRESHOLD = 0.80
```

---

## DAG Pipeline Flow

```
data_ingestion
      ↓
data_validation (retry on failure)
      ↓
handle_missing_values ──┐
                        ├── (parallel)
feature_engineering  ───┘
      ↓
data_encoding
      ↓
model_training  →  MLflow run starts
      ↓
model_evaluation  →  logs metrics
      ↓
branching_logic
    ↙         ↘
register_model  reject_model
    ↘         ↙
    pipeline_end
```

---

## Experiment Results

| Run | Model | n_estimators | max_depth | Accuracy | Decision |
|-----|-------|-------------|-----------|----------|----------|
| 1 | RandomForest | 100 | 5 | 0.799 | REJECT |
| 2 | RandomForest | 200 | 10 | 0.799 | REJECT |
| 3 | LogisticRegression | — | — | 0.739 | REJECT |
| 4 | RandomForest | 200 | 10 | **0.832** | **REGISTER** ✅ |

**Best Model:** RandomForest with n_estimators=200, max_depth=10 — Accuracy: 83.2%

---

## Stopping the Services

```bash
docker compose down
```

To also remove volumes:
```bash
docker compose down -v
```
