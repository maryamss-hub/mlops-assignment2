import os
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.empty import EmptyOperator
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder

MODEL_TYPE = "LogisticRegression"
HYPERPARAMS = {
    "n_estimators": 200,
    "max_depth": 10,
    "C": 0.5,
    "max_iter": 500,
}

ACCURACY_THRESHOLD = 0.80
DATA_PATH = "/opt/airflow/data/titanic.csv"
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
# Use /tmp for artifacts — Airflow container always has write access here
ARTIFACT_ROOT = "file:///tmp/mlruns"
EXPERIMENT_NAME = "Titanic_Survival_Prediction"

default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(seconds=10),
}

def data_ingestion(**context):
    logging.info("=== TASK 2: Data Ingestion ===")
    df = pd.read_csv(DATA_PATH)
    print(f"Dataset Shape: {df.shape}")
    logging.info(f"Dataset shape: {df.shape}")
    missing = df.isnull().sum()
    logging.info(f"Missing values:\n{missing}")
    print(f"\nMissing values:\n{missing}")
    context["ti"].xcom_push(key="data_path", value=DATA_PATH)

def data_validation(**context):
    logging.info("=== TASK 3: Data Validation ===")
    try_number = context["ti"].try_number
    logging.info(f"Attempt number: {try_number}")
    if try_number == 1:
        logging.warning("Intentional failure on attempt 1 - retry demonstration!")
        raise ValueError("Simulated failure on first attempt (retry demonstration)")
    data_path = context["ti"].xcom_pull(key="data_path", task_ids="data_ingestion")
    df = pd.read_csv(data_path)
    age_pct = df["Age"].isnull().mean() * 100
    embarked_pct = df["Embarked"].isnull().mean() * 100
    logging.info(f"Age missing: {age_pct:.2f}%")
    logging.info(f"Embarked missing: {embarked_pct:.2f}%")
    print(f"Age missing %: {age_pct:.2f}%")
    print(f"Embarked missing %: {embarked_pct:.2f}%")
    if age_pct > 30:
        raise ValueError(f"Age missing {age_pct:.2f}% exceeds 30% threshold!")
    if embarked_pct > 30:
        raise ValueError(f"Embarked missing {embarked_pct:.2f}% exceeds 30% threshold!")
    logging.info("Validation passed")

def handle_missing_values(**context):
    logging.info("=== TASK 4a: Handle Missing Values ===")
    data_path = context["ti"].xcom_pull(key="data_path", task_ids="data_ingestion")
    df = pd.read_csv(data_path)
    df["Age"].fillna(df["Age"].median(), inplace=True)
    df["Embarked"].fillna(df["Embarked"].mode()[0], inplace=True)
    out_path = "/opt/airflow/data/titanic_missing_handled.csv"
    df.to_csv(out_path, index=False)
    context["ti"].xcom_push(key="missing_handled_path", value=out_path)
    print(f"Age nulls remaining: {df['Age'].isnull().sum()}")
    print(f"Embarked nulls remaining: {df['Embarked'].isnull().sum()}")

def feature_engineering(**context):
    logging.info("=== TASK 4b: Feature Engineering ===")
    data_path = context["ti"].xcom_pull(key="data_path", task_ids="data_ingestion")
    df = pd.read_csv(data_path)
    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
    df["IsAlone"] = (df["FamilySize"] == 1).astype(int)
    out_path = "/opt/airflow/data/titanic_features.csv"
    df.to_csv(out_path, index=False)
    context["ti"].xcom_push(key="features_path", value=out_path)
    print(f"FamilySize sample:\n{df['FamilySize'].value_counts().head()}")
    print(f"IsAlone:\n{df['IsAlone'].value_counts()}")

def data_encoding(**context):
    logging.info("=== TASK 5: Data Encoding ===")
    missing_path = context["ti"].xcom_pull(key="missing_handled_path", task_ids="handle_missing_values")
    features_path = context["ti"].xcom_pull(key="features_path", task_ids="feature_engineering")
    df = pd.read_csv(missing_path)
    df_feat = pd.read_csv(features_path)
    df["FamilySize"] = df_feat["FamilySize"]
    df["IsAlone"] = df_feat["IsAlone"]
    le = LabelEncoder()
    df["Sex"] = le.fit_transform(df["Sex"])
    df["Embarked"] = le.fit_transform(df["Embarked"])
    drop_cols = ["PassengerId", "Name", "Ticket", "Cabin"]
    df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)
    out_path = "/opt/airflow/data/titanic_encoded.csv"
    df.to_csv(out_path, index=False)
    context["ti"].xcom_push(key="encoded_path", value=out_path)
    print(f"Final columns: {list(df.columns)}")
    print(f"Shape: {df.shape}")

def model_training(**context):
    logging.info("=== TASK 6: Model Training ===")
    encoded_path = context["ti"].xcom_pull(key="encoded_path", task_ids="data_encoding")
    df = pd.read_csv(encoded_path)
    X = df.drop("Survived", axis=1)
    y = df["Survived"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Set tracking URI to MLflow server
    # Use /tmp as artifact location — container always has write access
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    # Create or get experiment with /tmp artifact root
    from mlflow.tracking import MlflowClient
    client = MlflowClient()
    exp = client.get_experiment_by_name(EXPERIMENT_NAME)
    if exp is None:
        exp_id = client.create_experiment(
            EXPERIMENT_NAME,
            artifact_location=ARTIFACT_ROOT
        )
    else:
        exp_id = exp.experiment_id

    with mlflow.start_run(experiment_id=exp_id) as run:
        run_id = run.info.run_id
        mlflow.log_param("model_type", MODEL_TYPE)
        mlflow.log_param("dataset_size", len(df))
        mlflow.log_param("train_size", len(X_train))
        mlflow.log_param("test_size", len(X_test))

        if MODEL_TYPE == "RandomForest":
            mlflow.log_param("n_estimators", HYPERPARAMS["n_estimators"])
            mlflow.log_param("max_depth", HYPERPARAMS["max_depth"])
            model = RandomForestClassifier(
                n_estimators=HYPERPARAMS["n_estimators"],
                max_depth=HYPERPARAMS["max_depth"],
                random_state=42,
            )
        else:
            mlflow.log_param("C", HYPERPARAMS["C"])
            mlflow.log_param("max_iter", HYPERPARAMS["max_iter"])
            model = LogisticRegression(
                C=HYPERPARAMS["C"],
                max_iter=HYPERPARAMS["max_iter"],
                random_state=42,
            )

        model.fit(X_train, y_train)

        # Log model — artifacts go to /tmp/mlruns (writable)
        mlflow.sklearn.log_model(model, "model")

        X_test.to_csv("/opt/airflow/data/X_test.csv", index=False)
        y_test.to_csv("/opt/airflow/data/y_test.csv", index=False)
        context["ti"].xcom_push(key="run_id", value=run_id)
        context["ti"].xcom_push(key="X_test_path", value="/opt/airflow/data/X_test.csv")
        context["ti"].xcom_push(key="y_test_path", value="/opt/airflow/data/y_test.csv")
        print(f"Training complete. Run ID: {run_id}")

def model_evaluation(**context):
    logging.info("=== TASK 7: Model Evaluation ===")
    run_id = context["ti"].xcom_pull(key="run_id", task_ids="model_training")
    X_test = pd.read_csv(context["ti"].xcom_pull(key="X_test_path", task_ids="model_training"))
    y_test = pd.read_csv(context["ti"].xcom_pull(key="y_test_path", task_ids="model_training"))
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    with mlflow.start_run(run_id=run_id):
        model = mlflow.sklearn.load_model(f"runs:/{run_id}/model")
        y_pred = model.predict(X_test)
        acc  = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec  = recall_score(y_test, y_pred, zero_division=0)
        f1   = f1_score(y_test, y_pred, zero_division=0)
        mlflow.log_metric("accuracy",  acc)
        mlflow.log_metric("precision", prec)
        mlflow.log_metric("recall",    rec)
        mlflow.log_metric("f1_score",  f1)
        print(f"\nAccuracy: {acc:.4f}\nPrecision: {prec:.4f}\nRecall: {rec:.4f}\nF1: {f1:.4f}")
    context["ti"].xcom_push(key="accuracy", value=acc)

def branching_logic(**context):
    logging.info("=== TASK 8: Branching Logic ===")
    accuracy = context["ti"].xcom_pull(key="accuracy", task_ids="model_evaluation")
    logging.info(f"Accuracy: {accuracy:.4f} | Threshold: {ACCURACY_THRESHOLD}")
    if accuracy >= ACCURACY_THRESHOLD:
        logging.info("Decision: REGISTER model")
        return "register_model"
    else:
        logging.info("Decision: REJECT model")
        return "reject_model"

def register_model(**context):
    logging.info("=== TASK 9a: Model Registration ===")
    run_id   = context["ti"].xcom_pull(key="run_id",   task_ids="model_training")
    accuracy = context["ti"].xcom_pull(key="accuracy", task_ids="model_evaluation")
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    result = mlflow.register_model(f"runs:/{run_id}/model", f"Titanic_{MODEL_TYPE}")
    logging.info(f"Registered: Titanic_{MODEL_TYPE} v{result.version} | Accuracy: {accuracy:.4f}")
    print(f"Model registered as version {result.version} with accuracy {accuracy:.4f}")

def reject_model(**context):
    logging.info("=== TASK 9b: Model Rejected ===")
    accuracy = context["ti"].xcom_pull(key="accuracy", task_ids="model_evaluation")
    run_id   = context["ti"].xcom_pull(key="run_id",   task_ids="model_training")
    reason   = f"Accuracy {accuracy:.4f} below threshold {ACCURACY_THRESHOLD}"
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    with mlflow.start_run(run_id=run_id):
        mlflow.log_param("rejection_reason", reason)
        mlflow.set_tag("status", "REJECTED")
    logging.warning(f"Model REJECTED: {reason}")
    print(f"Model rejected. Reason: {reason}")

with DAG(
    dag_id="mlops_airflow_mlflow_pipeline",
    default_args=default_args,
    description="End-to-end ML pipeline: Titanic survival prediction",
    schedule_interval=None,
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["mlops", "titanic", "mlflow"],
) as dag:

    t_ingest   = PythonOperator(task_id="data_ingestion",        python_callable=data_ingestion)
    t_validate = PythonOperator(task_id="data_validation",       python_callable=data_validation, retries=1, retry_delay=timedelta(seconds=5))
    t_missing  = PythonOperator(task_id="handle_missing_values", python_callable=handle_missing_values)
    t_features = PythonOperator(task_id="feature_engineering",   python_callable=feature_engineering)
    t_encode   = PythonOperator(task_id="data_encoding",         python_callable=data_encoding)
    t_train    = PythonOperator(task_id="model_training",        python_callable=model_training)
    t_evaluate = PythonOperator(task_id="model_evaluation",      python_callable=model_evaluation)
    t_branch   = BranchPythonOperator(task_id="branching_logic", python_callable=branching_logic)
    t_register = PythonOperator(task_id="register_model",        python_callable=register_model)
    t_reject   = PythonOperator(task_id="reject_model",          python_callable=reject_model)
    t_end      = EmptyOperator(task_id="pipeline_end", trigger_rule="none_failed_min_one_success")

    t_ingest >> t_validate >> [t_missing, t_features]
    [t_missing, t_features] >> t_encode
    t_encode >> t_train >> t_evaluate >> t_branch
    t_branch >> [t_register, t_reject]
    [t_register, t_reject] >> t_end