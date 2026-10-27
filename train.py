
import os
import mlflow
import mlflow.sklearn
import pandas as pd

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)

# Use a local SQLite database for MLflow tracking
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("Iris_ML_Experiment")

# Load dataset
iris = load_iris(as_frame=True)
X = iris.data
y = iris.target

# Save a copy of the original dataset
os.makedirs("data", exist_ok=True)
X.assign(target=y).to_csv(
    "data/iris.csv", index=False
)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Model parameters
n_estimators = 100
max_depth = 5

with mlflow.start_run(run_name="RandomForest_Iris"):

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42
    )

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    # Calculate metrics
    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(
        y_test, predictions, average="weighted",
        zero_division=0
    )
    recall = recall_score(
        y_test, predictions, average="weighted",
        zero_division=0
    )
    f1 = f1_score(
        y_test, predictions, average="weighted",
        zero_division=0
    )

    # Log parameters
    mlflow.log_params({
        "model_type": "RandomForestClassifier",
        "n_estimators": n_estimators,
        "max_depth": max_depth,
        "test_size": 0.2,
        "random_state": 42
    })

    # Log evaluation metrics
    mlflow.log_metrics({
        "accuracy": accuracy,
        "precision_weighted": precision,
        "recall_weighted": recall,
        "f1_weighted": f1
    })

    # Log model artifact
    mlflow.sklearn.log_model(
    sk_model=model,
    artifact_path="model",
    skops_trusted_types=["sklearn.tree._tree.Tree"]
    )


    # Save classification report as an artifact
    report = classification_report(
        y_test,
        predictions,
        target_names=iris.target_names,
        output_dict=True,
        zero_division=0
    )

    pd.DataFrame(report).transpose().to_csv(
        "classification_report.csv"
    )

    mlflow.log_artifact("classification_report.csv")

    print("Experiment: Iris_ML_Experiment")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-score: {f1:.4f}")
    print("Parameters, metrics and model logged to MLflow.")
