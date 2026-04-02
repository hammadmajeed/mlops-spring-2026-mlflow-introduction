import mlflow
import mlflow.sklearn
import argparse
import warnings

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve

# -------------------------------
# Optional: Clean output for class
# -------------------------------
warnings.filterwarnings("ignore", category=FutureWarning)

# -------------------------------
# MLflow Tracking (Server-based)
# -------------------------------
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("breast_cancer_experiments")

# 🔥 Enable autologging
#mlflow.autolog()

# -------------------------------
# Argument parser
# -------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--C", type=float, default=1.0, help="Regularization strength")
parser.add_argument("--l1_ratio", type=float, default=0.0, help="ElasticNet mixing (0=L2, 1=L1)")
args = parser.parse_args()

# -------------------------------
# Start MLflow run
# -------------------------------
with mlflow.start_run():

    # Load dataset
    data = load_breast_cancer()
    X = data.data
    y = data.target

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Adaptive max_iter for convergence
    max_iter = 8000 if args.l1_ratio > 0.5 else 3000

    # Pipeline: Scaling + Logistic Regression
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            C=args.C,
            penalty="elasticnet",
            l1_ratio=args.l1_ratio,
            solver="saga",
            max_iter=max_iter,
            tol=1e-4
        ))
    ])

    # Train
    model.fit(X_train, y_train)

    # Predictions
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]

    # Metrics
    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds)
    roc = roc_auc_score(y_test, probs)

    # -------------------------------
    # MLflow Logging
    # -------------------------------
    
    mlflow.log_param("C", args.C)
    mlflow.log_param("l1_ratio", args.l1_ratio)
    mlflow.log_param("max_iter", max_iter)

    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("f1_score", f1)
    mlflow.log_metric("roc_auc", roc)

    # Log model (new API)
    mlflow.sklearn.log_model(model, name="model")

    # Output
    print(f"Accuracy: {acc:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"ROC-AUC: {roc:.4f}")


    fpr, tpr, _ = roc_curve(y_test, probs)

    plt.plot(fpr, tpr)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.savefig("roc_curve.png")

    mlflow.log_artifact("roc_curve.png")