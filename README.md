# 🎯 **Problem: Regularized Logistic Regression for Binary Classification (with Hyperparameter Optimization)**

* Dataset: **Breast Cancer Classification (sklearn)**
* Model: Logistic Regression with **L1/L2 regularization**
* Focus:

  * Effect of regularization (C, penalty)
  * Solver behavior
  * Overfitting vs generalization
  * Metric trade-offs (Accuracy, F1, ROC-AUC)

This makes MLflow **actually useful**, not just logging numbers.

---

# 🧪 ML Experiment Tracking with MLflow (Advanced Tutorial)

## 🚀 Overview

In this lab, we will solve a **binary classification problem** and track experiments using MLflow.

We will:

* Train Logistic Regression models
* Tune regularization parameters
* Compare multiple runs
* Analyze trade-offs using MLflow UI

---

## 🧠 Learning Objectives

By the end of this tutorial, you will:

* Track hyperparameter optimization experiments
* Analyze model performance across runs
* Understand impact of regularization
* Log and compare multiple metrics

---

## 📁 Project Structure

```bash
mlflow-advanced/
│
├── src/
│   ├── train.py
├── data/
│   ├── dataset.csv
├── mlflow.db
├── requirements.txt
└── README.md
```

---

## ⚙️ Setup

```bash
git clone https://github.com/your-username/mlflow-advanced.git
cd mlflow-advanced

python3 -m venv venv
source venv/bin/activate

pip install -r requirements.txt
```

---

## 📦 Requirements

```txt
mlflow
scikit-learn
pandas
numpy
matplotlib
```

---

## 📊 Dataset

We use:

* `sklearn.datasets.load_breast_cancer()`

Why?

* Real-world dataset
* Non-trivial classification
* Sensitive to regularization

---

## 🏗️ Step 1: Training Script with MLflow

Create `src/train.py`:

```python
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

# -------------------------------
# Optional: Clean output for class
# -------------------------------
warnings.filterwarnings("ignore", category=FutureWarning)

# -------------------------------
# MLflow Tracking (Server-based)
# -------------------------------
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("breast_cancer_experiments")

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
```

---

## ▶️ Step 2: Run Experiments

Run multiple configurations:

```bash
python src/train.py --C 0.01 --penalty l2
python src/train.py --C 0.1 --penalty l2
python src/train.py --C 1 --penalty l2
python src/train.py --C 10 --penalty l2
```

---

## 🖥️ Step 3: Launch MLflow UI

```bash
mlflow ui
```

Open:

```
http://127.0.0.1:5000
```

---

## 🔍 What to Analyze in MLflow

Students should observe:

* Effect of **C (regularization strength)**
* Effect of **L1_RATIO**
* Trade-offs:

  * Accuracy vs F1 vs ROC-AUC

👉 Key insight:

> Best accuracy ≠ best model

---

## 📈 Step 4: Log ROC Curve

Add:

```python
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve

fpr, tpr, _ = roc_curve(y_test, probs)

plt.plot(fpr, tpr)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.savefig("roc_curve.png")

mlflow.log_artifact("roc_curve.png")
```

---

## 🔁 Step 5: Hyperparameter Sweep (Manual)

Modify script:

```python
for C in [0.01, 0.1, 1, 10]:
    with mlflow.start_run():
        model = LogisticRegression(C=C, max_iter=500)
        ...
```

---

## 🧪 Exercises

### 🟢 Basic

1. Run experiments with at least 5 different values of `C`
2. Identify the best model based on:

   * Accuracy
   * ROC-AUC

---

### 🟡 Intermediate

3. Compare L1 vs L2 regularization
4. Which metric is most stable across runs? Why?
5. Plot and log ROC curves for at least 3 runs

---

### 🔴 Advanced

6. Perform grid search manually using loops and log each run
7. Add another model (e.g., SVM) and compare with Logistic Regression
8. Automatically identify and print best run using MLflow API

---

## 💡 Conceptual Questions

1. What is the role of regularization parameter **C**?
2. Why does L1 produce sparse models?
3. Why is ROC-AUC important in imbalanced datasets?
4. Why is experiment tracking critical in hyperparameter tuning?
5. How would MLflow integrate into a CI/CD pipeline?

---

## 🎯 Key Takeaways

* ML experiments are **not single runs**
* Tracking is essential for:

  * reproducibility
  * comparison
  * optimization

👉 MLflow becomes critical when:

* experiments scale
* models become complex
* teams collaborate

