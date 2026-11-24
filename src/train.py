from pathlib import Path
import json
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, confusion_matrix
)
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from src.data_prep import load_and_clean

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"
MODEL_DIR.mkdir(exist_ok=True)
REPORT_DIR.mkdir(exist_ok=True)

def main():
    df = load_and_clean()
    X = df.drop(columns=["Churn", "customerID"])
    y = df["Churn"]

    categorical = X.select_dtypes(include=["object"]).columns.tolist()
    numerical = X.select_dtypes(exclude=["object"]).columns.tolist()

    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])
    prep = ColumnTransformer([
        ("numeric", numeric_pipe, numerical),
        ("categorical", categorical_pipe, categorical)
    ])

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1500, class_weight="balanced"),
        "Decision Tree": DecisionTreeClassifier(max_depth=7, min_samples_leaf=12, class_weight="balanced", random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=250, min_samples_leaf=3, class_weight="balanced", n_jobs=-1, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42),
        "Hist Gradient Boosting": HistGradientBoostingClassifier(max_iter=150, learning_rate=0.08, random_state=42)
    }

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    rows, fitted = [], {}
    for name, estimator in models.items():
        pipe = Pipeline([("preprocess", prep), ("model", estimator)])
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        proba = pipe.predict_proba(X_test)[:, 1] if hasattr(pipe, "predict_proba") else None
        metrics = {
            "model": name,
            "accuracy": accuracy_score(y_test, pred),
            "precision": precision_score(y_test, pred, zero_division=0),
            "recall": recall_score(y_test, pred, zero_division=0),
            "f1": f1_score(y_test, pred, zero_division=0),
            "roc_auc": roc_auc_score(y_test, proba) if proba is not None else None
        }
        rows.append(metrics)
        fitted[name] = pipe
        print(f"{name}: accuracy={metrics['accuracy']:.4f}, f1={metrics['f1']:.4f}")

    results = pd.DataFrame(rows).sort_values("f1", ascending=False)
    results.to_csv(REPORT_DIR / "model_comparison.csv", index=False)
    best_name = results.iloc[0]["model"]
    best_model = fitted[best_name]
    joblib.dump(best_model, MODEL_DIR / "best_churn_model.joblib")

    best_pred = best_model.predict(X_test)
    best_proba = best_model.predict_proba(X_test)[:, 1]
    with open(REPORT_DIR / "best_model_metrics.json", "w", encoding="utf-8") as f:
        json.dump({
            "selected_model": best_name,
            "test_rows": int(len(y_test)),
            "train_rows": int(len(y_train)),
            "accuracy": float(accuracy_score(y_test, best_pred)),
            "precision": float(precision_score(y_test, best_pred, zero_division=0)),
            "recall": float(recall_score(y_test, best_pred, zero_division=0)),
            "f1": float(f1_score(y_test, best_pred, zero_division=0)),
            "roc_auc": float(roc_auc_score(y_test, best_proba))
        }, f, indent=2)

    report = classification_report(y_test, best_pred, target_names=["Retained", "Churned"])
    (REPORT_DIR / "classification_report.txt").write_text(report, encoding="utf-8")

    cm = confusion_matrix(y_test, best_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Retained", "Churned"],
                yticklabels=["Retained", "Churned"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Confusion Matrix — {best_name}")
    plt.tight_layout()
    plt.savefig(REPORT_DIR / "confusion_matrix.png", dpi=160)
    plt.close()

    print("\nModel comparison:\n", results.to_string(index=False))
    print(f"\nSelected model by test F1: {best_name}")
    print(f"Saved model: {MODEL_DIR / 'best_churn_model.joblib'}")
    print("Important: metrics are a held-out split result on the IBM sample dataset, not a guarantee of real-world performance.")

if __name__ == "__main__":
    main()
