"""Train 5 classifiers on Breast Cancer Wisconsin (Diagnostic) and save them.

Run from the project root:   python model/train_models.py
Outputs: model/*.pkl, test_data.csv, model/metrics.csv
"""
import os
import joblib
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score,
                             recall_score, f1_score, matthews_corrcoef)

RANDOM_STATE = 42
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def get_models():
    """Return a dict {display name: untrained estimator}."""
    return {
        "Logistic Regression": Pipeline([("scale", StandardScaler()),
                                         ("clf", LogisticRegression(max_iter=1000))]),
        "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=RANDOM_STATE),
        "kNN": Pipeline([("scale", StandardScaler()),
                         ("clf", KNeighborsClassifier(n_neighbors=7))]),
        "Naive Bayes": GaussianNB(),
        "Support Vector Machine": Pipeline([("scale", StandardScaler()),
                                            ("clf", SVC(kernel="rbf", C=1.0, probability=True,
                                                        random_state=RANDOM_STATE))]),
    }


def file_name(model_name):
    return model_name.lower().replace(" ", "_") + ".pkl"


def compute_metrics(y_true, y_pred, y_prob):
    return {"Accuracy": accuracy_score(y_true, y_pred),
            "AUC": roc_auc_score(y_true, y_prob),
            "Precision": precision_score(y_true, y_pred),
            "Recall": recall_score(y_true, y_pred),
            "F1": f1_score(y_true, y_pred),
            "MCC": matthews_corrcoef(y_true, y_pred)}


def main():
    data = load_breast_cancer(as_frame=True)
    X, y = data.data, data.target          # target: 1 = benign, 0 = malignant
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)

    rows = {}
    for name, model in get_models().items():
        model.fit(X_tr, y_tr)
        joblib.dump(model, os.path.join(HERE, file_name(name)))
        rows[name] = compute_metrics(y_te, model.predict(X_te), model.predict_proba(X_te)[:, 1])

    metrics = pd.DataFrame(rows).T.round(4)
    metrics.index.name = "ML Model Name"
    metrics.to_csv(os.path.join(HERE, "metrics.csv"))
    print(metrics.to_markdown() if hasattr(metrics, "to_markdown") else metrics)

    test_df = X_te.copy()
    test_df["target"] = y_te.values
    test_df.to_csv(os.path.join(ROOT, "test_data.csv"), index=False)
    print(f"\nSaved {len(test_df)} test rows to test_data.csv")


if __name__ == "__main__":
    main()
