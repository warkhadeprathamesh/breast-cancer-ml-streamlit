import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score, recall_score,
                             f1_score, matthews_corrcoef, confusion_matrix,
                             classification_report, roc_curve)

st.set_page_config(page_title="Breast Cancer Classifier Lab", page_icon="🔬", layout="wide")

BASE = os.path.dirname(os.path.abspath(__file__))
MODEL_FILES = {
    "Logistic Regression": "logistic_regression.pkl",
    "Decision Tree": "decision_tree.pkl",
    "kNN": "knn.pkl",
    "Naive Bayes": "naive_bayes.pkl",
    "Support Vector Machine": "support_vector_machine.pkl",
}
TARGET = "target"
LABELS = ["Malignant (0)", "Benign (1)"]


@st.cache_resource
def load_model(name):
    """Load a saved model; if the pickle is incompatible with the installed
    scikit-learn version, retrain it from the bundled dataset."""
    try:
        return joblib.load(os.path.join(BASE, "model", MODEL_FILES[name]))
    except Exception:
        from sklearn.datasets import load_breast_cancer
        from sklearn.model_selection import train_test_split
        from model.train_models import get_models, RANDOM_STATE
        d = load_breast_cancer(as_frame=True)
        X_tr, _, y_tr, _ = train_test_split(d.data, d.target, test_size=0.2,
                                            stratify=d.target, random_state=RANDOM_STATE)
        m = get_models()[name]
        m.fit(X_tr, y_tr)
        return m


def score(y, pred, prob):
    return {"Accuracy": accuracy_score(y, pred), "AUC": roc_auc_score(y, prob),
            "Precision": precision_score(y, pred), "Recall": recall_score(y, pred),
            "F1": f1_score(y, pred), "MCC": matthews_corrcoef(y, pred)}


# ---------------- Sidebar ----------------
st.sidebar.title("🔬 Control Panel")
st.sidebar.write("1. Upload **test data** (CSV)\n2. Pick a model\n3. Explore the results")
uploaded = st.sidebar.file_uploader("Upload test CSV", type="csv")
model_name = st.sidebar.selectbox("Select model", list(MODEL_FILES))
threshold = st.sidebar.slider("Decision threshold (benign)", 0.05, 0.95, 0.50, 0.05)
try:
    with open(os.path.join(BASE, "test_data.csv"), "rb") as f:
        st.sidebar.download_button("⬇ Download sample test_data.csv", f, "test_data.csv", "text/csv")
except FileNotFoundError:
    pass

# ---------------- Main ----------------
st.title("Breast Cancer Diagnosis: Model Comparison App")
st.caption("UCI Breast Cancer Wisconsin (Diagnostic) · 5 classifiers · positive class = Benign (1)")

if uploaded is None:
    st.info("👈 Upload a CSV in the sidebar to begin. Use the sample test_data.csv "
            "(30 feature columns + a `target` column).")
    st.stop()

df = pd.read_csv(uploaded)
if TARGET not in df.columns:
    st.error(f"Your CSV must contain a '{TARGET}' column (0 = malignant, 1 = benign).")
    st.stop()

# Make sure feature columns match what the models were trained on
ref = load_model("Logistic Regression")
expected = list(ref.feature_names_in_) if hasattr(ref, "feature_names_in_") else \
    list(ref.named_steps["clf"].feature_names_in_)
missing = [c for c in expected if c not in df.columns]
if missing:
    st.error(f"Missing {len(missing)} required feature column(s): {missing[:5]} ...")
    st.stop()

X, y = df[expected], df[TARGET].astype(int)
if y.nunique() < 2:
    st.error("The target column needs both classes (0 and 1) to compute AUC.")
    st.stop()

c1, c2, c3 = st.columns(3)
c1.metric("Rows", len(df))
c2.metric("Features", len(expected))
c3.metric("Benign share", f"{y.mean():.1%}")
with st.expander("Preview uploaded data"):
    st.dataframe(df.head(10), use_container_width=True)

model = load_model(model_name)
prob = model.predict_proba(X)[:, 1]
pred = (prob >= threshold).astype(int)
m = score(y, pred, prob)

tab1, tab2, tab3, tab4 = st.tabs(["📊 Metrics", "🧮 Confusion matrix & report",
                                  "📈 ROC curve", "🏆 Compare all models"])

with tab1:
    st.subheader(f"{model_name}: evaluation metrics")
    cols = st.columns(6)
    for col, (k, v) in zip(cols, m.items()):
        col.metric(k, f"{v:.4f}")

with tab2:
    left, right = st.columns(2)
    with left:
        cm = confusion_matrix(y, pred, labels=[0, 1])
        fig, ax = plt.subplots(figsize=(4.5, 3.8))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                    xticklabels=LABELS, yticklabels=LABELS, ax=ax)
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
        st.pyplot(fig)
        st.caption("Top-right cell = malignant predicted as benign (the costliest error).")
    with right:
        rep = classification_report(y, pred, target_names=LABELS, output_dict=True, zero_division=0)
        st.dataframe(pd.DataFrame(rep).T.round(3), use_container_width=True)

with tab3:
    fpr, tpr, _ = roc_curve(y, prob)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(fpr, tpr, label=f"{model_name} (AUC = {m['AUC']:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate"); ax.legend()
    st.pyplot(fig)

with tab4:
    rows = {}
    for n in MODEL_FILES:
        mm = load_model(n)
        p = mm.predict_proba(X)[:, 1]
        rows[n] = score(y, (p >= threshold).astype(int), p)
    comp = pd.DataFrame(rows).T.round(4)
    st.dataframe(comp.style.highlight_max(axis=0, color="#c8e6c9"), use_container_width=True)
    st.bar_chart(comp[["Accuracy", "F1", "MCC"]])
    st.success(f"Best MCC on this data: **{comp['MCC'].idxmax()}** ({comp['MCC'].max():.4f})")
