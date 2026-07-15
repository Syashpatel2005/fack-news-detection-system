"""
Fake News Detector — Simple Streamlit Dashboard
================================================
Project folder layout expected:
    app.py
    all_model.pkl        (or all_models.pkl — either name is auto-detected)
    requirement.txt

Run with:
    streamlit run app.py
"""

import os
import re
import pickle
import datetime
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix
)

sns.set_theme(style="whitegrid")

st.set_page_config(page_title="Fake News Detector", page_icon="📰", layout="wide")

# --------------------------------------------------------------------------
# Load all_model.pkl from disk (cached — read once, reused for every rerun)
# --------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CANDIDATE_NAMES = ["all_model.pkl", "all_models.pkl"]


@st.cache_resource(show_spinner="Loading trained models...")
def load_models(path, mtime):
    with open(path, "rb") as f:
        return pickle.load(f)


model_path = None
for name in CANDIDATE_NAMES:
    p = os.path.join(BASE_DIR, name)
    if os.path.exists(p):
        model_path = p
        break

if model_path is None:
    st.error(
        "Couldn't find `all_model.pkl` (or `all_models.pkl`) next to app.py. "
        "Place your pickle file in the same folder and rerun."
    )
    st.stop()

data = load_models(model_path, os.path.getmtime(model_path))
trained_models = data["models"]          # {"LR__Text Only": (model, preds, y_test), ...}
tfidf_A = data.get("tfidf_A")
tfidf_B = data.get("tfidf_B")
scaler_B = data.get("scaler_B")

# --------------------------------------------------------------------------
# Basic text cleaning (matches the notebook's clean_text step closely enough
# for TF-IDF scoring at inference time)
# --------------------------------------------------------------------------
def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# --------------------------------------------------------------------------
# Header
# --------------------------------------------------------------------------
st.title("📰 Fake News Detector")
st.caption(f"Models loaded from `{os.path.basename(model_path)}` ✅")

st.sidebar.title("📰 Navigation")
page = st.sidebar.radio(
    "Go to",
    ["Check a headline or article", "Model Performance", "Confusion Matrix & Class Balance"],
)
st.sidebar.divider()
st.sidebar.caption(f"Models file: `{os.path.basename(model_path)}`")
st.sidebar.caption(f"Available models: {len(trained_models)}")

# --------------------------------------------------------------------------
# Page: Check a headline or article
# --------------------------------------------------------------------------
if page == "Check a headline or article":
    st.header("🔎 Check a headline or article")

    model_names = list(trained_models.keys())
    chosen = st.selectbox("Model", model_names)
    version = chosen.split("__")[1]  # "Text Only" or "Text + Date"

    if version != "Text Only":
        st.caption("This model uses **text + publish date** as features.")

    with st.form("predict_form"):
        text_in = st.text_area("News text", height=160,
                                placeholder="Paste the headline or article body...")

        date_in = None
        if version != "Text Only":
            date_in = st.date_input(
                "Article date",
                value=datetime.date.today(),
                help="Publish date of the article — feeds the model's date features.",
            )

        submitted = st.form_submit_button("Predict", type="primary")

    if submitted:
        if not text_in.strip():
            st.warning("Enter some text first.")
        else:
            cleaned = clean_text(text_in)
            vectorizer = tfidf_A if version == "Text Only" else tfidf_B
            model = trained_models[chosen][0]

            X = vectorizer.transform([cleaned])
            if version != "Text Only" and scaler_B is not None:
                from scipy.sparse import hstack, csr_matrix
                date_features = scaler_B.transform(
                    [[date_in.year, date_in.month, date_in.day]]
                )
                X = hstack([X, csr_matrix(date_features)])

            pred = model.predict(X)[0]
            label = "🟢 Real" if pred == 1 else "🔴 Fake"
            st.subheader(f"Prediction: {label}")

            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(X)[0]
                prob_df = pd.DataFrame({"Class": ["Fake", "Real"], "Probability": proba})
                fig, ax = plt.subplots(figsize=(4, 2.5))
                sns.barplot(data=prob_df, x="Class", y="Probability",
                            hue="Class", palette="Set2", legend=False, ax=ax)
                ax.set_ylim(0, 1)
                st.pyplot(fig, clear_figure=True)

# --------------------------------------------------------------------------
# Page: Model performance comparison
# --------------------------------------------------------------------------
elif page == "Model Performance":
    st.header("📊 Model Performance")

    rows = []
    for key, (model, preds, y_test) in trained_models.items():
        model_name, version = key.split("__")
        rows.append({
            "Model": model_name,
            "Version": version,
            "Accuracy": accuracy_score(y_test, preds),
            "Precision": precision_score(y_test, preds),
            "Recall": recall_score(y_test, preds),
            "F1": f1_score(y_test, preds),
        })
    results_df = pd.DataFrame(rows).sort_values("F1", ascending=False)

    c1, c2 = st.columns([1, 1])
    with c1:
        st.dataframe(
            results_df.style.format({c: "{:.4f}" for c in
                                      ["Accuracy", "Precision", "Recall", "F1"]}),
            use_container_width=True, hide_index=True,
        )
    with c2:
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.barplot(data=results_df, x="Model", y="F1", hue="Version",
                    palette="Set2", ax=ax)
        ax.set_title("F1-score by model & feature version")
        st.pyplot(fig, clear_figure=True)

# --------------------------------------------------------------------------
# Page: Confusion matrix + class imbalance (from a chosen model's test split)
# --------------------------------------------------------------------------
elif page == "Confusion Matrix & Class Balance":
    st.header("🧩 Confusion Matrix & Class Balance")

    model_names = list(trained_models.keys())
    cm_pick = st.selectbox("Model for confusion matrix / test-set balance", model_names, key="cm")
    model, preds, y_test = trained_models[cm_pick]

    c1, c2 = st.columns([1, 1])
    with c1:
        cm = confusion_matrix(y_test, preds)
        fig, ax = plt.subplots(figsize=(4, 3.5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=["Fake", "Real"], yticklabels=["Fake", "Real"], ax=ax)
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
        ax.set_title("Confusion Matrix")
        st.pyplot(fig, clear_figure=True)

    with c2:
        counts = pd.Series(y_test).value_counts().sort_index()
        counts_df = pd.DataFrame({
            "label": ["Fake (0)", "Real (1)"],
            "count": [int(counts.get(0, 0)), int(counts.get(1, 0))],
        })
        ratio = counts_df["count"].max() / max(counts_df["count"].min(), 1)
        st.metric("Test-set imbalance ratio", f"{ratio:.2f} : 1")
        fig, ax = plt.subplots(figsize=(4, 3.5))
        sns.barplot(data=counts_df, x="label", y="count", hue="label",
                    palette="Set2", legend=False, ax=ax)
        ax.set_title("Test-set Class Distribution")
        st.pyplot(fig, clear_figure=True)
