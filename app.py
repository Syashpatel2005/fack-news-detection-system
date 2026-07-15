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
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.sparse import hstack, csr_matrix
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix,
)

sns.set_theme(style="whitegrid")
st.set_page_config(page_title="Fake News Detector", page_icon="📰", layout="wide")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CANDIDATE_NAMES = ["all_model.pkl", "all_models.pkl"]

# The trained pickle is ~100MB, too big for a normal GitHub push, so it's
# hosted on Google Drive instead and pulled down the first time the app
# runs. The Drive file's sharing must be set to "Anyone with the link".
# Extracted from: https://drive.google.com/file/d/1nfir5pQWlzSNOuy44InBiK9Lp9CyR-EG/view
GDRIVE_FILE_ID = "1nfir5pQWlzSNOuy44InBiK9Lp9CyR-EG"
DOWNLOAD_NAME = "all_models.pkl"

# Below this many non-zero TF-IDF features, a prediction has almost no
# real signal behind it (very short / out-of-vocabulary text) — see the
# "About the Model" page for why this matters.
LOW_SIGNAL_THRESHOLD = 5

# A few example headlines/snippets written to match the STYLE this model was
# actually trained on (formal wire-service phrasing vs. informal/sensational
# phrasing), so predictions on these examples are meaningful demos.
EXAMPLE_REAL = (
    "WASHINGTON (Reuters) - The Federal Reserve said on Wednesday it would "
    "hold interest rates steady, citing stable inflation and continued "
    "strength in the labor market. Officials said further decisions would "
    "depend on upcoming economic data."
)
EXAMPLE_FAKE = (
    "You won't BELIEVE what this senator just admitted on camera! Insiders "
    "say the mainstream media is covering up the shocking truth, and what "
    "happens next will leave you speechless. Share before this gets deleted!"
)


# --------------------------------------------------------------------------
# Load models (cached — read once, reused for every rerun)
# --------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading trained models...")
def load_models(path, mtime):
    with open(path, "rb") as f:
        return pickle.load(f)


def find_local_model_file():
    for name in CANDIDATE_NAMES:
        path = os.path.join(BASE_DIR, name)
        if os.path.exists(path):
            return path
    return None


def download_model_from_drive():
    """Fallback used when the pkl isn't shipped in the repo (too big for
    GitHub). Downloads it from Google Drive on first run only — once saved
    locally, later runs use find_local_model_file() instead."""
    import gdown

    dest = os.path.join(BASE_DIR, DOWNLOAD_NAME)
    url = f"https://drive.google.com/uc?id={GDRIVE_FILE_ID}"

    with st.spinner("Downloading model file from Google Drive (~100MB, one-time)..."):
        try:
            gdown.download(url, dest, quiet=False)
        except Exception as e:
            st.error(f"Download from Google Drive failed: {e}")
            return None

    return dest if os.path.exists(dest) else None


model_path = find_local_model_file() or download_model_from_drive()
if model_path is None:
    st.error(
        "Couldn't find `all_model.pkl` / `all_models.pkl` locally, and the "
        "Google Drive download failed. Place the pickle file next to "
        "app.py, or check that the Drive file is shared as "
        "\"Anyone with the link.\""
    )
    st.stop()

data = load_models(model_path, os.path.getmtime(model_path))
trained_models = data["models"]          # {"LR__Text Only": (model, preds, y_test), ...}
tfidf_A = data.get("tfidf_A")
tfidf_B = data.get("tfidf_B")
scaler_B = data.get("scaler_B")


# --------------------------------------------------------------------------
# Text cleaning + prediction helpers
# --------------------------------------------------------------------------
def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def build_features(text, version, article_date=None):
    """Clean + vectorize input text, adding date features if the chosen
    model version needs them. Returns (X, non_zero_feature_count)."""
    cleaned = clean_text(text)
    vectorizer = tfidf_A if version == "Text Only" else tfidf_B
    X = vectorizer.transform([cleaned])
    non_zero = X.nnz

    if version != "Text Only" and scaler_B is not None:
        date_scaled = scaler_B.transform(
            [[article_date.year, article_date.month, article_date.day]]
        )
        X = hstack([X, csr_matrix(date_scaled)])

    return X, non_zero


def predict(model_key, text, article_date=None):
    """Run a full prediction and return a dict of everything the UI needs."""
    version = model_key.split("__")[1]
    model = trained_models[model_key][0]

    X, non_zero = build_features(text, version, article_date)
    pred = model.predict(X)[0]
    proba = model.predict_proba(X)[0] if hasattr(model, "predict_proba") else None

    return {
        "label": "Real" if pred == 1 else "Fake",
        "proba": proba,
        "low_signal": non_zero < LOW_SIGNAL_THRESHOLD,
        "non_zero_features": non_zero,
    }


# --------------------------------------------------------------------------
# Page: Check a headline or article
# --------------------------------------------------------------------------
def page_predict():
    st.header("🔎 Check a headline or article")
    st.caption(
        "Works best on full news-style paragraphs. Short one-line claims "
        "give the model very little to go on — see **About the Model** "
        "in the sidebar for why."
    )

    model_names = list(trained_models.keys())
    chosen = st.selectbox("Model", model_names)
    version = chosen.split("__")[1]  # "Text Only" or "Text + Date"

    if version != "Text Only":
        st.caption("This model uses **text + publish date** as features.")

    # Example buttons let users try text that matches the training style
    col1, col2, _ = st.columns([1, 1, 3])
    if col1.button("Try a real-news example"):
        st.session_state["news_text"] = EXAMPLE_REAL
    if col2.button("Try a fake-news example"):
        st.session_state["news_text"] = EXAMPLE_FAKE

    with st.form("predict_form"):
        text_in = st.text_area(
            "News text",
            height=160,
            placeholder="Paste the headline or article body...",
            key="news_text",
        )

        date_in = None
        if version != "Text Only":
            date_in = st.date_input(
                "Article date",
                value=datetime.date.today(),
                help="Publish date of the article — feeds the model's date features.",
            )

        submitted = st.form_submit_button("Predict", type="primary")

    if not submitted:
        return
    if not text_in.strip():
        st.warning("Enter some text first.")
        return

    result = predict(chosen, text_in, date_in)
    icon = "🟢" if result["label"] == "Real" else "🔴"
    st.subheader(f"Prediction: {icon} {result['label']}")

    if result["low_signal"]:
        st.warning(
            f"⚠️ Low signal: only {result['non_zero_features']} recognized "
            "features found in this text (short input, or words the model "
            "never saw in training). Treat this prediction as unreliable."
        )

    if result["proba"] is not None:
        prob_df = pd.DataFrame({"Class": ["Fake", "Real"], "Probability": result["proba"]})
        fig, ax = plt.subplots(figsize=(4, 2.5))
        sns.barplot(data=prob_df, x="Class", y="Probability",
                    hue="Class", palette="Set2", legend=False, ax=ax)
        ax.set_ylim(0, 1)
        st.pyplot(fig, clear_figure=True)


# --------------------------------------------------------------------------
# Page: Model performance comparison
# --------------------------------------------------------------------------
def page_performance():
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

    col1, col2 = st.columns(2)
    with col1:
        st.dataframe(
            results_df.style.format({c: "{:.4f}" for c in
                                      ["Accuracy", "Precision", "Recall", "F1"]}),
            use_container_width=True, hide_index=True,
        )
    with col2:
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.barplot(data=results_df, x="Model", y="F1", hue="Version",
                    palette="Set2", ax=ax)
        ax.set_title("F1-score by model & feature version")
        st.pyplot(fig, clear_figure=True)


# --------------------------------------------------------------------------
# Page: Confusion matrix + class imbalance
# --------------------------------------------------------------------------
def page_confusion():
    st.header("🧩 Confusion Matrix & Class Balance")

    model_names = list(trained_models.keys())
    cm_pick = st.selectbox("Model for confusion matrix / test-set balance", model_names, key="cm")
    model, preds, y_test = trained_models[cm_pick]

    col1, col2 = st.columns(2)
    with col1:
        cm = confusion_matrix(y_test, preds)
        fig, ax = plt.subplots(figsize=(4, 3.5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=["Fake", "Real"], yticklabels=["Fake", "Real"], ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title("Confusion Matrix")
        st.pyplot(fig, clear_figure=True)

    with col2:
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


# --------------------------------------------------------------------------
# Page: About the Model & Data
# --------------------------------------------------------------------------
def page_about():
    st.header("ℹ️ About the Model & Data")

    st.subheader("What data trained this model")
    st.markdown(
        "- **Real news**: wire-service articles (Reuters-style), mostly "
        "**US politics and world news**.\n"
        "- **Fake news**: articles pulled from flagged fake-news / hoax / "
        "hyper-partisan sites, covering similar US-politics topics.\n"
        "- Text was lowercased, stripped of URLs/HTML/punctuation/numbers, "
        "and reduced to lemmatized keywords before being converted into "
        "TF-IDF word/phrase features. The model never sees the raw sentence "
        "— only which words and short phrases appear and how often."
    )

    st.subheader("What this model is actually detecting")
    st.markdown(
        "This is a **writing-style classifier**, not a fact-checker. It "
        "learned to tell apart the *phrasing and formatting* typical of "
        "wire-service journalism versus typical of fake-news sites — it "
        "has no knowledge of real-world facts, current events, or who's "
        "alive or dead. It cannot verify a claim; it can only recognize "
        "whether the writing *style* resembles its Real or Fake training "
        "examples."
    )

    col1, col2 = st.columns(2)
    with col1:
        st.success("✅ Predicts well on:")
        st.markdown(
            "- Full news-length paragraphs, not single short claims\n"
            "- US politics / government / world-news topics\n"
            "- Formal, third-person wire-service phrasing\n"
            "- English text similar in period/style to its training data"
        )
    with col2:
        st.error("❌ Struggles on:")
        st.markdown(
            "- One-line claims or headlines with very few words\n"
            "- Names/topics outside US politics (little or no training "
            "vocabulary for them)\n"
            "- Sensational text that happens to be written formally, or "
            "true text written informally\n"
            "- Anything requiring actual fact-checking (dates, deaths, "
            "statistics) — the model can't verify facts, only style"
        )

    st.subheader("Example inputs (matched to this model's training style)")
    ex1, ex2 = st.columns(2)
    with ex1:
        st.caption("Likely predicted **Real** — formal wire-service style")
        st.code(EXAMPLE_REAL, language=None)
    with ex2:
        st.caption("Likely predicted **Fake** — sensational, clickbait style")
        st.code(EXAMPLE_FAKE, language=None)

    st.info(
        "Tip: go to **Check a headline or article** and use the "
        "\"Try a real/fake example\" buttons to see these run through the "
        "model live."
    )


# --------------------------------------------------------------------------
# Sidebar navigation
# --------------------------------------------------------------------------
st.title("📰 Fake News Detector")
st.caption(f"Models loaded from `{os.path.basename(model_path)}` ✅")

st.sidebar.title("📰 Navigation")
page = st.sidebar.radio(
    "Go to",
    [
        "Check a headline or article",
        "Model Performance",
        "Confusion Matrix & Class Balance",
        "About the Model & Data",
    ],
)
st.sidebar.divider()
st.sidebar.caption(f"Models file: `{os.path.basename(model_path)}`")
st.sidebar.caption(f"Available models: {len(trained_models)}")

PAGES = {
    "Check a headline or article": page_predict,
    "Model Performance": page_performance,
    "Confusion Matrix & Class Balance": page_confusion,
    "About the Model & Data": page_about,
}
PAGES[page]()
