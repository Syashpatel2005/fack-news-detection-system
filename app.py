"""
Fake News Detector — Interactive Streamlit Dashboard
=====================================================
Companion dashboard for the fake_news_detector_balanced.ipynb pipeline.

Run with:
    streamlit run app.py

Optional inputs (upload via the sidebar, all optional — the dashboard
degrades gracefully and shows sample/demo data if a file is missing):
    - Fake.csv / True.csv   -> raw Kaggle "Fake and Real News" files
    - preprocessed_fake_real_news.csv -> output of the notebook's cleaning step
    - all_models.pkl        -> pickle saved at the end of the notebook
                                (dict with 'models', 'tfidf_A', 'tfidf_B', 'scaler_B')
"""

import io
import os
import re
import pickle
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

try:
    from imblearn.under_sampling import RandomUnderSampler
    from imblearn.over_sampling import SMOTE
    IMBLEARN_OK = True
except ImportError:
    IMBLEARN_OK = False

sns.set_theme(style="whitegrid")

st.set_page_config(
    page_title="Fake News Detector Dashboard",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded",
)

RANDOM_STATE = 42
DEFAULT_MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "all_models.pkl")

# --------------------------------------------------------------------------
# Text preprocessing (mirrors the notebook's pipeline, kept dependency-light
# so the dashboard doesn't require an nltk download step to boot up)
# --------------------------------------------------------------------------
BASIC_STOPWORDS = set("""
a about above after again against all am an and any are aren't as at be
because been before being below between both but by can't cannot could
couldn't did didn't do does doesn't doing don't down during each few for
from further had hadn't has hasn't have haven't having he he'd he'll he's
her here here's hers herself him himself his how how's i i'd i'll i'm i've
if in into is isn't it it's its itself let's me more most mustn't my myself
no nor not of off on once only or other ought our ours ourselves out over
own same shan't she she'd she'll she's should shouldn't so some such than
that that's the their theirs them themselves then there there's these they
they'd they'll they're they've this those through to too under until up
very was wasn't we we'd we'll we're we've were weren't what what's when
when's where where's which while who who's whom why why's with won't would
wouldn't you you'd you'll you're you've your yours yourself yourselves said
also would could us
""".split())


def clean_text_basic(text: str) -> str:
    """Lightweight version of the notebook's clean/tokenize/stopword pipeline."""
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"@\w+|#\w+", "", text)
    text = re.sub(r"\d+", "", text)
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    tokens = [t for t in text.split() if t not in BASIC_STOPWORDS and len(t) > 2]
    return " ".join(tokens)


# --------------------------------------------------------------------------
# Cached loaders
# --------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def make_demo_data(n_fake=1800, n_real=600, seed=RANDOM_STATE):
    """Synthetic, clearly-labeled demo set so the dashboard is fully explorable
    even with zero uploads. Deliberately imbalanced (3:1) to mirror the
    notebook's real-world Fake/Real skew."""
    rng = np.random.default_rng(seed)
    fake_vocab = ["shocking", "secret", "government", "hidden", "cure", "conspiracy",
                  "leaked", "banned", "miracle", "exposed", "cover", "elite", "hoax"]
    real_vocab = ["reported", "according", "officials", "study", "announced",
                  "committee", "statement", "policy", "meeting", "data", "research"]
    subjects_fake = ["News", "politics", "left-news", "Government News", "US_News"]
    subjects_real = ["politicsNews", "worldnews"]

    def gen_text(vocab, k=25):
        return " ".join(rng.choice(vocab, size=k))

    rows = []
    for _ in range(n_fake):
        rows.append({"text": gen_text(fake_vocab + real_vocab[:2]),
                      "subject": rng.choice(subjects_fake), "label": 0})
    for _ in range(n_real):
        rows.append({"text": gen_text(real_vocab + fake_vocab[:2]),
                      "subject": rng.choice(subjects_real), "label": 1})
    df = pd.DataFrame(rows).sample(frac=1, random_state=seed).reset_index(drop=True)
    df["clean_text"] = df["text"].apply(clean_text_basic)
    return df


@st.cache_data(show_spinner=False)
def load_raw_csvs(fake_bytes, true_bytes):
    fake = pd.read_csv(io.BytesIO(fake_bytes))
    true = pd.read_csv(io.BytesIO(true_bytes))
    fake = fake.copy(); true = true.copy()
    fake["label"] = 0
    true["label"] = 1
    df = pd.concat([fake, true], ignore_index=True)
    text_col = "text" if "text" in df.columns else df.columns[1]
    df["clean_text"] = df[text_col].astype(str).apply(clean_text_basic)
    df = df.rename(columns={text_col: "text"})
    return df


@st.cache_data(show_spinner=False)
def load_preprocessed_csv(file_bytes):
    df = pd.read_csv(io.BytesIO(file_bytes))
    if "clean_text" not in df.columns and "text" in df.columns:
        df["clean_text"] = df["text"].astype(str).apply(clean_text_basic)
    return df


@st.cache_resource(show_spinner=False)
def load_models_pkl_bytes(file_bytes):
    return pickle.load(io.BytesIO(file_bytes))


@st.cache_resource(show_spinner=False)
def load_models_pkl_path(path, mtime):
    # `mtime` is only in the signature so Streamlit's cache invalidates
    # automatically if you swap in a newer all_models.pkl on disk.
    with open(path, "rb") as f:
        return pickle.load(f)


# --------------------------------------------------------------------------
# Sidebar — data & model inputs
# --------------------------------------------------------------------------
st.sidebar.title("📰 Fake News Detector")
st.sidebar.caption("Dashboard for the imbalance-aware ML pipeline")

st.sidebar.subheader("1. Data")
data_mode = st.sidebar.radio(
    "Data source",
    ["Use demo data", "Upload preprocessed CSV", "Upload raw Fake.csv + True.csv"],
    index=0,
)

df = None
if data_mode == "Use demo data":
    df = make_demo_data()
    st.sidebar.caption("Synthetic 3:1 imbalanced demo set (no upload needed).")
elif data_mode == "Upload preprocessed CSV":
    up = st.sidebar.file_uploader("preprocessed_fake_real_news.csv", type="csv")
    if up is not None:
        df = load_preprocessed_csv(up.getvalue())
else:
    f_up = st.sidebar.file_uploader("Fake.csv", type="csv", key="fake")
    t_up = st.sidebar.file_uploader("True.csv", type="csv", key="true")
    if f_up is not None and t_up is not None:
        df = load_raw_csvs(f_up.getvalue(), t_up.getvalue())

if df is None:
    st.sidebar.info("Waiting for upload — showing demo data meanwhile.")
    df = make_demo_data()

st.sidebar.subheader("2. Trained models")
saved_models = None
model_source = None

if os.path.exists(DEFAULT_MODEL_PATH):
    mtime = os.path.getmtime(DEFAULT_MODEL_PATH)
    saved_models = load_models_pkl_path(DEFAULT_MODEL_PATH, mtime)
    model_source = "disk"
    st.sidebar.success("✅ all_models.pkl loaded from disk (cached)")

pkl_up = st.sidebar.file_uploader(
    "…or upload/override all_models.pkl", type="pkl"
)
if pkl_up is not None:
    saved_models = load_models_pkl_bytes(pkl_up.getvalue())
    model_source = "upload"

if saved_models is None:
    st.sidebar.warning(
        "No all_models.pkl found. Place your file next to app.py, or upload it here."
    )

st.sidebar.subheader("3. Navigate")
page = st.sidebar.radio(
    "Section",
    ["Overview", "Class Imbalance", "Imbalance Handling Lab", "Exploratory Analysis",
     "Train & Evaluate", "Live Prediction"],
)

# --------------------------------------------------------------------------
# Shared helpers
# --------------------------------------------------------------------------
def label_counts(frame):
    c = frame["label"].value_counts().sort_index()
    return pd.DataFrame({
        "label": ["Fake (0)", "Real (1)"],
        "count": [int(c.get(0, 0)), int(c.get(1, 0))],
    })


def imbalance_ratio(frame):
    c = frame["label"].value_counts()
    if len(c) < 2 or c.min() == 0:
        return float("nan")
    return c.max() / c.min()


# ==========================================================================
# PAGE: Overview
# ==========================================================================
if page == "Overview":
    st.title("📰 Fake News Detector — Model Dashboard")
    st.markdown(
        "This dashboard accompanies a notebook that trains **Multinomial Naive "
        "Bayes**, **KNN**, and **Logistic Regression** on TF-IDF text features, "
        "with three different strategies for handling class imbalance between "
        "the Fake and Real classes."
    )

    counts = label_counts(df)
    ratio = imbalance_ratio(df)
    total = int(counts["count"].sum())

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total articles", f"{total:,}")
    c2.metric("Fake", f"{int(counts.loc[0,'count']):,}")
    c3.metric("Real", f"{int(counts.loc[1,'count']):,}")
    c4.metric("Imbalance ratio", f"{ratio:.2f} : 1")

    st.divider()
    left, right = st.columns([1, 1])
    with left:
        st.subheader("Class distribution")
        fig, ax = plt.subplots(figsize=(4.5, 3.5))
        sns.barplot(data=counts, x="label", y="count", hue="label",
                    palette="Set2", legend=False, ax=ax)
        ax.set_xlabel(""); ax.set_ylabel("Count")
        st.pyplot(fig, clear_figure=True)
    with right:
        st.subheader("Subject mix (if available)")
        if "subject" in df.columns:
            top_subj = df["subject"].value_counts().head(8)
            fig, ax = plt.subplots(figsize=(4.5, 3.5))
            sns.barplot(x=top_subj.values, y=top_subj.index, hue=top_subj.index,
                        palette="viridis", legend=False, ax=ax)
            ax.set_xlabel("Count"); ax.set_ylabel("")
            st.pyplot(fig, clear_figure=True)
        else:
            st.info("No `subject` column in this dataset.")

    if saved_models is not None:
        st.success("Trained models detected — see **Train & Evaluate** and "
                    "**Live Prediction** for full results.")
    else:
        st.info("No `all_models.pkl` uploaded — Train & Evaluate can still "
                "train fresh, lightweight models on whatever data is loaded.")

# ==========================================================================
# PAGE: Class Imbalance
# ==========================================================================
elif page == "Class Imbalance":
    st.title("⚖️ Class Imbalance")
    counts = label_counts(df)
    ratio = imbalance_ratio(df)

    st.markdown(
        f"Out of **{int(counts['count'].sum()):,}** articles, the majority "
        f"class outnumbers the minority class **{ratio:.2f}× **. Left "
        "untreated, this biases every model toward the majority class."
    )

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Counts")
        st.dataframe(counts, use_container_width=True, hide_index=True)
        fig, ax = plt.subplots(figsize=(4.5, 4))
        sns.barplot(data=counts, x="label", y="count", hue="label",
                    palette="Set2", legend=False, ax=ax)
        ax.set_title("Class Distribution Before Balancing")
        st.pyplot(fig, clear_figure=True)
    with c2:
        st.subheader("Proportions")
        pct = counts.copy()
        pct["pct"] = (pct["count"] / pct["count"].sum() * 100).round(1)
        st.dataframe(pct, use_container_width=True, hide_index=True)
        fig, ax = plt.subplots(figsize=(4.5, 4))
        ax.pie(counts["count"], labels=counts["label"], autopct="%1.1f%%",
               colors=sns.color_palette("Set2"), startangle=90)
        ax.set_title("Class Share")
        st.pyplot(fig, clear_figure=True)

    st.divider()
    st.subheader("Why this matters")
    st.markdown(
        "- A model that always predicts the majority class can still score "
        "high **accuracy** while being useless — this is why the notebook "
        "reports **F1** (and F1 on the minority class specifically) rather "
        "than accuracy alone.\n"
        "- Three fixes are used depending on what each model supports: "
        "`sample_weight` for Naive Bayes, `class_weight='balanced'` for "
        "Logistic Regression, and `RandomUnderSampler` for KNN (which has no "
        "weighting mechanism at all)."
    )

# ==========================================================================
# PAGE: Imbalance Handling Lab
# ==========================================================================
elif page == "Imbalance Handling Lab":
    st.title("🧪 Imbalance Handling Lab")
    st.markdown(
        "Pick a resampling strategy and see how it reshapes the **training "
        "split's** class counts. (Test data is never resampled — it must stay "
        "representative of the real world.)"
    )

    method = st.selectbox(
        "Strategy",
        ["None (baseline)", "Random Undersampling", "SMOTE (oversampling)",
         "Class-weighted (no resampling)"],
    )
    test_size = st.slider("Test set size", 0.1, 0.4, 0.2, 0.05)

    y = df["label"].values
    X_idx = np.arange(len(df)).reshape(-1, 1)
    X_train_idx, X_test_idx, y_train, y_test = train_test_split(
        X_idx, y, test_size=test_size, random_state=RANDOM_STATE, stratify=y
    )

    before = pd.Series(y_train).value_counts().sort_index()

    if method == "Random Undersampling" and IMBLEARN_OK:
        rus = RandomUnderSampler(random_state=RANDOM_STATE)
        _, y_res = rus.fit_resample(X_train_idx, y_train)
        after = pd.Series(y_res).value_counts().sort_index()
    elif method == "SMOTE (oversampling)" and IMBLEARN_OK:
        sm = SMOTE(random_state=RANDOM_STATE)
        _, y_res = sm.fit_resample(X_train_idx, y_train)
        after = pd.Series(y_res).value_counts().sort_index()
    elif method == "Class-weighted (no resampling)":
        after = before  # row counts unchanged; weights applied during fit
        weights = compute_sample_weight("balanced", y_train)
        st.caption(
            f"Row counts stay the same. Effective weight per Fake sample: "
            f"{weights[y_train==0][0]:.3f} — per Real sample: "
            f"{weights[y_train==1][0]:.3f}"
        )
    else:
        after = before
        if not IMBLEARN_OK and "SMOTE" in method or "Undersampling" in method:
            st.warning("`imbalanced-learn` isn't installed in this environment; "
                       "install it (`pip install imbalanced-learn`) to see this "
                       "strategy applied. Showing baseline instead.")

    comp = pd.DataFrame({
        "label": ["Fake (0)", "Real (1)"],
        "Before": [int(before.get(0, 0)), int(before.get(1, 0))],
        "After": [int(after.get(0, 0)), int(after.get(1, 0))],
    })
    comp_melt = comp.melt(id_vars="label", var_name="stage", value_name="count")

    c1, c2 = st.columns([1, 1])
    with c1:
        st.dataframe(comp, use_container_width=True, hide_index=True)
    with c2:
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.barplot(data=comp_melt, x="label", y="count", hue="stage",
                    palette="muted", ax=ax)
        ax.set_title(f"Training set — {method}")
        st.pyplot(fig, clear_figure=True)

# ==========================================================================
# PAGE: Exploratory Analysis
# ==========================================================================
elif page == "Exploratory Analysis":
    st.title("🔍 Exploratory Data Analysis")

    df["text_len"] = df["clean_text"].astype(str).apply(lambda t: len(t.split()))

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Word count distribution by class")
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.histplot(data=df, x="text_len", hue="label", bins=40,
                     palette="Set2", element="step", ax=ax)
        ax.set_xlabel("Words per article (cleaned)")
        st.pyplot(fig, clear_figure=True)
    with c2:
        st.subheader("Word count — boxplot")
        fig, ax = plt.subplots(figsize=(5, 4))
        plot_df = df.copy()
        plot_df["label"] = plot_df["label"].map({0: "Fake", 1: "Real"})
        sns.boxplot(data=plot_df, x="label", y="text_len", hue="label",
                    palette="Set2", legend=False, ax=ax)
        st.pyplot(fig, clear_figure=True)

    st.divider()
    st.subheader("Most frequent words per class")
    top_n = st.slider("Top N words", 5, 30, 15)
    which = st.radio("Class", ["Fake", "Real"], horizontal=True)
    lbl = 0 if which == "Fake" else 1
    text_blob = " ".join(df.loc[df["label"] == lbl, "clean_text"].astype(str))
    word_counts = pd.Series(text_blob.split()).value_counts().head(top_n)
    if len(word_counts):
        fig, ax = plt.subplots(figsize=(7, max(3, top_n * 0.3)))
        sns.barplot(x=word_counts.values, y=word_counts.index,
                    hue=word_counts.index, palette="rocket", legend=False, ax=ax)
        ax.set_xlabel("Frequency"); ax.set_ylabel("")
        st.pyplot(fig, clear_figure=True)
    else:
        st.info("Not enough text to compute word frequencies.")

    if "subject" in df.columns:
        st.divider()
        st.subheader("Articles by subject × label")
        cross = pd.crosstab(df["subject"], df["label"]).rename(
            columns={0: "Fake", 1: "Real"})
        st.dataframe(cross, use_container_width=True)

# ==========================================================================
# PAGE: Train & Evaluate
# ==========================================================================
elif page == "Train & Evaluate":
    st.title("🛠️ Train & Evaluate")

    if saved_models is not None:
        st.success("Using models loaded from `all_models.pkl`.")
        trained = saved_models["models"]
        rows = []
        for key, (model, preds, y_test) in trained.items():
            model_name, version = key.split("__")
            rows.append({
                "Model": model_name, "Version": version,
                "Accuracy": accuracy_score(y_test, preds),
                "Precision": precision_score(y_test, preds),
                "Recall": recall_score(y_test, preds),
                "F1": f1_score(y_test, preds),
            })
        results_df = pd.DataFrame(rows).sort_values("F1", ascending=False)
        st.dataframe(results_df.style.format({c: "{:.4f}" for c in
                     ["Accuracy", "Precision", "Recall", "F1"]}),
                     use_container_width=True, hide_index=True)

        fig, ax = plt.subplots(figsize=(9, 4.5))
        sns.barplot(data=results_df, x="Model", y="F1", hue="Version",
                    palette="Set2", ax=ax)
        ax.set_title("F1-score by model & feature version")
        st.pyplot(fig, clear_figure=True)

        st.divider()
        pick = st.selectbox("Inspect a model's confusion matrix", list(trained.keys()))
        model, preds, y_test = trained[pick]
        cm = confusion_matrix(y_test, preds)
        c1, c2 = st.columns([1, 1])
        with c1:
            fig, ax = plt.subplots(figsize=(4.5, 4))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                        xticklabels=["Fake", "Real"], yticklabels=["Fake", "Real"], ax=ax)
            ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
            st.pyplot(fig, clear_figure=True)
        with c2:
            st.text(classification_report(y_test, preds, target_names=["Fake", "Real"]))

    else:
        st.info(
            "No `all_models.pkl` uploaded — training lightweight demo models "
            "on the currently loaded data instead (TF-IDF + NB / LR / KNN, "
            "capped for speed)."
        )
        strategy = st.selectbox(
            "Imbalance strategy to apply during training",
            ["class_weight (Logistic Regression only)", "Random Undersampling",
             "None"],
        )
        max_features = st.slider("TF-IDF max features", 500, 5000, 2000, 500)

        if st.button("Train demo models", type="primary"):
            with st.spinner("Vectorizing and training..."):
                X_train_txt, X_test_txt, y_train, y_test = train_test_split(
                    df["clean_text"].astype(str), df["label"],
                    test_size=0.25, random_state=RANDOM_STATE, stratify=df["label"]
                )
                tfidf = TfidfVectorizer(max_features=max_features)
                X_train = tfidf.fit_transform(X_train_txt)
                X_test = tfidf.transform(X_test_txt)

                if strategy == "Random Undersampling" and IMBLEARN_OK:
                    rus = RandomUnderSampler(random_state=RANDOM_STATE)
                    X_train_bal, y_train_bal = rus.fit_resample(X_train, y_train)
                else:
                    X_train_bal, y_train_bal = X_train, y_train

                models = {
                    "Naive Bayes": MultinomialNB(),
                    "KNN": KNeighborsClassifier(n_neighbors=5, metric="cosine"),
                    "Logistic Regression": LogisticRegression(
                        max_iter=1000,
                        class_weight="balanced"
                        if strategy.startswith("class_weight") else None,
                    ),
                }
                rows = []
                cms = {}
                for name, model in models.items():
                    model.fit(X_train_bal, y_train_bal)
                    preds = model.predict(X_test)
                    rows.append({
                        "Model": name,
                        "Accuracy": accuracy_score(y_test, preds),
                        "Precision": precision_score(y_test, preds, zero_division=0),
                        "Recall": recall_score(y_test, preds, zero_division=0),
                        "F1": f1_score(y_test, preds, zero_division=0),
                    })
                    cms[name] = confusion_matrix(y_test, preds)

                st.session_state["demo_results"] = pd.DataFrame(rows)
                st.session_state["demo_cms"] = cms

        if "demo_results" in st.session_state:
            results_df = st.session_state["demo_results"]
            st.dataframe(results_df.style.format({c: "{:.4f}" for c in
                         ["Accuracy", "Precision", "Recall", "F1"]}),
                         use_container_width=True, hide_index=True)
            fig, ax = plt.subplots(figsize=(7, 4))
            sns.barplot(data=results_df, x="Model", y="F1", hue="Model",
                        palette="Set2", legend=False, ax=ax)
            st.pyplot(fig, clear_figure=True)

            pick = st.selectbox("Confusion matrix for", results_df["Model"])
            fig, ax = plt.subplots(figsize=(4.5, 4))
            sns.heatmap(st.session_state["demo_cms"][pick], annot=True, fmt="d",
                        cmap="Blues", xticklabels=["Fake", "Real"],
                        yticklabels=["Fake", "Real"], ax=ax)
            st.pyplot(fig, clear_figure=True)

# ==========================================================================
# PAGE: Live Prediction
# ==========================================================================
elif page == "Live Prediction":
    st.title("🕵️ Live Prediction")
    text_in = st.text_area("Paste a news headline or article body", height=180,
                            placeholder="Type or paste text here...")

    if saved_models is not None:
        model_choice = st.selectbox("Model", list(saved_models["models"].keys()))
        if st.button("Predict", type="primary") and text_in.strip():
            cleaned = clean_text_basic(text_in)
            version = model_choice.split("__")[1]
            tfidf = saved_models["tfidf_A"] if version == "Text Only" else saved_models["tfidf_B"]
            X = tfidf.transform([cleaned])
            model = saved_models["models"][model_choice][0]
            pred = model.predict(X)[0]
            label = "🟢 Real" if pred == 1 else "🔴 Fake"
            st.subheader(f"Prediction: {label}")
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(X)[0]
                prob_df = pd.DataFrame({"class": ["Fake", "Real"], "probability": proba})
                fig, ax = plt.subplots(figsize=(4, 3))
                sns.barplot(data=prob_df, x="class", y="probability",
                            hue="class", palette="Set2", legend=False, ax=ax)
                ax.set_ylim(0, 1)
                st.pyplot(fig, clear_figure=True)
    else:
        st.info(
            "No `all_models.pkl` uploaded. Train demo models on the "
            "**Train & Evaluate** page first, or upload the pickle in the "
            "sidebar to use the notebook's real trained models here."
        )
        if "demo_results" in st.session_state and text_in.strip():
            st.warning(
                "Demo models trained on this page aren't persisted across "
                "pages in this simplified flow — upload `all_models.pkl` for "
                "a fully wired live-prediction experience."
            )

st.sidebar.divider()
st.sidebar.caption("Built from fake_news_detector_balanced.ipynb")