# 📰 Fake News Detector

A Streamlit dashboard that classifies news text as **Real** or **Fake** using
TF-IDF features and a set of trained classical ML models (Logistic
Regression, Naive Bayes, KNN), trained on a Reuters-vs-fake-news-sites
dataset (mostly US politics / world news).

## Project structure

```
fake-news-detection-system/
├── app.py                 # Streamlit dashboard
├── all_models.pkl         # Trained models + vectorizers (~100MB, see below)
├── requirements.txt
└── README.md
```

## ⚠️ About `all_models.pkl` (100MB — not stored in this repo)

The trained model file is about **100MB**, which is over GitHub's 25MB
web-upload limit, so it isn't committed to this repository. It's hosted on
Google Drive instead:

**Download link:** https://drive.google.com/file/d/1nfir5pQWlzSNOuy44InBiK9Lp9CyR-EG/view?usp=drive_link

You don't need to download it manually — **`app.py` does this automatically**
the first time it runs, if `all_model.pkl` / `all_models.pkl` isn't already
present next to it. That first run will show a "Downloading model file from
Google Drive (~100MB, one-time)..." spinner; every run after that loads the
local copy instantly.

If you'd rather download it yourself (e.g. to commit it to a different
storage backend, or if the automatic download fails):

1. Open the [Drive link](https://drive.google.com/file/d/1nfir5pQWlzSNOuy44InBiK9Lp9CyR-EG/view?usp=drive_link) above.
2. Click **Download**.
3. Save the file as `all_models.pkl` in the same folder as `app.py`.

> The Drive file must stay shared as **"Anyone with the link"** for the
> automatic download in `app.py` to work — if it's switched to restricted
> access, the download step will fail with a permissions error.

## Setup

```bash
git clone <your-repo-url>
cd fake-news-detection-system
pip install -r requirements.txt
```

## Run

```bash
streamlit run app.py
```

On first launch, if `all_models.pkl` isn't found locally, the app downloads
it from Google Drive automatically (~100MB, one-time). After that it loads
from disk on every subsequent run.

## What's in the dashboard

- **Check a headline or article** — paste text, pick a model, get a
  Real/Fake prediction with a confidence chart. Includes one-click "Try a
  real/fake example" buttons, and a low-signal warning for very short or
  out-of-vocabulary inputs.
- **Model Performance** — accuracy/precision/recall/F1 comparison across
  all trained models and feature versions.
- **Confusion Matrix & Class Balance** — confusion matrix and test-set
  class distribution for any selected model.
- **About the Model & Data** — what the model was trained on, what kinds
  of text it predicts well vs. poorly, and why.

## Model limitations (read this before trusting a prediction)

This model is a **writing-style classifier**, not a fact-checker. It learned
to distinguish the phrasing/formatting typical of wire-service journalism
from that of flagged fake-news sites — it has no knowledge of real-world
facts or current events, and can't verify whether a specific claim is true.

- ✅ Works best on full news-length paragraphs, formal third-person
  wire-service style, US politics / world-news topics.
- ❌ Struggles on short one-line claims, topics/names outside its training
  vocabulary, and anything that actually requires fact-checking (e.g. "is
  this person alive").

See the **About the Model & Data** page in the app for details and examples.
