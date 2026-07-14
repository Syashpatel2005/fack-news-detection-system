# Fake News Detector — Streamlit Dashboard

Companion dashboard for `fake_news_detector_balanced.ipynb`. Works standalone
with built-in demo data, or with your own files uploaded from the sidebar.

## Run it

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Fast model loading

Drop your `all_models.pkl` in the **same folder as `app.py`**. On startup the
app auto-loads it from disk and caches it (`st.cache_resource`), so it's read
once and reused instantly for every page and prediction — no re-uploading,
no re-parsing on each interaction. You can still override it anytime from
the sidebar's "…or upload/override all_models.pkl" uploader (e.g. to try a
newer pickle without restarting the app).

## Sidebar inputs (all optional)

| Input | What it enables |
|---|---|
| `all_models.pkl` next to `app.py` (auto-loaded, fastest) or uploaded | Real model metrics, confusion matrices, and live predictions using your trained NB / KNN / LR models |
| `preprocessed_fake_real_news.csv` (from the notebook) | Real class-imbalance stats, EDA, word frequencies on your actual data |
| `Fake.csv` + `True.csv` (raw Kaggle files) | Same as above, computed from scratch |

If nothing is provided, the dashboard uses a synthetic 3:1 imbalanced demo
dataset so every section is still explorable.

## Pages

1. **Overview** — headline metrics, class split, subject mix
2. **Class Imbalance** — counts, ratio, proportions, why it matters
3. **Imbalance Handling Lab** — pick undersampling / SMOTE / class-weighting
   and see the training-set class counts before vs. after
4. **Exploratory Analysis** — word-count distributions, top words per class,
   subject × label crosstab
5. **Train & Evaluate** — real results table + confusion matrices from your
   `all_models.pkl`, or train lightweight demo models on the fly if you
   haven't uploaded one
6. **Live Prediction** — paste text, get a Fake/Real call with class
   probabilities (requires `all_models.pkl`)
