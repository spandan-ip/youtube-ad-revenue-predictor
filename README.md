# YouTube Ad Revenue Predictor

An end-to-end machine learning project that predicts estimated YouTube ad
revenue from video engagement and audience metrics, served via a FastAPI
REST API.

## Overview

- Trained and compared **Linear Regression**, **Random Forest**, and
  **XGBoost** on a 100K-row dataset of YouTube video statistics
- Applied log-transformation to correct skewed feature/target
  distributions before training
- Selected the best-performing model based on test-set R²/RMSE
- Achieved **R² ≈ 0.97, MAE ≈ $82** on held-out test data
- Served the trained model through a FastAPI backend with input validation

## Project structure

```
YOUTUBE_AD_REVENUE_PROJ/
├── app/
│   └── app.py                 # FastAPI app — serves predictions
├── data/
│   └── youtube_ad_revenue_dataset.csv
├── model/
│   └── model.py                # Training script (trains + saves the model)
├── model.pkl                   # Trained model (best of the 3 compared)
├── model_columns.pkl           # Exact feature column order the model expects
├── requirements.txt
└── README.md
```

## Features used

| Feature | Description |
|---|---|
| views | Total video views |
| watch_time_hours | Total hours watched |
| average_view_duration_sec | Average watch time per viewer |
| ctr_percent | Click-through rate (%) |
| impressions | Number of thumbnail impressions |
| subscribers_gained | New subscribers from the video |
| likes / comments / shares | Engagement counts |
| category | Content category (Gaming, Education, Music, etc.) |
| country | Primary audience country |
| cpm_usd | Cost per 1,000 monetized views |
| monetized_playbacks | Number of monetized views |

**Target:** `estimated_ad_revenue_usd`

## Setup

```bash
# 1. Create and activate a virtual environment
python -m venv .venv
.venv\Scripts\activate        # Windows
source .venv/bin/activate     # Mac/Linux

# 2. Install dependencies
pip install -r requirements.txt

# 3. (Optional) Retrain the model
python model/model.py

# 4. Run the API
cd app
uvicorn app:app --reload
```

## Using the API

- **Swagger UI:** `http://127.0.0.1:8000/docs` — interactive docs to test
  the `/predict` endpoint directly in your browser
- **Health check:** `http://127.0.0.1:8000/health`

**Example request to `/predict`:**
```json
{
  "views": 500000,
  "watch_time_hours": 45000,
  "average_view_duration_sec": 320,
  "ctr_percent": 5.2,
  "impressions": 8000000,
  "subscribers_gained": 1200,
  "likes": 25000,
  "comments": 1800,
  "shares": 900,
  "category": "Tech",
  "country": "US",
  "cpm_usd": 4.5,
  "monetized_playbacks": 380000
}
```

**Example response:**
```json
{
  "predicted_ad_revenue_usd": 3703.81
}
```

## Model performance

| Model | Test R² | MAE | RMSE |
|---|---|---|---|
| Linear Regression | 0.9700 | $81.74 | $185.79 |
| Random Forest | 0.9499 | $52.25 | $240.14 |
| XGBoost | 0.9571 | $49.88 | $222.20 |

**Linear Regression** was selected as the final model — it had the best
Test R² and RMSE with the least overfitting, despite the tree-based
models having slightly lower MAE.

## Key learnings

- Log-transforming skewed features (views, watch time, engagement counts)
  was essential — without it, Linear Regression badly underpredicted
  high-revenue videos due to squared-error loss being dominated by the
  dense cluster of low-revenue rows
- Found and fixed a **data leakage bug** where the test set was
  accidentally used for XGBoost's early stopping validation
- Found and fixed a **one-hot encoding bug** at inference time —
  `drop_first=True` behaves differently on a single-row request than on
  a full training set, silently zeroing out category/country signal

## Tech stack

`Python` · `pandas` · `NumPy` · `scikit-learn` · `XGBoost` · `FastAPI` ·
`Pydantic` · `Git`

## Limitations

- Trained on synthetic data — statistically realistic, but not scraped
  from real YouTube data
- Linear Regression extrapolates linearly for inputs far outside the
  training range (e.g. viral outliers), which may produce imprecise
  estimates at extreme scales
- No time dimension — predictions assume steady-state metrics, not
  trends over time
