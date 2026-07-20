from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import pandas as pd
import numpy as np

# create fastapi app
app = FastAPI()

# load traing model
print("loading model...")

with open("../model.pkl", "rb") as file:
    model = pickle.load(file)

with open("../model_columns.pkl", "rb") as file:
    model_columns = pickle.load(file)

print("model loaded succefully.")

# request body
class video_Datas(BaseModel):
    views: int                       
    watch_time_hours: float
    average_view_duration_sec: float
    ctr_percent: float
    impressions: int
    subscribers_gained: int
    likes: int
    comments: int
    shares: int
    category: str
    country: str
    cpm_usd: float
    monetized_playbacks: int

# Home Route
@app.get("/")
def home():
    return{
        "message": "youtube ad revenue prediction api is running"
    }

# Prediction Route
@app.post("/predict")
def predict(data: video_Datas):
    input_dict = data.model_dump()
    df = pd.DataFrame([input_dict])

    skewed_cols = ['views', 'impressions', 'watch_time_hours', 'likes',
                   'comments', 'shares', 'monetized_playbacks', 'subscribers_gained']
    for col in skewed_cols:
        df[col] = np.log1p(df[col])

    # FIX: no drop_first here — reindex below handles alignment correctly
    df_encoded = pd.get_dummies(df, columns=["category", "country"])

    df_final = df_encoded.reindex(columns=model_columns, fill_value=0)

    predicted_log = model.predict(df_final)
    predicted_revenue = np.expm1(predicted_log[0])
    predicted_revenue = max(float(predicted_revenue), 0.0)  # can't be negative

    estimated_revenue_usd = round(predicted_revenue, 2)
    return {
        "predicted_ad_revenue_usd": estimated_revenue_usd,
        "input_data": input_dict
    }
# HEALTH CHECK ROUTE
@app.get("/health")
def health_check():
    return {
        "status": "API is healthy",
        "model_loaded": True
    }
