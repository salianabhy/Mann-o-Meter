import os
import warnings
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import Literal
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

# Filter benign scikit-learn unpickle warnings
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "Mental_Health_Model.pkl")

model = joblib.load(MODEL_PATH)
top_countries = ['Other','India','USA','Canada','Australia','UK','Germany','Mexico','Turkey','France']

app = FastAPI(title="Mental Health Score API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Student data input schema
class StudentData(BaseModel):
    age                     : int = Field(..., ge=10, le=100)
    gender                  : Literal['Male', 'Female']
    country                 : str
    academic_level          : Literal['Undergraduate', 'Graduate', 'High School']
    most_used_platform      : Literal['Facebook', 'LinkedIn', 'Instagram', 'Snapchat','Twitter','YouTube', 'TikTok', 'LINE', 'KakaoTalk', 'VKontakte', 'WhatsApp','WeChat']
    purpose_of_use          : Literal['Networking', 'Education', 'Entertainment', 'News']
    avg_daily_usage_hours   : float = Field(..., ge=0, le=24)
    daily_unlocks           : int   = Field(..., ge=0)
    study_hours             : float = Field(..., ge=0, le=24)
    physical_activity_hours : float = Field(..., ge=0, le=24)
    sleep_hours_per_night   : float = Field(..., ge=0, le=24)
    stress_level            : Literal['Medium', 'Low', 'Very High', 'High']


# Describe what we send back
class PredictionResponse(BaseModel):
    predicted_mental_health_score: float


@app.get('/api/health')
def health_check():
    return {"status": "ok", "message": "Mental Health Score Prediction API is running"}


@app.get('/greet')
def greet():
    return {"message": "Welcome to Sheryians AI School Guys"}


def calibrate_score(raw_pred: float) -> float:
    """
    Random Forest ensemble tree averaging naturally contracts continuous predictions
    into ~[4.8, 8.0] around the training mean (6.2). This calibration maps the predictions
    smoothly across the full 0–10 scale while strictly preserving all learned feature weights:
      - Raw ~4.80 (severe strain)   -> ~1.5 - 2.5
      - Raw ~6.10 (median/average)  -> ~5.5 - 6.2
      - Raw ~7.95 (optimal habits)  -> ~8.8 - 9.8
    """
    raw = float(raw_pred)
    if raw <= 6.10:
        scaled = 1.5 + (raw - 4.80) * (4.5 / 1.30)
    else:
        scaled = 6.0 + (raw - 6.10) * (3.8 / 1.85)
    return round(float(np.clip(scaled, 1.0, 9.9)), 2)


@app.post('/predict', response_model=PredictionResponse)
def predict(data: StudentData):
    country_group = data.country if data.country in top_countries else "Other"

    input_row = pd.DataFrame([{
        'Age'                       : data.age,
        'Gender'                    : data.gender,
        'Country'                   : data.country,
        'Academic_Level'            : data.academic_level,
        'Most_Used_Platform'        : data.most_used_platform,
        'Purpose_Of_Use'            : data.purpose_of_use,
        'Avg_Daily_Usage_Hours'     : data.avg_daily_usage_hours,
        'Daily_Unlocks'             : data.daily_unlocks,
        'Study_Hours'               : data.study_hours,
        'Physical_Activity_Hours'   : data.physical_activity_hours,
        'Sleep_Hours_Per_Night'     : data.sleep_hours_per_night,
        'Stress_Level'              : data.stress_level,
        'Grouped_country'           : country_group
    }])

    raw_prediction = model.predict(input_row)[0]
    final_score = calibrate_score(raw_prediction)
    return PredictionResponse(predicted_mental_health_score=final_score)


# Mount current directory to serve static assets (index.html, style.css, script.js)
app.mount("/", StaticFiles(directory=BASE_DIR, html=True), name="static")


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)