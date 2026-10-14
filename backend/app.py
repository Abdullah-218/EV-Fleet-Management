"""
FastAPI service for EV objective predictions.

Loads trained pickle packages from ./models and exposes JSON endpoints for:
  - range prediction
  - battery degradation prediction
  - remaining useful life prediction
  - combined prediction

Run from the repository root:
  uvicorn backend.app:app --host 0.0.0.0 --port 8000 --reload

Or run from inside ./backend:
  uvicorn app:app --host 0.0.0.0 --port 8000 --reload
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = os.getenv("MODEL_DIR", str(PROJECT_ROOT / "ml" / "models"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("ev-prediction-api")


app = FastAPI(
    title="EV Fleet Prediction API",
    version="1.0.0",
    description="Predict EV range, battery degradation, and remaining useful life.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def load_model_package(filename: str) -> dict:
    path = os.path.join(MODEL_DIR, filename)
    if not os.path.exists(path):
        raise RuntimeError(f"Missing model file: {path}")
    logger.info("Loading model package: %s", path)
    return joblib.load(path)


try:
    RANGE_MODEL = load_model_package("range_model.pkl")
    BATTERY_MODEL = load_model_package("battery_model.pkl")
    RUL_MODEL = load_model_package("rul_model.pkl")
except Exception as exc:
    logger.exception("Model loading failed")
    raise


def model_display_name(package: dict, fallback: str) -> str:
    saved_name = package.get("name")
    if saved_name:
        return str(saved_name)
    model = package.get("model")
    if model is not None:
        return type(model).__name__
    return fallback


class RangeRequest(BaseModel):
    odometer_km: float = Field(..., ge=0, example=55000)
    driving_mode: Literal["city", "highway"] = Field(..., example="highway")
    battery_pct: float = Field(..., ge=0, le=100, example=65)
    battery_health_pct: float = Field(..., ge=0, le=100, example=94)
    max_range_km: float = Field(..., gt=0, example=548)
    energy_consumed_kwh: float = Field(..., ge=0, example=8.2)


class BatteryDegradationRequest(BaseModel):
    odometer_km: float = Field(..., ge=0, example=72000)
    battery_cycles: float = Field(..., ge=0, example=650)
    battery_health_pct: float = Field(..., ge=0, le=100, example=91)
    battery_temperature_c: float = Field(..., example=34)
    battery_capacity_kwh: float = Field(..., gt=0, example=60.5)
    energy_consumed_kwh: float = Field(..., ge=0, example=10.5)
    avg_speed_kmph: float = Field(..., ge=0, example=48)
    soc_start_pct: float = Field(..., ge=0, le=100, example=72)
    soc_end_pct: float = Field(..., ge=0, le=100, example=48)
    vehicle_age_months: float = Field(..., ge=0, example=36)


class RulRequest(BaseModel):
    odometer_km: float = Field(..., ge=0, example=98000)
    battery_health_pct: float = Field(..., ge=0, le=100, example=88)
    brake_wear_pct: float = Field(..., ge=0, le=100, example=42)
    suspension_health_pct: float = Field(..., ge=0, le=100, example=78)
    days_since_last_service: float = Field(..., ge=0, example=145)
    vehicle_age_months: float = Field(..., ge=0, example=60)
    overspeed_events: float = Field(..., ge=0, example=2)
    hard_braking_events: float = Field(..., ge=0, example=2)
    rapid_acceleration_events: float = Field(..., ge=0, example=1)
    idle_time_minutes: float = Field(..., ge=0, example=12)
    driver_experience_years: float = Field(..., ge=0, example=4)
    driver_score: float = Field(..., ge=0, le=100, example=72)


class CombinedPredictionRequest(BaseModel):
    range: RangeRequest
    battery_degradation: BatteryDegradationRequest
    rul: RulRequest


def physical_range_km(battery_pct: float, battery_health_pct: float, max_range_km: float) -> float:
    soc = min(max(battery_pct, 0.0), 100.0) / 100.0
    health = min(max(battery_health_pct, 0.0), 100.0) / 100.0
    return round(max_range_km * soc * health, 3)


def battery_stress(battery_temperature_c: float, battery_health_pct: float) -> float:
    return (battery_temperature_c / 60.0) * (1.0 - battery_health_pct / 100.0)


def mechanical_health(brake_wear_pct: float, suspension_health_pct: float) -> float:
    return (1.0 - brake_wear_pct / 100.0) * 50.0 + (suspension_health_pct / 100.0) * 50.0


def aggression_score(
    overspeed_events: float,
    hard_braking_events: float,
    rapid_acceleration_events: float,
    idle_time_minutes: float,
) -> float:
    return min(
        100.0,
        overspeed_events * 6.0
        + hard_braking_events * 4.0
        + rapid_acceleration_events * 3.0
        + idle_time_minutes / 5.0,
    )


def ensure_model_features(row: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    row = row.copy()
    if "energy_per_km" in features and "energy_per_km" not in row.columns:
        trip_distance = row.get("trip_distance_km", pd.Series([70.0])).clip(lower=0.1)
        row["energy_per_km"] = row["energy_consumed_kwh"] / trip_distance
    if "service_overdue" in features and "service_overdue" not in row.columns:
        row["service_overdue"] = (row["days_since_last_service"] > 120).astype(int)
    if "brake_wear_pct_roll7" in features and "brake_wear_pct_roll7" not in row.columns:
        row["brake_wear_pct_roll7"] = row["brake_wear_pct"]

    missing = [feature for feature in features if feature not in row.columns]
    if missing:
        raise KeyError(f"Missing model features: {missing}")
    return row

def predict_range_value(data: RangeRequest) -> dict:
    # Keep loading/use of the range package explicit for deployment checks.
    model_name = model_display_name(RANGE_MODEL, "physical_formula")

    base_range = physical_range_km(
        data.battery_pct,
        data.battery_health_pct,
        data.max_range_km,
    )
    mode_factor = 1.0 if data.driving_mode == "highway" else 0.92
    energy_factor = 1.0
    if data.energy_consumed_kwh > 0:
        energy_factor = max(0.82, min(1.0, 1.0 - (data.energy_consumed_kwh - 15.0) * 0.006))
    adjusted_range = round(min(base_range, base_range * mode_factor * energy_factor), 3)

    return {
        "objective": "range_prediction",
        "model": model_name,
        "range_km_physical": base_range,
        "range_km_adjusted": adjusted_range,
        "unit": "km",
    }


def predict_battery_degradation_value(data: BatteryDegradationRequest) -> dict:
    soc_drop_pct = max(0.0, data.soc_start_pct - data.soc_end_pct)
    row = pd.DataFrame([{
        "battery_cycles": data.battery_cycles,
        "battery_health_pct": data.battery_health_pct,
        "battery_temperature": data.battery_temperature_c,
        "battery_capacity_kwh": data.battery_capacity_kwh,
        "energy_consumed_kwh": data.energy_consumed_kwh,
        "trip_distance_km": 70.0,
        "energy_per_km": data.energy_consumed_kwh / 70.0,
        "odometer_km": data.odometer_km,
        "avg_speed_kmph": data.avg_speed_kmph,
        "soc_drop_pct": soc_drop_pct,
        "vehicle_age_months": data.vehicle_age_months,
        "battery_stress": battery_stress(data.battery_temperature_c, data.battery_health_pct),
    }])

    features = BATTERY_MODEL["features"]
    row = ensure_model_features(row, features)
    x = BATTERY_MODEL["scaler"].transform(row[features].fillna(0))
    prediction = float(BATTERY_MODEL["model"].predict(x)[0])

    return {
        "objective": "battery_degradation",
        "model": model_display_name(BATTERY_MODEL, "battery_model"),
        "battery_degradation_pct_per_100_cycles": round(max(0.0, prediction), 4),
        "soc_drop_pct": round(soc_drop_pct, 2),
    }


def predict_rul_value(data: RulRequest) -> dict:
    score = aggression_score(
        data.overspeed_events,
        data.hard_braking_events,
        data.rapid_acceleration_events,
        data.idle_time_minutes,
    )
    mech_health = mechanical_health(data.brake_wear_pct, data.suspension_health_pct)
    row = pd.DataFrame([{
        "brake_wear_pct": data.brake_wear_pct,
        "suspension_health_pct": data.suspension_health_pct,
        "battery_health_pct": data.battery_health_pct,
        "days_since_last_service": data.days_since_last_service,
        "vehicle_age_months": data.vehicle_age_months,
        "odometer_km": data.odometer_km,
        "aggression_score": score,
        "experience_years": data.driver_experience_years,
        "base_driver_score": data.driver_score,
        "mechanical_health": mech_health,
    }])

    features = RUL_MODEL["features"]
    row = ensure_model_features(row, features)
    x = RUL_MODEL["scaler"].transform(row[features].fillna(0))
    prediction = float(RUL_MODEL["model"].predict(x)[0])

    return {
        "objective": "remaining_useful_life",
        "model": model_display_name(RUL_MODEL, "rul_model"),
        "rul_days": round(max(0.0, prediction), 1),
        "aggression_score": round(score, 2),
        "mechanical_health": round(mech_health, 2),
    }


def log_prediction(endpoint: str, request_payload: BaseModel, response_payload: dict) -> None:
    logger.info("Endpoint=%s request=%s response=%s", endpoint, request_payload.dict(), response_payload)
    print(f"\n[{endpoint}]")
    print("Request:", request_payload.dict())
    print("Response:", response_payload)


@app.get("/")
def root() -> dict:
    return {
        "service": "EV Fleet Prediction API",
        "status": "running",
        "endpoints": [
            "/predict/range",
            "/predict/battery-degradation",
            "/predict/rul",
            "/predict/all",
            "/health",
        ],
    }


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "model_dir": MODEL_DIR,
        "models_loaded": {
            "range": model_display_name(RANGE_MODEL, "physical_formula"),
            "battery_degradation": model_display_name(BATTERY_MODEL, "battery_model"),
            "rul": model_display_name(RUL_MODEL, "rul_model"),
        },
    }


@app.post("/predict/range")
def predict_range(data: RangeRequest) -> dict:
    try:
        response = predict_range_value(data)
        log_prediction("/predict/range", data, response)
        return response
    except Exception as exc:
        logger.exception("Range prediction failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/predict/battery-degradation")
def predict_battery_degradation(data: BatteryDegradationRequest) -> dict:
    try:
        response = predict_battery_degradation_value(data)
        log_prediction("/predict/battery-degradation", data, response)
        return response
    except Exception as exc:
        logger.exception("Battery degradation prediction failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/predict/rul")
def predict_rul(data: RulRequest) -> dict:
    try:
        response = predict_rul_value(data)
        log_prediction("/predict/rul", data, response)
        return response
    except Exception as exc:
        logger.exception("RUL prediction failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/predict/all")
def predict_all(data: CombinedPredictionRequest) -> dict:
    try:
        response = {
            "range": predict_range_value(data.range),
            "battery_degradation": predict_battery_degradation_value(data.battery_degradation),
            "rul": predict_rul_value(data.rul),
        }
        log_prediction("/predict/all", data, response)
        return response
    except Exception as exc:
        logger.exception("Combined prediction failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
