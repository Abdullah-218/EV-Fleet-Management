# EV Fleet Prediction API

A machine learning-powered REST API for electric vehicle fleet management. It exposes three predictive models covering the key operational health indicators of an EV: **expected range**, **battery degradation rate**, and **remaining useful life (RUL)**.

---

## Project Overview

Managing an EV fleet means answering three questions on every trip:

| Question | Model |
|---|---|
| How far can this vehicle travel right now? | Range Prediction |
| How fast is the battery degrading? | Battery Degradation |
| How many days before the next major service? | Remaining Useful Life |

The API is built with **FastAPI** and loads pre-trained `scikit-learn` model packages from the `models/` directory. A standalone CLI runner (`predict_ev_objectives.py`) is also included for testing without the server.

---

## Project Structure

```
New_EV_FLEET/
├── app.py                      # FastAPI server (main entry point)
├── predict_ev_objectives.py    # Standalone CLI predictor
├── new_model.ipynb             # Model training notebook
├── models/
│   ├── range_model.pkl         # Range prediction model package
│   ├── battery_model.pkl       # Battery degradation model package
│   ├── rul_model.pkl           # RUL prediction model package
│   └── encoders.pkl            # Feature encoders
└── dataset/
    ├── telemetry_merged1.csv   # Training telemetry dataset
    ├── driver_master.csv       # Driver metadata
    └── vehicle_master.csv      # Vehicle metadata
```

---

## Setup & Running

### Prerequisites

```bash
pip install fastapi uvicorn joblib pandas scikit-learn pydantic
```

### Start the API server

```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

The API will be available at `http://localhost:8000`.  
Interactive Swagger docs are auto-generated at `http://localhost:8000/docs`.

### Run the standalone CLI predictor

```bash
python3 predict_ev_objectives.py
```

Follow the prompts to select an objective and enter vehicle/trip data. Option `4` runs a demo with preset realistic values.

---

## API Endpoints

### `GET /`
Returns service status and a list of all available endpoints.

### `GET /health`
Returns the loaded model names and server health status.

---

### `POST /predict/range`

Predicts the expected driving range based on current battery state and driving conditions.

**Request Body**

| Field | Type | Description | Example |
|---|---|---|---|
| `odometer_km` | float | Total distance driven (km) | `55000` |
| `driving_mode` | string | `"city"` or `"highway"` | `"highway"` |
| `battery_pct` | float | Current battery state of charge (0–100) | `65` |
| `battery_health_pct` | float | Battery health percentage (0–100) | `94` |
| `max_range_km` | float | Vehicle's rated maximum range (km) | `548` |
| `energy_consumed_kwh` | float | Energy consumed in the recent trip (kWh) | `8.2` |

**Response**

```json
{
  "objective": "range_prediction",
  "model": "physical_formula",
  "range_km_physical": 335.18,
  "range_km_adjusted": 335.18,
  "unit": "km"
}
```

| Field | Description |
|---|---|
| `range_km_physical` | Hard physics-constrained range based on SOC × health × max range |
| `range_km_adjusted` | Practical adjusted range factoring in driving mode and energy usage |

---

### `POST /predict/battery-degradation`

Predicts the battery degradation rate in percentage per 100 charge cycles.

**Request Body**

| Field | Type | Description | Example |
|---|---|---|---|
| `odometer_km` | float | Total distance driven (km) | `72000` |
| `battery_cycles` | float | Total charge/discharge cycles completed | `650` |
| `battery_health_pct` | float | Battery health percentage (0–100) | `91` |
| `battery_temperature_c` | float | Battery temperature in Celsius | `34` |
| `battery_capacity_kwh` | float | Battery pack capacity (kWh) | `60.5` |
| `energy_consumed_kwh` | float | Energy consumed in the recent trip (kWh) | `10.5` |
| `avg_speed_kmph` | float | Average trip speed (km/h) | `48` |
| `soc_start_pct` | float | State of charge at trip start (0–100) | `72` |
| `soc_end_pct` | float | State of charge at trip end (0–100) | `48` |
| `vehicle_age_months` | float | Vehicle age in months | `36` |

**Response**

```json
{
  "objective": "battery_degradation",
  "model": "GradientBoostingRegressor",
  "battery_degradation_pct_per_100_cycles": 0.4821,
  "soc_drop_pct": 24.0
}
```

| Field | Description |
|---|---|
| `battery_degradation_pct_per_100_cycles` | Predicted health loss per 100 charge cycles |
| `soc_drop_pct` | State-of-charge drop observed during the trip |

---

### `POST /predict/rul`

Predicts the Remaining Useful Life of the vehicle — the estimated number of days before the next major maintenance event.

**Request Body**

| Field | Type | Description | Example |
|---|---|---|---|
| `odometer_km` | float | Total distance driven (km) | `98000` |
| `battery_health_pct` | float | Battery health percentage (0–100) | `88` |
| `brake_wear_pct` | float | Brake wear percentage (0–100) | `42` |
| `suspension_health_pct` | float | Suspension health percentage (0–100) | `78` |
| `days_since_last_service` | float | Days elapsed since last service | `145` |
| `vehicle_age_months` | float | Vehicle age in months | `60` |
| `overspeed_events` | float | Number of overspeed events in recent trip | `2` |
| `hard_braking_events` | float | Number of hard braking events in recent trip | `2` |
| `rapid_acceleration_events` | float | Number of rapid acceleration events in recent trip | `1` |
| `idle_time_minutes` | float | Total idle time in recent trip (minutes) | `12` |
| `driver_experience_years` | float | Driver's years of experience | `4` |
| `driver_score` | float | Driver behaviour score (0–100) | `72` |

**Response**

```json
{
  "objective": "remaining_useful_life",
  "model": "GradientBoostingRegressor",
  "rul_days": 312.5,
  "aggression_score": 26.4,
  "mechanical_health": 72.0
}
```

| Field | Description |
|---|---|
| `rul_days` | Estimated days remaining before next major service |
| `aggression_score` | Derived driving aggression score (0–100, lower is better) |
| `mechanical_health` | Composite brake + suspension health score (0–100) |

---

### `POST /predict/all`

Runs all three predictions in a single request.

**Request Body**

Wraps all three request bodies under their respective keys:

```json
{
  "range": { ...RangeRequest fields... },
  "battery_degradation": { ...BatteryDegradationRequest fields... },
  "rul": { ...RulRequest fields... }
}
```

**Response**

```json
{
  "range": { ...range response... },
  "battery_degradation": { ...battery degradation response... },
  "rul": { ...rul response... }
}
```

---

## Further Development

### Retrain the models
Open `new_model.ipynb` in Jupyter and re-run the cells with updated data. The notebook reads from `dataset/` and saves model packages to `models/`. Both paths can be overridden via environment variables:

```bash
DATASET_DIR=my_data MODEL_DIR=my_models jupyter notebook new_model.ipynb
```

No changes to `app.py` are needed as long as the feature names remain the same.

### Add a new prediction objective
1. Train a new model in the notebook and save it as `models/<new_model>.pkl` using the same package format: `{"model": ..., "scaler": ..., "features": [...], "name": "..."}`.
2. Add a Pydantic request model class in `app.py`.
3. Add a `predict_<objective>_value()` function following the existing pattern.
4. Register a new `@app.post("/predict/<objective>")` route.

### Connect live telemetry
Replace the manual JSON inputs with a telemetry pipeline (MQTT, Kafka, or a database query) that feeds real-time vehicle data to the `/predict/all` endpoint.

### Containerise the service

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY . .
RUN pip install fastapi uvicorn joblib pandas scikit-learn pydantic
EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
```

Build and run:

```bash
docker build -t ev-fleet-api .
docker run -p 8000:8000 ev-fleet-api
```

### Environment variables

| Variable | Default | Description |
|---|---|---|
| `MODEL_DIR` | `models` | Path to the directory containing `.pkl` model packages |
| `DATASET_DIR` | `dataset` | Path to the directory containing training CSV files |

---

## Tech Stack

- **Python 3.10+**
- **FastAPI** — REST API framework
- **scikit-learn** — ML model training and inference
- **pandas** — Feature engineering
- **joblib** — Model serialisation
- **pydantic** — Request validation
- **uvicorn** — ASGI server
