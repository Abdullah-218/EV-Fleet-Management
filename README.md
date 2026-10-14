# EV Fleet Management and Predictive Maintenance

## Project Report / Technical Documentation

This project is a machine-learning-powered electric vehicle (EV) fleet management
system. It estimates three operational objectives that are useful to a fleet
operator:

1. **Current driving range** - how far an EV can reasonably travel with its
   present state of charge and battery health.
2. **Battery degradation rate** - the expected battery health loss per 100
   charge/discharge cycles.
3. **Remaining useful life (RUL)** - the estimated number of days before the
   vehicle requires a major maintenance intervention.

The project combines data preparation, feature engineering, model comparison,
model serialization, a FastAPI inference service, request validation, and a
standalone command-line demo. It is designed as a decision-support system:
predictions can help schedule charging, identify vehicles needing inspection,
and plan preventive maintenance before a breakdown occurs.

> **Important implementation fact:** three model packages are saved in
> `ml/models/`, but range inference deliberately uses a physics-constrained
> formula rather than the Random Forest output. The Random Forest is retained
> in the range package for training comparison, deployment checks, and future
> experimentation. Battery degradation and RUL use their serialized regressors
> during inference.

---

## 1. Problem Statement

EV fleet operators must make decisions using changing battery, vehicle, trip,
driver, and maintenance conditions. A single static specification such as the
manufacturer's maximum range is not sufficient because:

- state of charge changes continuously;
- battery capacity reduces with age, cycles, temperature, and usage;
- city driving and high energy consumption reduce practical range;
- aggressive driving increases mechanical and energy stress;
- brake, suspension, battery, mileage, and service history jointly affect
  maintenance risk.

The project converts these measurable signals into three predictions:

| Business question | Output | Operational use |
|---|---|---|
| Can this vehicle complete its next trip? | Adjusted range in km | Trip assignment and charging decisions |
| Is the battery aging faster than expected? | Degradation percentage per 100 cycles | Battery monitoring and replacement planning |
| Which vehicles need attention first? | Remaining useful life in days | Preventive maintenance scheduling |

The predictions are estimates, not safety certifications. A vehicle should
still be inspected according to the manufacturer's maintenance and safety
requirements.

---

## 2. Objectives and Scope

### Objectives

- Build three regression solutions for EV fleet operations.
- Use physically meaningful derived features instead of only raw columns.
- Compare multiple candidate regressors and select the best candidate using
  holdout performance.
- Persist the selected models and their preprocessing information.
- Expose predictions through a typed HTTP API.
- Provide a CLI path for local demonstrations without running a web server.
- Keep the range result within an explicit physical boundary.

### In scope

- Batch training from `ml/dataset/telemetry_merged1.csv`.
- Vehicle-level train/test separation.
- Inference for one request at a time.
- Combined inference through `/predict/all`.
- Basic request validation, logging, health checks, and interactive API docs.

### Out of scope

- Direct control of a vehicle, charger, battery-management system, or braking
  system.
- Streaming telemetry ingestion.
- Authentication, authorization, rate limiting, or persistent prediction
  storage.
- Automatic retraining in production.
- A formal battery safety or warranty decision.

---

## 3. System Architecture

```text
CSV telemetry (ml/dataset)
     |
     v
ml/new_model.ipynb
  - clean/normalize columns
  - derive features and targets
  - split by vehicle
  - scale features
  - compare regressors
  - serialize model + scaler + feature list
     |
     v
ml/models/*.pkl
     |
     +--------------------+
     |                    |
     v                    v
backend/app.py       backend/predict_ev_objectives.py
FastAPI service      CLI predictor
     |
     v
JSON predictions
     |
     v
frontend/ (React dashboard)
```

At API startup, `backend/app.py` loads:

- `range_model.pkl`
- `battery_model.pkl`
- `rul_model.pkl`

Each serialized package contains the estimator, a `StandardScaler`, and the
feature names expected by that estimator. The API constructs a one-row
`pandas.DataFrame`, derives any required values, orders columns using the
saved feature list, scales the row, and performs prediction.

---

## 4. Repository Structure

```text
New_EV_FLEET/
├── backend/                        # REST API and CLI inference
│   ├── app.py                      # FastAPI application, schemas and routes
│   ├── predict_ev_objectives.py    # Interactive CLI and demo inference runner
│   └── requirements.txt            # API runtime dependencies
├── ml/                             # Training workspace and model artifacts
│   ├── new_model.ipynb             # Data prep, training, comparison, export
│   ├── requirements.txt            # Notebook and training dependencies
│   ├── models/
│   │   ├── range_model.pkl
│   │   ├── battery_model.pkl
│   │   ├── rul_model.pkl
│   │   └── encoders.pkl
│   └── dataset/
│       ├── telemetry_merged1.csv
│       ├── driver_master.csv
│       └── vehicle_master.csv
├── frontend/                       # React/Vite dashboard
│   ├── src/
│   ├── package.json
│   └── README.md
├── ev_fleet_architecture.png
├── ev_fleet_architecture.svg
├── .gitignore
└── README.md
```

The three folders have distinct responsibilities:

| Folder | Responsibility | Depends on |
|---|---|---|
| `ml/` | Dataset storage, feature engineering, training, model export | pandas, scikit-learn, XGBoost, LightGBM |
| `backend/` | Loading artifacts and serving predictions over HTTP/CLI | `ml/models/` artifacts |
| `frontend/` | Operator interface and visualization | `backend/` HTTP API |

The notebook currently trains from `telemetry_merged1.csv` only. The driver and
vehicle master CSVs are retained as project data assets, but they are not
directly loaded by the current notebook training path.

### Path resolution

`backend/app.py` and `backend/predict_ev_objectives.py` resolve the project root
from their own file location:

```python
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = os.getenv("MODEL_DIR", str(PROJECT_ROOT / "ml" / "models"))
```

This means the API and CLI work whether they are launched from the repository
root or from inside `backend/`. The notebook applies the same idea using its
working directory, so it can be launched from the root or from `ml/`. All three
paths remain overridable with the `MODEL_DIR` and `DATASET_DIR` environment
variables.

---

## 5. Dataset and Data Preparation

The notebook reports **5,000 rows and 47 columns** after loading the telemetry
dataset. Rows are sorted by `vehicle_id` and `timestamp`. Duplicate merge
suffixes such as `_x` and `_y` are normalized for selected fields.

### Training and test strategy

The split is performed by vehicle, not by randomly mixing individual telemetry
rows:

- **Training vehicles:** `VH001` through `VH008`
- **Holdout test vehicles:** `VH009` and `VH010`

This is more realistic than a random row split because it tests whether the
learned relationships transfer to vehicles not used during training. The
reported metrics should therefore be interpreted as holdout performance on
these two vehicles, not as a guarantee for every future fleet.

### Derived features

The notebook creates the following features:

| Derived feature | Meaning / calculation |
|---|---|
| `vehicle_age_months` | `(2024 - manufacture_year) * 12` |
| `soc_drop_pct` | Positive difference between start and end SOC |
| `energy_per_km` | Energy consumed divided by trip distance |
| `aggression_score` | Weighted overspeed, hard braking, rapid acceleration and idle time score, capped at 100 |
| `mechanical_health` | 50% brake condition contribution plus 50% suspension health contribution |
| `battery_stress` | Temperature-related stress multiplied by battery-health loss fraction |
| `service_overdue` | `1` when days since service is greater than 120, otherwise `0` |
| `physical_range_km` | Maximum range multiplied by normalized SOC and normalized battery health |
| `battery_degradation_rate_redesigned` | Transferable fleet-level degradation formula |
| `*_roll7` | Seven-record vehicle-group rolling average for selected telemetry columns |

Categorical columns `brand`, `car_name`, `weather_condition`,
`traffic_density`, and `driver_type` are label encoded. The encoders are saved
to `ml/models/encoders.pkl` for reproducibility, although the current API request
schemas use only the numeric fields needed by its inference functions.

### Missing values and feature compatibility

Training fills missing feature values with zero after scaling preparation.
During inference, `ensure_model_features()` verifies that every saved feature
exists. It can derive:

- `energy_per_km` when it is required;
- `service_overdue` from `days_since_last_service`;
- `brake_wear_pct_roll7` from the current brake-wear value.

If a required feature is still missing, the request fails explicitly rather
than silently producing a prediction from an incomplete row.

---

## 6. The Three Predictive Models

### 6.1 Model 1 - Range Prediction

#### Purpose

Estimate the distance available from the vehicle's current battery state. This
is useful for trip feasibility, dispatching, charging recommendations, and
avoiding range-related service failures.

#### Training target

The notebook creates `physical_range_km`:

```text
SOC fraction    = clamp(soc_end_pct, 0, 100) / 100
Health fraction = clamp(battery_health_pct, 0, 100) / 100
physical range  = max_range_km × SOC fraction × Health fraction
```

The target is clipped so it remains between zero and the vehicle's rated
maximum range. The invariant is intentionally enforced:

```text
100% SOC and 100% health on a 500 km EV = 500 km
5% SOC and 100% health on a 500 km EV   = 25 km
```

#### Candidate features

The saved range package contains these ten features:

| Feature | Why it matters |
|---|---|
| `soc_end_pct` | Energy currently available in the pack |
| `battery_health_pct` | Available capacity relative to a healthy battery |
| `max_range_km` | Vehicle-specific rated range |
| `battery_capacity_kwh` | Energy storage capacity |
| `energy_per_km` | Recent efficiency or consumption intensity |
| `weather_condition` | Environmental effect on efficiency |
| `running_mode` | Operating or driving mode |
| `ambient_temperature` | Temperature impact on energy use |
| `soc_drop_pct` | Recent battery usage intensity |
| `brand` | Vehicle/platform-specific behavior |

#### Model selection

The notebook compares Ridge, Random Forest, XGBoost, and LightGBM. Random
Forest is selected for the serialized package:

- Estimator: `RandomForestRegressor`
- Saved package name: `Random Forest`
- Scaler: `StandardScaler`
- Holdout R²: **0.9907**
- Holdout MAE: **2.20 km**
- Relative MAE: **3.3%**

#### Runtime behavior

The current API does **not** call the saved Random Forest for the final range
number. It calculates:

```text
base_range = max_range_km × SOC fraction × health fraction
mode factor = 1.00 for highway, 0.92 for city
energy factor = bounded practical penalty based on recent energy consumption
adjusted range = min(base_range, base_range × mode factor × energy factor)
```

The energy factor is bounded between `0.82` and `1.0`, so practical adjustments
never increase the physically available range. The response returns both the
strict physical value and the adjusted operational value.

> The API accepts `driving_mode` (`city` or `highway`), while the training
> feature list uses notebook column names such as `running_mode`. This is
> intentional in the current deployment path because runtime range prediction
> uses the physical formula and does not transform the request into the saved
> model's categorical feature vector.

---

### 6.2 Model 2 - Battery Degradation Rate

#### Purpose

Estimate battery health loss per 100 charge cycles. This supports battery
warranty analysis, replacement planning, charging-policy evaluation, and early
detection of abnormal aging.

#### Target formula used during training

The notebook replaces the original vehicle-specific synthetic degradation label
with a transferable fleet-level target:

```text
cycle term    = 0.00055 × battery_cycles
age term      = 0.0040 × vehicle_age_months
temperature   = 0.0120 × max(battery_temperature - 25, 0)
depth-of-use  = 0.0035 × soc_drop_pct
usage term    = 0.0000012 × odometer_km
capacity term = 0.08 × clamp(60 / battery_capacity_kwh, 0.6, 1.8)

target = clamp(
    0.22 + cycle term + age term + temperature + depth-of-use
    + usage term + capacity term,
    0.2,
    2.5
)
```

The target unit is **percentage points of degradation per 100 cycles**.

#### Features

| Feature | Why it matters |
|---|---|
| `battery_cycles` | Cumulative charge/discharge aging |
| `battery_health_pct` | Current condition of the battery |
| `battery_temperature` | Heat-related chemical aging |
| `battery_capacity_kwh` | Pack size and capacity context |
| `energy_consumed_kwh` | Recent energy demand |
| `odometer_km` | Long-term vehicle usage |
| `avg_speed_kmph` | Driving intensity and thermal load proxy |
| `soc_drop_pct` | Depth of recent battery discharge |
| `vehicle_age_months` | Calendar aging |
| `battery_stress` | Derived temperature/health stress indicator |

#### Model selection

The notebook compares Ridge, Random Forest, XGBoost, LightGBM, and ElasticNet.
Ridge is selected:

- Estimator: `Ridge(alpha=25.0)`
- Saved package name: `Ridge`
- Target key: `battery_degradation_rate_redesigned`
- Scaler: `StandardScaler`
- Holdout R²: **0.9637**
- Holdout MAE: **0.0259 percentage points**
- Relative MAE: **1.8%**

#### Runtime pipeline

The API derives `soc_drop_pct` as:

```text
max(0, soc_start_pct - soc_end_pct)
```

It also calculates:

```text
battery_stress = (battery_temperature_c / 60)
                 × (1 - battery_health_pct / 100)
```

The API currently assumes a 70 km trip distance when constructing the
intermediate `energy_per_km` value because the API schema does not request a
`trip_distance_km` field. The saved Ridge feature list does not include
`energy_per_km`, so this assumption does not affect the current ten-feature
model input; it is retained for feature compatibility and future model
versions.

---

### 6.3 Model 3 - Remaining Useful Life (RUL)

#### Purpose

Estimate the number of days remaining before a major maintenance event. RUL is
useful for maintenance prioritization, workshop capacity planning, vehicle
availability planning, and reducing unplanned downtime.

#### Target

The target is `remaining_useful_life_days`, expressed as an estimated number of
days. The training notebook clips predictions to non-negative values, and the
API also applies `max(0.0, prediction)` before returning the result.

#### Features

| Feature | Why it matters |
|---|---|
| `brake_wear_pct` | Current brake deterioration |
| `suspension_health_pct` | Suspension condition |
| `battery_health_pct` | Major EV subsystem condition |
| `days_since_last_service` | Maintenance recency |
| `vehicle_age_months` | Calendar aging |
| `odometer_km` | Cumulative mechanical usage |
| `aggression_score` | Driving stress and abuse indicator |
| `experience_years` | Driver experience context |
| `base_driver_score` | Existing driver behavior score |
| `mechanical_health` | Combined brake/suspension health index |

#### Derived RUL indicators

The API calculates an aggression score capped at 100:

```text
aggression_score = min(
    100,
    6 × overspeed_events
    + 4 × hard_braking_events
    + 3 × rapid_acceleration_events
    + idle_time_minutes / 5
)
```

It calculates mechanical health as:

```text
mechanical_health =
    50 × (1 - brake_wear_pct / 100)
    + 50 × (suspension_health_pct / 100)
```

The score is an operational indicator, not a certified engineering health
rating.

#### Model selection

The notebook compares Ridge, Random Forest, XGBoost, and LightGBM. XGBoost is
selected:

- Estimator: `XGBRegressor`
- Saved package name: `XGBoost`
- Scaler: `StandardScaler`
- Holdout R²: **0.9719**
- Holdout MAE: **10.3 days**
- Relative MAE: **9.0%**

---

## 7. Training Results

The following values are recorded by the training notebook for the selected
models on the VH009/VH010 holdout:

| Objective | Selected estimator | R² | MAE | Reported interpretation |
|---|---|---:|---:|---|
| Physical range | Random Forest | 0.9907 | 2.20 km | Excellent |
| Battery degradation | Ridge | 0.9637 | 0.0259 percentage points | Excellent |
| Remaining useful life | XGBoost | 0.9719 | 10.3 days | Excellent |

The notebook also reports:

- Range physical-target mean: approximately **66.2 km** on the holdout.
- Battery degradation-target mean: approximately **1.450% per 100 cycles**.
- RUL target mean: approximately **114.2 days**.
- Range formula comparison against the original raw range label: **5.8 km MAE**
  on the holdout.

These metrics describe the dataset and split used in the notebook. They should
be regenerated after changing data, feature definitions, target formulas, or
library versions.

---

## 8. API Reference

### Service endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| `GET` | `/` | Service status and route list |
| `GET` | `/health` | Model loading and health information |
| `POST` | `/predict/range` | Range estimate |
| `POST` | `/predict/battery-degradation` | Battery degradation estimate |
| `POST` | `/predict/rul` | RUL estimate |
| `POST` | `/predict/all` | All three predictions |

Interactive OpenAPI documentation is available at `/docs`; ReDoc is available
at `/redoc` when the server is running.

### Input validation

Pydantic models reject invalid requests before prediction. Examples:

- percentages must be between 0 and 100;
- odometer, cycles, time, and event counts cannot be negative;
- `max_range_km` and `battery_capacity_kwh` must be greater than zero;
- `driving_mode` must be `city` or `highway`.

Prediction exceptions are logged and returned as HTTP 500 responses with an
error detail. Missing model files fail application startup rather than
pretending that a model is available.

### Range request

```json
{
  "odometer_km": 55000,
  "driving_mode": "highway",
  "battery_pct": 65,
  "battery_health_pct": 94,
  "max_range_km": 548,
  "energy_consumed_kwh": 8.2
}
```

Example response shape:

```json
{
  "objective": "range_prediction",
  "model": "Random Forest",
  "range_km_physical": 335.18,
  "range_km_adjusted": 335.18,
  "unit": "km"
}
```

The `model` field reports the saved package name even though the current range
value is generated by the physical formula.

### Battery degradation request

```json
{
  "odometer_km": 72000,
  "battery_cycles": 650,
  "battery_health_pct": 91,
  "battery_temperature_c": 34,
  "battery_capacity_kwh": 60.5,
  "energy_consumed_kwh": 10.5,
  "avg_speed_kmph": 48,
  "soc_start_pct": 72,
  "soc_end_pct": 48,
  "vehicle_age_months": 36
}
```

Example response shape:

```json
{
  "objective": "battery_degradation",
  "model": "Ridge",
  "battery_degradation_pct_per_100_cycles": 0.4821,
  "soc_drop_pct": 24.0
}
```

### RUL request

```json
{
  "odometer_km": 98000,
  "battery_health_pct": 88,
  "brake_wear_pct": 42,
  "suspension_health_pct": 78,
  "days_since_last_service": 145,
  "vehicle_age_months": 60,
  "overspeed_events": 2,
  "hard_braking_events": 2,
  "rapid_acceleration_events": 1,
  "idle_time_minutes": 12,
  "driver_experience_years": 4,
  "driver_score": 72
}
```

Example response shape:

```json
{
  "objective": "remaining_useful_life",
  "model": "XGBoost",
  "rul_days": 312.5,
  "aggression_score": 26.4,
  "mechanical_health": 72.0
}
```

### Combined request

`POST /predict/all` accepts:

```json
{
  "range": { "...": "RangeRequest fields" },
  "battery_degradation": { "...": "BatteryDegradationRequest fields" },
  "rul": { "...": "RulRequest fields" }
}
```

The response contains `range`, `battery_degradation`, and `rul` objects. The
three predictions are computed in one request, but they remain independent
models and do not feed their outputs into one another.

---

## 9. Running the Project

### Prerequisites

- Python 3.10 or newer
- FastAPI
- Uvicorn
- pandas
- scikit-learn
- joblib
- Pydantic
- Jupyter and the packages used in the notebook for retraining
- XGBoost and LightGBM when rerunning the complete training notebook

Install the API dependencies (or use `backend/requirements.txt`):

```bash
python3 -m pip install -r backend/requirements.txt
```

For model retraining, also install the notebook dependencies (or use
`ml/requirements.txt`):

```bash
python3 -m pip install -r ml/requirements.txt
```

### Start the API

Run from the repository root:

```bash
uvicorn backend.app:app --host 0.0.0.0 --port 8000 --reload
```

Or run from inside `backend/`:

```bash
cd backend
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

Both forms work because the application resolves `ml/models/` from its own file
location rather than from the current working directory.

Open:

- `http://localhost:8000/`
- `http://localhost:8000/health`
- `http://localhost:8000/docs`

### Run the CLI

From the repository root:

```bash
python3 backend/predict_ev_objectives.py
```

Or from inside `backend/`:

```bash
cd backend
python3 predict_ev_objectives.py
```

The CLI supports:

1. Range prediction
2. Battery degradation prediction
3. RUL prediction
4. A demo that runs all three with realistic sample values

The CLI and API use the same overall formulas and serialized packages, but
their request objects are not identical. In particular, the CLI asks for
`trip_distance_km` for battery degradation, whereas the API currently uses a
70 km compatibility value internally.

### Environment variables

| Variable | Default | Purpose |
|---|---|---|
| `MODEL_DIR` | `<project root>/ml/models` | Directory containing serialized model packages |
| `DATASET_DIR` | `<project root>/ml/dataset` | Directory containing training CSV files |

Example:

```bash
MODEL_DIR=/data/ev_models uvicorn backend.app:app --host 0.0.0.0 --port 8000
```

---

## 10. Retraining and Model Export

Open `ml/new_model.ipynb` in Jupyter and run the cells in order. Launch Jupyter
from the repository root or from `ml/`; the notebook resolves `dataset/` and
`models/` relative to `ml/` in either case. The training workflow:

1. Loads and normalizes telemetry.
2. Sorts records by vehicle and timestamp.
3. Builds derived features and target columns.
4. Encodes selected categorical columns.
5. Separates training and holdout vehicles.
6. Fits `StandardScaler` using training rows only.
7. Trains candidate regressors.
8. Evaluates R², MAE, RMSE, and relative MAE.
9. Selects the candidate with the highest R².
10. Saves the estimator, scaler, feature list, and metadata using joblib.

The package format is:

```python
{
    "model": trained_estimator,
    "scaler": fitted_standard_scaler,
    "features": ["feature_a", "feature_b", "..."],
    "name": "selected_model_name",
    # "target": "...",             # used by battery package
    # "prediction_mode": "...",    # used by range package
}
```

If feature names or target semantics change, update and test the corresponding
request schema and inference function. The API should never rely on positional
columns without using the saved feature list.

---

## 11. Technical Design Choices

### Why use three models?

Range, battery degradation, and RUL are different prediction problems with
different units, targets, and causal signals. Separate models make each output
easier to validate and allow the best algorithm for each objective to be used.

### Why use a physical formula for range?

A purely statistical range model can produce impossible boundary values. The
formula guarantees that full SOC and full health produce the rated maximum
range, and that reduced SOC/health cannot produce a result above the rated
maximum. A practical adjustment is applied only downward.

### Why use StandardScaler?

The training features have different scales: kilometres, percentages, cycles,
temperatures, event counts, and scores. The scaler standardizes feature
magnitude before regression. The same fitted scaler must be used at inference;
the fitted scaler is therefore stored with each model.

### Why compare multiple algorithms?

Model comparison prevents selecting an algorithm solely by convention. The
notebook evaluates linear regularized models and tree/boosting models, then
selects the highest holdout R² for each objective.

### Why split by vehicle?

Rows from the same vehicle are correlated. A random row split could put nearly
identical records from one vehicle into both train and test sets and produce
overly optimistic results. A vehicle-level holdout better measures transfer to
unseen vehicles.

---

## 12. Limitations and Risks

- The dataset is finite and the metrics are specific to the recorded fleet and
  VH009/VH010 holdout.
- The training code uses 2024 as the reference year for vehicle age; this
  should be parameterized for a live system.
- `driver_master.csv` and `vehicle_master.csv` are not currently part of the
  notebook's active training load.
- The API uses permissive CORS (`allow_origins=["*"]`) and has no
  authentication. This is acceptable for local demonstration only, not a
  production deployment.
- Pickle/joblib files should be loaded only from trusted sources.
- Serialized scikit-learn objects can produce version compatibility warnings
  when loaded with a different scikit-learn version than the training
  environment. The current artifacts were observed to have been created with
  scikit-learn 1.3.2 and loaded in an environment with 1.5.2; retraining or
  pinning the original environment is recommended for production.
- The range endpoint's API schema and notebook feature names are not identical
  because runtime range uses the physical formula.
- RUL is a maintenance estimate, not a guaranteed failure date.
- There is no uncertainty interval, drift monitoring, feature-quality
  dashboard, or automatic model retraining.
- The current API prints request payloads and responses to logs. Production
  logging should redact sensitive fleet or driver information.

---

## 13. Suggested Production Improvements

1. Add a pinned `requirements.txt` or `pyproject.toml`.
2. Include `trip_distance_km` in the battery degradation API request.
3. Add automated tests for formulas, validation boundaries, model loading, and
   endpoint response schemas.
4. Add authentication, restricted CORS, rate limiting, and HTTPS.
5. Replace manual requests with MQTT, Kafka, or a fleet database integration.
6. Add prediction history and maintenance outcomes for model monitoring.
7. Add confidence intervals or calibrated prediction bands.
8. Schedule retraining when drift or new maintenance labels are detected.
9. Version model packages and record training data, code, and dependency
   versions as metadata.
10. Add a model registry and approval process before deploying a new model.

---

## 14. Report/Viva Explanation

The project can be explained in the following sequence:

1. **Need:** EV fleet operators need range visibility and predictive
   maintenance rather than reacting after a vehicle fails.
2. **Data:** telemetry records contain battery, trip, vehicle, driver, and
   maintenance signals.
3. **Feature engineering:** raw signals are converted into SOC drop, efficiency,
   aggression, mechanical health, battery stress, and age indicators.
4. **Modeling:** separate regressors are compared for range, degradation, and
   RUL because the targets represent different physical processes.
5. **Validation:** vehicles VH001–VH008 train the models and VH009–VH010 are
   held out for evaluation.
6. **Deployment:** selected estimators and scalers are serialized with joblib.
7. **Serving:** FastAPI validates JSON requests and returns structured
   predictions.
8. **Safety:** range is constrained by a physical formula and all outputs are
   decision-support estimates rather than direct vehicle-control commands.

### One-paragraph abstract

This project presents an EV fleet management and predictive-maintenance
solution that estimates current driving range, battery degradation rate, and
remaining useful life from telemetry and vehicle-operation data. A notebook
performs data preparation, domain-based feature engineering, vehicle-level
holdout validation, model comparison, scaling, and joblib serialization. The
best recorded estimators are Random Forest for the range training objective,
Ridge for battery degradation, and XGBoost for RUL. At deployment, a FastAPI
service validates input requests and exposes individual and combined prediction
endpoints. Range inference additionally applies a physics-based SOC and battery
health constraint so the prediction remains bounded by the rated vehicle
range. The system is intended to support charging, vehicle assignment,
battery monitoring, and preventive maintenance decisions.

---

## 15. License and Academic Use

This repository is an academic/project implementation. Add the appropriate
license, dataset attribution, and institutional details before public
distribution or commercial deployment.

---

## 16. Frontend Folder

The separate [`frontend/`](./frontend/) application provides a professional
fleet command-center UI for the FastAPI service. It includes:

- Fleet overview and model health status
- Range intelligence inputs and physical/practical range results
- Battery degradation inputs and degradation-per-100-cycle results
- RUL inputs covering vehicle, driver, mechanical, and service signals
- Combined `/predict/all` assessment
- Responsive desktop, tablet, and mobile layout
- Animated CSS 3D electric vehicle visualization
- Loading, offline, backend-error, reset, and result states

Run the backend first, then start the frontend:

```bash
uvicorn backend.app:app --host 0.0.0.0 --port 8000 --reload
cd frontend
npm install
npm run dev
```

Open `http://localhost:5173`. Vite proxies `/api` requests to the FastAPI
server at `http://localhost:8000`. Set `VITE_API_URL` when the backend is
hosted elsewhere.

---

## 17. Backend Folder

The [`backend/`](./backend/) folder contains only the serving layer. It does not
contain training logic or datasets.

| File | Purpose |
|---|---|
| [`backend/app.py`](./backend/app.py) | FastAPI app, Pydantic schemas, feature construction, and prediction routes |
| [`backend/predict_ev_objectives.py`](./backend/predict_ev_objectives.py) | Interactive CLI and preset demo runner |
| [`backend/requirements.txt`](./backend/requirements.txt) | API runtime dependencies |

The backend reads trained artifacts from `ml/models/` at startup and fails fast
if a required package is missing, so a broken deployment cannot silently serve
stale or absent models.

---

## 18. ML Folder

The [`ml/`](./ml/) folder owns everything related to data and model training.

| Path | Purpose |
|---|---|
| [`ml/new_model.ipynb`](./ml/new_model.ipynb) | Data preparation, feature engineering, model comparison, and export |
| [`ml/dataset/`](./ml/dataset/) | Source telemetry and master data CSVs |
| [`ml/models/`](./ml/models/) | Serialized model packages consumed by the backend |
| [`ml/requirements.txt`](./ml/requirements.txt) | Training and notebook dependencies |

Retraining writes new packages to `ml/models/`, so the backend picks up updated
models on its next restart without any code change.
