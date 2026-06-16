"""
Standalone EV prediction runner.

Uses the trained model packages in ./models:
  - range_model.pkl    : range objective, constrained by physics
  - battery_model.pkl  : battery degradation objective
  - rul_model.pkl      : remaining useful life objective

Run:
  python3 predict_ev_objectives.py
"""

from __future__ import annotations

import os
from dataclasses import dataclass, asdict

import joblib
import pandas as pd


MODEL_DIR = os.getenv("MODEL_DIR", "models")
DATASET_DIR = os.getenv("DATASET_DIR", "dataset")


def _load_model(name: str) -> dict:
    path = os.path.join(MODEL_DIR, name)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found: {path}")
    return joblib.load(path)


def _ask_float(prompt: str, default: float | None = None) -> float:
    suffix = f" [{default}]" if default is not None else ""
    while True:
        raw = input(f"{prompt}{suffix}: ").strip()
        if not raw and default is not None:
            return float(default)
        try:
            return float(raw)
        except ValueError:
            print("Enter a number.")


def _ask_choice(prompt: str, choices: list[str], default: str) -> str:
    normalized = {c.lower(): c for c in choices}
    while True:
        raw = input(f"{prompt} {choices} [{default}]: ").strip().lower()
        if not raw:
            return default
        if raw in normalized:
            return normalized[raw]
        print(f"Choose one of: {', '.join(choices)}")


def physical_range_km(
    battery_pct: float,
    battery_health_pct: float,
    max_range_km: float,
) -> float:
    """
    Hard physical constraint:
      100% SOC + 100% health -> max_range_km exactly.
      Lower SOC/health scales proportionally.
    """
    soc = min(max(battery_pct, 0.0), 100.0) / 100.0
    health = min(max(battery_health_pct, 0.0), 100.0) / 100.0
    return round(max_range_km * soc * health, 3)


def _battery_stress(battery_temperature_c: float, battery_health_pct: float) -> float:
    return (battery_temperature_c / 60.0) * (1.0 - battery_health_pct / 100.0)


def _mechanical_health(brake_wear_pct: float, suspension_health_pct: float) -> float:
    return (1.0 - brake_wear_pct / 100.0) * 50.0 + (suspension_health_pct / 100.0) * 50.0


@dataclass
class RangeInput:
    odometer_km: float
    driving_mode: str
    battery_pct: float
    battery_health_pct: float
    max_range_km: float
    energy_consumed_kwh: float


@dataclass
class BatteryDegradationInput:
    odometer_km: float
    battery_cycles: float
    battery_health_pct: float
    battery_temperature_c: float
    battery_capacity_kwh: float
    energy_consumed_kwh: float
    trip_distance_km: float
    avg_speed_kmph: float
    soc_start_pct: float
    soc_end_pct: float
    vehicle_age_months: float


@dataclass
class RulInput:
    odometer_km: float
    battery_health_pct: float
    brake_wear_pct: float
    suspension_health_pct: float
    days_since_last_service: float
    vehicle_age_months: float
    overspeed_events: float
    hard_braking_events: float
    rapid_acceleration_events: float
    idle_time_minutes: float
    driver_experience_years: float
    driver_score: float


def predict_range(data: RangeInput) -> dict:
    # The saved range package is loaded to verify model availability, but the
    # final range prediction intentionally uses the hard physical formula.
    _load_model("range_model.pkl")

    mode = data.driving_mode.lower()
    mode_factor = 1.0 if mode == "highway" else 0.92
    energy_factor = 1.0
    if data.energy_consumed_kwh > 0:
        # Light practical penalty for very high trip energy use. Never boost
        # above the SOC/health physical range.
        energy_factor = max(0.82, min(1.0, 1.0 - (data.energy_consumed_kwh - 15.0) * 0.006))

    base_range = physical_range_km(
        data.battery_pct,
        data.battery_health_pct,
        data.max_range_km,
    )
    adjusted_range = round(min(base_range, base_range * mode_factor * energy_factor), 3)

    return {
        "objective": "range_prediction",
        "range_km_physical": base_range,
        "range_km_adjusted": adjusted_range,
        "note": "Use range_km_physical when strict SOC/health bounds are required.",
    }


def _ensure_model_features(row: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    row = row.copy()
    if "energy_per_km" in features and "energy_per_km" not in row.columns:
        trip_distance = row.get("trip_distance_km", pd.Series([50.0])).clip(lower=0.1)
        row["energy_per_km"] = row["energy_consumed_kwh"] / trip_distance
    if "service_overdue" in features and "service_overdue" not in row.columns:
        row["service_overdue"] = (row["days_since_last_service"] > 120).astype(int)
    if "brake_wear_pct_roll7" in features and "brake_wear_pct_roll7" not in row.columns:
        row["brake_wear_pct_roll7"] = row["brake_wear_pct"]
    missing = [feature for feature in features if feature not in row.columns]
    if missing:
        raise KeyError(f"Missing model features: {missing}")
    return row


def predict_battery_degradation(data: BatteryDegradationInput) -> dict:
    pkg = _load_model("battery_model.pkl")
    soc_drop_pct = max(0.0, data.soc_start_pct - data.soc_end_pct)
    row = pd.DataFrame([{
        "battery_cycles": data.battery_cycles,
        "battery_health_pct": data.battery_health_pct,
        "battery_temperature": data.battery_temperature_c,
        "battery_capacity_kwh": data.battery_capacity_kwh,
        "energy_consumed_kwh": data.energy_consumed_kwh,
        "trip_distance_km": data.trip_distance_km,
        "energy_per_km": data.energy_consumed_kwh / max(data.trip_distance_km, 0.1),
        "odometer_km": data.odometer_km,
        "avg_speed_kmph": data.avg_speed_kmph,
        "soc_drop_pct": soc_drop_pct,
        "vehicle_age_months": data.vehicle_age_months,
        "battery_stress": _battery_stress(data.battery_temperature_c, data.battery_health_pct),
    }])

    features = pkg["features"]
    row = _ensure_model_features(row, features)
    x = pkg["scaler"].transform(row[features].fillna(0))
    pred = float(pkg["model"].predict(x)[0])

    return {
        "objective": "battery_degradation",
        "battery_degradation_pct_per_100_cycles": round(max(0.0, pred), 4),
        "soc_drop_pct": round(soc_drop_pct, 2),
    }


def predict_rul(data: RulInput) -> dict:
    pkg = _load_model("rul_model.pkl")
    aggression_score = min(
        100.0,
        data.overspeed_events * 6.0
        + data.hard_braking_events * 4.0
        + data.rapid_acceleration_events * 3.0
        + data.idle_time_minutes / 5.0,
    )
    row = pd.DataFrame([{
        "brake_wear_pct": data.brake_wear_pct,
        "suspension_health_pct": data.suspension_health_pct,
        "battery_health_pct": data.battery_health_pct,
        "days_since_last_service": data.days_since_last_service,
        "vehicle_age_months": data.vehicle_age_months,
        "odometer_km": data.odometer_km,
        "aggression_score": aggression_score,
        "experience_years": data.driver_experience_years,
        "base_driver_score": data.driver_score,
        "mechanical_health": _mechanical_health(data.brake_wear_pct, data.suspension_health_pct),
    }])

    features = pkg["features"]
    row = _ensure_model_features(row, features)
    x = pkg["scaler"].transform(row[features].fillna(0))
    pred = float(pkg["model"].predict(x)[0])

    return {
        "objective": "remaining_useful_life",
        "rul_days": round(max(0.0, pred), 1),
        "aggression_score": round(aggression_score, 2),
        "mechanical_health": round(float(row["mechanical_health"].iloc[0]), 2),
    }


def prompt_range() -> RangeInput:
    return RangeInput(
        odometer_km=_ask_float("Odometer km", 45000),
        driving_mode=_ask_choice("Driving mode", ["city", "highway"], "city"),
        battery_pct=_ask_float("Battery percentage / SOC", 60),
        battery_health_pct=_ask_float("Battery health percent", 92),
        max_range_km=_ask_float("Vehicle max range km", 500),
        energy_consumed_kwh=_ask_float("Energy consumed kWh in recent trip", 12),
    )


def prompt_battery() -> BatteryDegradationInput:
    return BatteryDegradationInput(
        odometer_km=_ask_float("Odometer km", 70000),
        battery_cycles=_ask_float("Battery charge cycles", 650),
        battery_health_pct=_ask_float("Battery health percent", 90),
        battery_temperature_c=_ask_float("Battery temperature C", 34),
        battery_capacity_kwh=_ask_float("Battery capacity kWh", 60),
        energy_consumed_kwh=_ask_float("Energy consumed kWh in recent trip", 14),
        trip_distance_km=_ask_float("Trip distance km", 70),
        avg_speed_kmph=_ask_float("Average speed km/h", 48),
        soc_start_pct=_ask_float("Trip start SOC percent", 75),
        soc_end_pct=_ask_float("Trip end SOC percent", 48),
        vehicle_age_months=_ask_float("Vehicle age in months", 36),
    )


def prompt_rul() -> RulInput:
    return RulInput(
        odometer_km=_ask_float("Odometer km", 90000),
        battery_health_pct=_ask_float("Battery health percent", 88),
        brake_wear_pct=_ask_float("Brake wear percent", 35),
        suspension_health_pct=_ask_float("Suspension health percent", 82),
        days_since_last_service=_ask_float("Days since last service", 120),
        vehicle_age_months=_ask_float("Vehicle age in months", 48),
        overspeed_events=_ask_float("Overspeed events in recent trip", 1),
        hard_braking_events=_ask_float("Hard braking events in recent trip", 1),
        rapid_acceleration_events=_ask_float("Rapid acceleration events in recent trip", 1),
        idle_time_minutes=_ask_float("Idle time minutes in recent trip", 8),
        driver_experience_years=_ask_float("Driver experience years", 5),
        driver_score=_ask_float("Driver score 0-100", 78),
    )


def print_result(result: dict) -> None:
    print("\nPrediction")
    print("-" * 40)
    for key, value in result.items():
        print(f"{key}: {value}")


def demo_predictions() -> None:
    print("\nDemo predictions with realistic out-of-dataset values")
    print_result(predict_range(RangeInput(
        odometer_km=55000,
        driving_mode="highway",
        battery_pct=65,
        battery_health_pct=94,
        max_range_km=548,
        energy_consumed_kwh=8.2,
    )))
    print_result(predict_battery_degradation(BatteryDegradationInput(
        odometer_km=72000,
        battery_cycles=650,
        battery_health_pct=91,
        battery_temperature_c=34,
        battery_capacity_kwh=60.5,
        energy_consumed_kwh=10.5,
        trip_distance_km=62,
        avg_speed_kmph=48,
        soc_start_pct=72,
        soc_end_pct=48,
        vehicle_age_months=36,
    )))
    print_result(predict_rul(RulInput(
        odometer_km=98000,
        battery_health_pct=88,
        brake_wear_pct=42,
        suspension_health_pct=78,
        days_since_last_service=145,
        vehicle_age_months=60,
        overspeed_events=2,
        hard_braking_events=2,
        rapid_acceleration_events=1,
        idle_time_minutes=12,
        driver_experience_years=4,
        driver_score=72,
    )))


def main() -> None:
    print("EV Objective Predictor")
    print("1. Range prediction")
    print("2. Battery degradation")
    print("3. Remaining useful life")
    print("4. Demo all")
    choice = _ask_choice("Select objective", ["1", "2", "3", "4"], "4")

    if choice == "1":
        print_result(predict_range(prompt_range()))
    elif choice == "2":
        print_result(predict_battery_degradation(prompt_battery()))
    elif choice == "3":
        print_result(predict_rul(prompt_rul()))
    else:
        demo_predictions()


if __name__ == "__main__":
    main()
