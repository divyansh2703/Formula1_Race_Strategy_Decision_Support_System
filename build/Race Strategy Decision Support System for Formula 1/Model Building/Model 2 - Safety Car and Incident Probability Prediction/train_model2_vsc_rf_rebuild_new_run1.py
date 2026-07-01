from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

try:
    from xgboost import XGBClassifier
except Exception:
    XGBClassifier = None

try:
    import optuna
except Exception:
    optuna = None


EVENTS = ["sc", "vsc", "rf"]

EVENT_ACTIVE_COLS = {
    "sc": "is_under_safety_car_event",
    "vsc": "is_under_vsc_event",
    "rf": "is_under_red_flag_event",
}

EVAL_HORIZONS_DEFAULT = [3, 5, 10]


@dataclass
class SplitIds:
    train_races: List[str]
    val_races: List[str]
    test_races: List[str]


def now_ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def parse_csv_list(s: str) -> List[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


def safe_numeric(s: pd.Series, fill: float = 0.0) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").fillna(fill)


def safe_roc_auc(y_true: np.ndarray, p_hat: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, p_hat))


def safe_pr_auc(y_true: np.ndarray, p_hat: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(average_precision_score(y_true, p_hat))


def safe_brier(y_true: np.ndarray, p_hat: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(brier_score_loss(y_true, p_hat))


def safe_logloss(y_true: np.ndarray, p_hat: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(log_loss(y_true, np.clip(p_hat, 1e-6, 1.0 - 1e-6)))


def score_split(y_true: np.ndarray, p_hat: np.ndarray) -> Dict:
    return {
        "n": int(len(y_true)),
        "event_rate": float(np.mean(y_true)) if len(y_true) else float("nan"),
        "roc_auc": safe_roc_auc(y_true, p_hat),
        "pr_auc": safe_pr_auc(y_true, p_hat),
        "brier": safe_brier(y_true, p_hat),
        "logloss": safe_logloss(y_true, p_hat),
    }


def reliability_table(y_true: np.ndarray, p_hat: np.ndarray, n_bins: int = 10) -> List[Dict]:
    df = pd.DataFrame({"y": y_true.astype(int), "p": p_hat.astype(float)})
    if len(df) == 0:
        return []

    try:
        df["bin"] = pd.qcut(df["p"], q=n_bins, duplicates="drop")
    except Exception:
        df["bin"] = pd.cut(df["p"], bins=n_bins, include_lowest=True)

    out = (
        df.groupby("bin", observed=True)
        .agg(
            n=("y", "size"),
            event_rate=("y", "mean"),
            mean_p=("p", "mean"),
            min_p=("p", "min"),
            max_p=("p", "max"),
        )
        .reset_index()
    )
    out["bin"] = out["bin"].astype(str)
    return out.to_dict(orient="records")


def safe_logit(p: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    p = np.clip(p, eps, 1.0 - eps)
    return np.log(p / (1.0 - p))


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def fit_platt_scaler(p_val_raw: np.ndarray, y_val: np.ndarray) -> Dict[str, float]:
    z = safe_logit(p_val_raw).reshape(-1, 1)
    lr = LogisticRegression(solver="lbfgs", max_iter=2000)
    lr.fit(z, y_val)
    return {"a": float(lr.coef_[0][0]), "b": float(lr.intercept_[0])}


def apply_platt(p_raw: np.ndarray, calib: Dict[str, float]) -> np.ndarray:
    z = safe_logit(p_raw)
    return sigmoid(calib["a"] * z + calib["b"])


def make_group_split(race_ids: np.ndarray, seed: int, test_frac: float, val_frac: float) -> SplitIds:
    rng = np.random.default_rng(seed)
    uniq = np.array(sorted(pd.unique(race_ids.astype(str))))
    rng.shuffle(uniq)

    n = len(uniq)
    n_test = max(1, int(round(n * test_frac)))
    n_val = max(1, int(round((n - n_test) * val_frac)))

    test_races = uniq[:n_test].tolist()
    val_races = uniq[n_test:n_test + n_val].tolist()
    train_races = uniq[n_test + n_val:].tolist()

    return SplitIds(train_races=train_races, val_races=val_races, test_races=test_races)


def build_preprocessor(cat_cols: List[str], num_cols: List[str]) -> ColumnTransformer:
    cat_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("ohe", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    num_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("cat", cat_pipe, cat_cols),
            ("num", num_pipe, num_cols),
        ],
        remainder="drop",
        sparse_threshold=0.3,
    )


def fit_xgb_with_early_stopping(model, X_train, y_train, X_val, y_val, rounds: int = 35) -> None:
    fit_kwargs = {"eval_set": [(X_val, y_val)], "verbose": False}

    try:
        model.fit(X_train, y_train, early_stopping_rounds=rounds, **fit_kwargs)
        return
    except TypeError:
        pass

    try:
        from xgboost.callback import EarlyStopping
        model.fit(
            X_train,
            y_train,
            callbacks=[EarlyStopping(rounds=rounds, save_best=True)],
            **fit_kwargs,
        )
        return
    except Exception:
        pass

    model.fit(X_train, y_train, **fit_kwargs)


def aggregate_race_lap(master: pd.DataFrame) -> pd.DataFrame:
    df = master.copy()
    df = df.sort_values(["race_id", "driver_id", "lap_number"]).reset_index(drop=True)

    needed = ["race_id", "driver_id", "lap_number"]
    for c in needed:
        if c not in df.columns:
            raise ValueError(f"Missing required column: {c}")

    numeric_try = [
        "position",
        "gap_ahead",
        "gap_behind",
        "tyre_age",
        "stint_number",
        "is_pit_lap",
        "is_in_lap",
        "is_out_lap",
        "pitted_this_lap_flag",
        "clean_air_flag",
        "track_temperature_used",
        "air_temperature_used",
        "rain_flag_used",
        "used_race_level_weather_flag",
        "laps_remaining",
        "track_length",
        "pit_lane_time_loss",
        "overtaking_difficulty_index",
        "safety_car_probability_baseline",
        "total_laps",
        "round",
        "season",
        "cars_ahead_count",
        "cars_behind_count",
        "avg_tyre_age_ahead",
        "avg_tyre_age_behind",
        "field_size",
        "lap_time",
    ]
    for c in numeric_try:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    for c in EVENT_ACTIVE_COLS.values():
        if c not in df.columns:
            raise ValueError(f"Missing event column: {c}")
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)

    for c in ["is_under_safety_car", "is_under_virtual_safety_car"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)

    gk = ["race_id", "lap_number"]

    base_cols = [
        "season",
        "round",
        "circuit_id",
        "total_laps",
        "track_length",
        "pit_lane_time_loss",
        "overtaking_difficulty_index",
        "safety_car_probability_baseline",
        "race_level_weather_condition",
        "track_temperature_used",
        "air_temperature_used",
        "rain_flag_used",
        "used_race_level_weather_flag",
        "laps_remaining",
        "race_phase",
        "field_size",
    ]
    base_cols = [c for c in base_cols if c in df.columns]
    race_lap = df.groupby(gk, as_index=False)[base_cols].first()

    act_cols = [
        c
        for c in [
            "is_under_safety_car",
            "is_under_virtual_safety_car",
            "is_under_safety_car_event",
            "is_under_vsc_event",
            "is_under_red_flag_event",
        ]
        if c in df.columns
    ]
    if act_cols:
        act = df.groupby(gk, as_index=False)[act_cols].max()
        race_lap = race_lap.merge(act, on=gk, how="left")

    agg = df[gk].drop_duplicates().copy()

    if "driver_id" in df.columns:
        tmp = (
            df.groupby(gk, as_index=False)["driver_id"]
            .nunique()
            .rename(columns={"driver_id": "drivers_running"})
        )
        agg = agg.merge(tmp, on=gk, how="left")

    if "team_id" in df.columns:
        tmp = (
            df.groupby(gk, as_index=False)["team_id"]
            .nunique()
            .rename(columns={"team_id": "teams_running"})
        )
        agg = agg.merge(tmp, on=gk, how="left")

    if "gap_ahead" in df.columns:
        x = (
            df.groupby(gk, as_index=False)
            .agg(
                min_gap_ahead=("gap_ahead", "min"),
                median_gap_ahead=("gap_ahead", "median"),
                mean_gap_ahead=("gap_ahead", "mean"),
            )
        )
        agg = agg.merge(x, on=gk, how="left")

        close1 = df.assign(close_ahead_1s=(df["gap_ahead"] <= 1.0).astype(float))
        close15 = df.assign(close_ahead_15s=(df["gap_ahead"] <= 1.5).astype(float))
        x1 = (
            close1.groupby(gk, as_index=False)["close_ahead_1s"]
            .mean()
            .rename(columns={"close_ahead_1s": "share_close_ahead_1s"})
        )
        x2 = (
            close15.groupby(gk, as_index=False)["close_ahead_15s"]
            .mean()
            .rename(columns={"close_ahead_15s": "share_close_ahead_15s"})
        )
        agg = agg.merge(x1, on=gk, how="left").merge(x2, on=gk, how="left")

    if "gap_behind" in df.columns:
        x = (
            df.groupby(gk, as_index=False)["gap_behind"]
            .mean()
            .rename(columns={"gap_behind": "mean_gap_behind"})
        )
        agg = agg.merge(x, on=gk, how="left")

        closeb = df.assign(close_behind_1s=(df["gap_behind"] <= 1.0).astype(float))
        xb = (
            closeb.groupby(gk, as_index=False)["close_behind_1s"]
            .mean()
            .rename(columns={"close_behind_1s": "share_close_behind_1s"})
        )
        agg = agg.merge(xb, on=gk, how="left")

    if "tyre_age" in df.columns:
        x = (
            df.groupby(gk, as_index=False)
            .agg(
                mean_tyre_age=("tyre_age", "mean"),
                std_tyre_age=("tyre_age", "std"),
            )
        )
        agg = agg.merge(x, on=gk, how="left")

        old = df.assign(old_tyre_15=(df["tyre_age"] >= 15).astype(float))
        xo = (
            old.groupby(gk, as_index=False)["old_tyre_15"]
            .mean()
            .rename(columns={"old_tyre_15": "share_old_tyres_15"})
        )
        agg = agg.merge(xo, on=gk, how="left")

    if "pitted_this_lap_flag" in df.columns:
        x = (
            df.groupby(gk, as_index=False)["pitted_this_lap_flag"]
            .mean()
            .rename(columns={"pitted_this_lap_flag": "share_pitting"})
        )
        agg = agg.merge(x, on=gk, how="left")

    if "clean_air_flag" in df.columns:
        x = (
            df.groupby(gk, as_index=False)["clean_air_flag"]
            .mean()
            .rename(columns={"clean_air_flag": "share_clean_air"})
        )
        agg = agg.merge(x, on=gk, how="left")

    if "lap_time" in df.columns:
        x = (
            df.groupby(gk, as_index=False)
            .agg(
                mean_lap_time=("lap_time", "mean"),
                std_lap_time=("lap_time", "std"),
            )
        )
        agg = agg.merge(x, on=gk, how="left")

    race_lap = race_lap.merge(agg, on=gk, how="left")
    race_lap = race_lap.sort_values(gk).reset_index(drop=True)

    lag_cols = [
        "mean_gap_ahead",
        "median_gap_ahead",
        "mean_gap_behind",
        "share_close_ahead_1s",
        "share_close_ahead_15s",
        "share_close_behind_1s",
        "mean_tyre_age",
        "std_tyre_age",
        "share_old_tyres_15",
        "share_pitting",
        "share_clean_air",
        "drivers_running",
        "teams_running",
        "mean_lap_time",
        "std_lap_time",
        "is_under_safety_car_event",
        "is_under_vsc_event",
        "is_under_red_flag_event",
        "is_under_safety_car",
        "is_under_virtual_safety_car",
    ]
    lag_cols = [c for c in lag_cols if c in race_lap.columns]

    for c in lag_cols:
        race_lap[f"{c}_lag1"] = race_lap.groupby("race_id", sort=False)[c].shift(1)

    for c in [x for x in race_lap.columns if x.endswith("_lag1")]:
        if race_lap[c].dtype == object:
            race_lap[c] = race_lap[c].fillna("UNKNOWN")
        else:
            race_lap[c] = race_lap[c].fillna(0.0)

    race_lap = add_vsc_instability_features(race_lap)
    race_lap = add_rf_intensity_context_features(race_lap)

    return race_lap


def add_vsc_instability_features(race_lap: pd.DataFrame) -> pd.DataFrame:
    df = race_lap.copy()

    def col(name: str) -> pd.Series:
        if name in df.columns:
            return safe_numeric(df[name])
        return pd.Series(0.0, index=df.index)

    df["delta_mean_gap_ahead"] = col("mean_gap_ahead") - col("mean_gap_ahead_lag1")
    df["delta_mean_gap_behind"] = col("mean_gap_behind") - col("mean_gap_behind_lag1")
    df["delta_median_gap_ahead"] = col("median_gap_ahead") - col("median_gap_ahead_lag1")
    df["delta_share_close_ahead_1s"] = col("share_close_ahead_1s") - col("share_close_ahead_1s_lag1")
    df["delta_share_close_ahead_15s"] = col("share_close_ahead_15s") - col("share_close_ahead_15s_lag1")
    df["delta_share_close_behind_1s"] = col("share_close_behind_1s") - col("share_close_behind_1s_lag1")
    df["delta_share_pitting"] = col("share_pitting") - col("share_pitting_lag1")
    df["delta_mean_tyre_age"] = col("mean_tyre_age") - col("mean_tyre_age_lag1")
    df["delta_share_old_tyres_15"] = col("share_old_tyres_15") - col("share_old_tyres_15_lag1")
    df["delta_mean_lap_time"] = col("mean_lap_time") - col("mean_lap_time_lag1")
    df["delta_std_lap_time"] = col("std_lap_time") - col("std_lap_time_lag1")

    df["traffic_disruption_index"] = (
        col("share_close_ahead_1s")
        + col("share_close_behind_1s")
        + col("share_pitting")
    )

    df["traffic_disruption_delta"] = (
        df["delta_share_close_ahead_1s"]
        + df["delta_share_close_behind_1s"]
        + df["delta_share_pitting"]
    )

    df["gap_compression_index"] = (
        -df["delta_mean_gap_ahead"].fillna(0.0)
        -df["delta_mean_gap_behind"].fillna(0.0)
    )

    df["lap_time_volatility_index"] = (
        np.abs(df["delta_mean_lap_time"].fillna(0.0))
        + np.abs(df["delta_std_lap_time"].fillna(0.0))
    )

    df["vsc_local_instability_index"] = (
        df["traffic_disruption_index"].fillna(0.0)
        + df["traffic_disruption_delta"].fillna(0.0)
        + df["gap_compression_index"].fillna(0.0)
        + df["lap_time_volatility_index"].fillna(0.0)
    )

    df["close_pack_x_pitting"] = (
        col("share_close_ahead_1s") * col("share_pitting")
    )
    df["close_pack_x_old_tyres"] = (
        col("share_close_ahead_15s") * col("share_old_tyres_15")
    )
    df["wet_x_instability"] = (
        col("rain_flag_used") * df["vsc_local_instability_index"].fillna(0.0)
    )
    df["pit_surge_x_gap_compression"] = (
        df["delta_share_pitting"].fillna(0.0) * df["gap_compression_index"].fillna(0.0)
    )
    df["pack_pressure_balance"] = (
        col("share_close_ahead_1s") + col("share_close_behind_1s") - col("share_clean_air")
    )

    for c in [
        "delta_mean_gap_ahead",
        "delta_mean_gap_behind",
        "delta_median_gap_ahead",
        "delta_share_close_ahead_1s",
        "delta_share_close_ahead_15s",
        "delta_share_close_behind_1s",
        "delta_share_pitting",
        "delta_mean_tyre_age",
        "delta_share_old_tyres_15",
        "delta_mean_lap_time",
        "delta_std_lap_time",
        "traffic_disruption_index",
        "traffic_disruption_delta",
        "gap_compression_index",
        "lap_time_volatility_index",
        "vsc_local_instability_index",
        "close_pack_x_pitting",
        "close_pack_x_old_tyres",
        "wet_x_instability",
        "pit_surge_x_gap_compression",
        "pack_pressure_balance",
    ]:
        df[c] = safe_numeric(df[c])

    return df


def add_rf_intensity_context_features(race_lap: pd.DataFrame) -> pd.DataFrame:
    df = race_lap.copy()

    total = safe_numeric(df["total_laps"], fill=np.nan).replace(0, np.nan)
    frac = safe_numeric(df["lap_number"], fill=np.nan) / total
    df["rf_phase_bucket"] = np.where(frac <= 0.33, "early", np.where(frac <= 0.66, "mid", "late"))
    df["rf_weather_bucket"] = np.where(safe_numeric(df.get("rain_flag_used", pd.Series(0.0, index=df.index))) > 0, "wet", "dry")
    df["rf_track_risk_proxy"] = (
        safe_numeric(df.get("overtaking_difficulty_index", pd.Series(0.0, index=df.index)))
        + safe_numeric(df.get("safety_car_probability_baseline", pd.Series(0.0, index=df.index)))
    )
    return df


def add_event_start_flags(race_lap: pd.DataFrame, event: str) -> pd.DataFrame:
    active_col = EVENT_ACTIVE_COLS[event]
    df = race_lap.sort_values(["race_id", "lap_number"]).copy()

    cur = pd.to_numeric(df[active_col], errors="coerce").fillna(0).astype(int)
    prev = df.groupby("race_id", sort=False)[active_col].shift(1).fillna(0).astype(int)
    df[f"{event}_start_this_lap"] = ((cur == 1) & (prev == 0)).astype(int)
    return df


def build_base_feature_sets(df: pd.DataFrame) -> Dict[str, Tuple[List[str], List[str]]]:
    sc_cat = [c for c in ["circuit_id", "race_phase", "race_level_weather_condition", "future_phase_bucket"] if c in df.columns]
    sc_num = [
        "season",
        "round",
        "lap_number",
        "laps_remaining",
        "total_laps",
        "track_length",
        "pit_lane_time_loss",
        "overtaking_difficulty_index",
        "safety_car_probability_baseline",
        "track_temperature_used",
        "air_temperature_used",
        "rain_flag_used",
        "used_race_level_weather_flag",
        "drivers_running",
        "teams_running",
        "field_size",
        "min_gap_ahead",
        "median_gap_ahead",
        "mean_gap_ahead",
        "mean_gap_behind",
        "share_close_ahead_1s",
        "share_close_ahead_15s",
        "share_close_behind_1s",
        "share_clean_air",
        "mean_tyre_age",
        "std_tyre_age",
        "share_old_tyres_15",
        "share_pitting",
        "mean_gap_ahead_lag1",
        "median_gap_ahead_lag1",
        "mean_gap_behind_lag1",
        "share_close_ahead_1s_lag1",
        "share_close_ahead_15s_lag1",
        "share_close_behind_1s_lag1",
        "mean_tyre_age_lag1",
        "std_tyre_age_lag1",
        "share_old_tyres_15_lag1",
        "share_pitting_lag1",
        "share_clean_air_lag1",
        "drivers_running_lag1",
        "is_under_vsc_event_lag1",
        "is_under_red_flag_event_lag1",
        "is_under_virtual_safety_car_lag1",
        "hazard_step",
        "lap_number_future",
        "laps_remaining_future",
        "future_progress_frac",
    ]
    sc_num = [c for c in sc_num if c in df.columns]

    vsc_cat = [c for c in ["circuit_id", "race_phase", "race_level_weather_condition", "future_phase_bucket"] if c in df.columns]
    vsc_num = [
        "season",
        "round",
        "lap_number",
        "laps_remaining",
        "total_laps",
        "track_length",
        "pit_lane_time_loss",
        "overtaking_difficulty_index",
        "safety_car_probability_baseline",
        "track_temperature_used",
        "air_temperature_used",
        "rain_flag_used",
        "used_race_level_weather_flag",
        "drivers_running",
        "teams_running",
        "field_size",
        "min_gap_ahead",
        "median_gap_ahead",
        "mean_gap_ahead",
        "mean_gap_behind",
        "share_close_ahead_1s",
        "share_close_ahead_15s",
        "share_close_behind_1s",
        "share_clean_air",
        "mean_tyre_age",
        "std_tyre_age",
        "share_old_tyres_15",
        "share_pitting",
        "mean_gap_ahead_lag1",
        "median_gap_ahead_lag1",
        "mean_gap_behind_lag1",
        "share_close_ahead_1s_lag1",
        "share_close_ahead_15s_lag1",
        "share_close_behind_1s_lag1",
        "mean_tyre_age_lag1",
        "std_tyre_age_lag1",
        "share_old_tyres_15_lag1",
        "share_pitting_lag1",
        "share_clean_air_lag1",
        "drivers_running_lag1",
        "teams_running_lag1",
        "is_under_safety_car_event_lag1",
        "is_under_red_flag_event_lag1",
        "is_under_safety_car_lag1",
        "delta_mean_gap_ahead",
        "delta_mean_gap_behind",
        "delta_median_gap_ahead",
        "delta_share_close_ahead_1s",
        "delta_share_close_ahead_15s",
        "delta_share_close_behind_1s",
        "delta_share_pitting",
        "delta_mean_tyre_age",
        "delta_share_old_tyres_15",
        "delta_mean_lap_time",
        "delta_std_lap_time",
        "traffic_disruption_index",
        "traffic_disruption_delta",
        "gap_compression_index",
        "lap_time_volatility_index",
        "vsc_local_instability_index",
        "close_pack_x_pitting",
        "close_pack_x_old_tyres",
        "wet_x_instability",
        "pit_surge_x_gap_compression",
        "pack_pressure_balance",
        "hazard_step",
        "lap_number_future",
        "laps_remaining_future",
        "future_progress_frac",
    ]
    vsc_num = [c for c in vsc_num if c in df.columns]

    rf_cat = [c for c in ["circuit_id", "race_phase", "race_level_weather_condition", "rf_phase_bucket", "rf_weather_bucket", "future_phase_bucket"] if c in df.columns]
    rf_num = [
        "season",
        "round",
        "lap_number",
        "laps_remaining",
        "total_laps",
        "track_length",
        "pit_lane_time_loss",
        "overtaking_difficulty_index",
        "safety_car_probability_baseline",
        "track_temperature_used",
        "air_temperature_used",
        "rain_flag_used",
        "used_race_level_weather_flag",
        "drivers_running",
        "teams_running",
        "field_size",
        "mean_gap_ahead",
        "mean_gap_behind",
        "share_close_ahead_1s",
        "share_close_ahead_15s",
        "share_close_behind_1s",
        "share_pitting",
        "share_clean_air",
        "mean_tyre_age",
        "share_old_tyres_15",
        "rf_track_risk_proxy",
        "is_under_safety_car_event_lag1",
        "is_under_vsc_event_lag1",
        "is_under_red_flag_event_lag1",
        "is_under_safety_car_lag1",
        "is_under_virtual_safety_car_lag1",
        "traffic_disruption_index",
        "vsc_local_instability_index",
        "hazard_step",
        "lap_number_future",
        "laps_remaining_future",
        "future_progress_frac",
    ]
    rf_num = [c for c in rf_num if c in df.columns]

    return {
        "sc": (sc_cat, sc_num),
        "vsc": (vsc_cat, vsc_num),
        "rf": (rf_cat, rf_num),
    }


def build_hazard_dataset(
    race_lap: pd.DataFrame,
    event: str,
    max_horizon: int,
    cat_cols: List[str],
    num_cols: List[str],
) -> pd.DataFrame:
    df = add_event_start_flags(race_lap, event=event)
    start_col = f"{event}_start_this_lap"
    active_col = EVENT_ACTIVE_COLS[event]

    origin_df = df.loc[~df[active_col].astype(bool)].copy()
    origin_df = origin_df.sort_values(["race_id", "lap_number"]).reset_index(drop=True)

    use_cols = ["race_id", "lap_number", "total_laps", "laps_remaining"] + cat_cols + num_cols
    use_cols = [c for c in use_cols if c in origin_df.columns]
    use_cols = list(dict.fromkeys(use_cols))
    base = origin_df[use_cols].copy()

    future_start = df[["race_id", "lap_number", start_col]].copy()

    parts = []
    for k in range(1, max_horizon + 1):
        tmp = base.copy()
        tmp["hazard_step"] = k
        tmp["future_lap_number"] = tmp["lap_number"] + k

        tmp = tmp.merge(
            future_start.rename(columns={"lap_number": "future_lap_number", start_col: "y_hazard"}),
            on=["race_id", "future_lap_number"],
            how="left",
        )
        tmp["y_hazard"] = tmp["y_hazard"].fillna(0).astype(int)

        if "lap_number" in tmp.columns:
            tmp["lap_number_future"] = tmp["lap_number"] + k
        if "laps_remaining" in tmp.columns:
            tmp["laps_remaining_future"] = np.maximum(pd.to_numeric(tmp["laps_remaining"], errors="coerce") - k, 0)

        if "total_laps" in tmp.columns:
            denom = pd.to_numeric(tmp["total_laps"], errors="coerce").replace(0, np.nan)
            tmp["future_progress_frac"] = (pd.to_numeric(tmp["lap_number"], errors="coerce") + k) / denom
            tmp["future_phase_bucket"] = np.where(
                tmp["future_progress_frac"] <= 0.33,
                "early",
                np.where(tmp["future_progress_frac"] <= 0.66, "mid", "late"),
            )
        else:
            tmp["future_progress_frac"] = np.nan
            tmp["future_phase_bucket"] = "unknown"

        parts.append(tmp)

    hz = pd.concat(parts, axis=0, ignore_index=True)

    if "total_laps" in hz.columns:
        hz = hz.loc[hz["future_lap_number"] <= hz["total_laps"]].copy()

    return hz


def tune_xgb_with_optuna(X_train, y_train, X_val, y_val, seed: int, n_trials: int, scale_pos_weight: float) -> Dict:
    if XGBClassifier is None:
        raise RuntimeError("xgboost is not installed")
    if optuna is None:
        raise RuntimeError("optuna is not installed")

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 120, 450),
            "max_depth": trial.suggest_int("max_depth", 2, 3),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.08, log=True),
            "subsample": trial.suggest_float("subsample", 0.75, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.60, 0.90),
            "min_child_weight": trial.suggest_float("min_child_weight", 8.0, 35.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 2.0, 50.0, log=True),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.05, 10.0, log=True),
            "gamma": trial.suggest_float("gamma", 1.0, 8.0),
        }

        model = XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            random_state=seed,
            n_jobs=6,
            scale_pos_weight=scale_pos_weight,
            **params,
        )
        fit_xgb_with_early_stopping(model, X_train, y_train, X_val, y_val, rounds=35)
        p_val = model.predict_proba(X_val)[:, 1]
        p_train = model.predict_proba(X_train)[:, 1]

        pr_val = safe_pr_auc(y_val, p_val)
        pr_train = safe_pr_auc(y_train, p_train)
        brier_val = safe_brier(y_val, p_val)

        overfit_penalty = max(0.0, pr_train - pr_val)
        score = pr_val - 0.25 * overfit_penalty - 0.10 * brier_val
        return float(score)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    best = dict(study.best_params)
    best["best_objective"] = float(study.best_value)
    return best


def fit_logistic_model(X_train_raw, y_train, X_val_raw, y_val, X_test_raw, y_test, cat_cols, num_cols) -> Dict:
    pre = build_preprocessor(cat_cols, num_cols)
    pipe = Pipeline(
        steps=[
            ("preprocessor", pre),
            ("model", LogisticRegression(max_iter=3000, solver="lbfgs", class_weight="balanced")),
        ]
    )
    pipe.fit(X_train_raw, y_train)

    p_train = pipe.predict_proba(X_train_raw)[:, 1]
    p_val = pipe.predict_proba(X_val_raw)[:, 1]
    p_test = pipe.predict_proba(X_test_raw)[:, 1]

    calib = fit_platt_scaler(p_val, y_val)
    p_train_c = apply_platt(p_train, calib)
    p_val_c = apply_platt(p_val, calib)
    p_test_c = apply_platt(p_test, calib)

    return {
        "pipeline": pipe,
        "calibration": calib,
        "train": score_split(y_train, p_train_c),
        "val": score_split(y_val, p_val_c),
        "test": score_split(y_test, p_test_c),
        "p_train": p_train_c,
        "p_val": p_val_c,
        "p_test": p_test_c,
    }


def fit_xgb_model(X_train_raw, y_train, X_val_raw, y_val, X_test_raw, y_test, cat_cols, num_cols, seed: int, optuna_trials: int) -> Dict:
    if XGBClassifier is None:
        raise RuntimeError("xgboost is not installed")

    pre = build_preprocessor(cat_cols, num_cols)
    X_train = pre.fit_transform(X_train_raw)
    X_val = pre.transform(X_val_raw)
    X_test = pre.transform(X_test_raw)

    pos = float(y_train.sum())
    neg = float(len(y_train) - y_train.sum())
    spw = neg / max(pos, 1.0)

    if optuna_trials > 0 and optuna is not None:
        best = tune_xgb_with_optuna(X_train, y_train, X_val, y_val, seed=seed, n_trials=optuna_trials, scale_pos_weight=spw)
    else:
        best = {
            "n_estimators": 220,
            "max_depth": 2,
            "learning_rate": 0.03,
            "subsample": 0.85,
            "colsample_bytree": 0.75,
            "min_child_weight": 15.0,
            "reg_lambda": 8.0,
            "reg_alpha": 0.5,
            "gamma": 2.0,
            "best_objective": float("nan"),
        }

    model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        random_state=seed,
        n_jobs=6,
        scale_pos_weight=spw,
        n_estimators=int(best["n_estimators"]),
        max_depth=int(best["max_depth"]),
        learning_rate=float(best["learning_rate"]),
        subsample=float(best["subsample"]),
        colsample_bytree=float(best["colsample_bytree"]),
        min_child_weight=float(best["min_child_weight"]),
        reg_lambda=float(best["reg_lambda"]),
        reg_alpha=float(best["reg_alpha"]),
        gamma=float(best["gamma"]),
    )

    fit_xgb_with_early_stopping(model, X_train, y_train, X_val, y_val, rounds=35)

    p_train = model.predict_proba(X_train)[:, 1]
    p_val = model.predict_proba(X_val)[:, 1]
    p_test = model.predict_proba(X_test)[:, 1]

    calib = fit_platt_scaler(p_val, y_val)
    p_train_c = apply_platt(p_train, calib)
    p_val_c = apply_platt(p_val, calib)
    p_test_c = apply_platt(p_test, calib)

    return {
        "preprocessor": pre,
        "model": model,
        "calibration": calib,
        "best_params": best,
        "scale_pos_weight": spw,
        "train": score_split(y_train, p_train_c),
        "val": score_split(y_val, p_val_c),
        "test": score_split(y_test, p_test_c),
        "p_train": p_train_c,
        "p_val": p_val_c,
        "p_test": p_test_c,
    }


def choose_model(log_res: Dict, xgb_res: Dict, favor: str = "balanced") -> str:
    log_val = log_res["val"]["pr_auc"]
    xgb_val = xgb_res["val"]["pr_auc"]

    log_brier = log_res["val"]["brier"]
    xgb_brier = xgb_res["val"]["brier"]

    log_gap = max(0.0, log_res["train"]["pr_auc"] - log_res["val"]["pr_auc"])
    xgb_gap = max(0.0, xgb_res["train"]["pr_auc"] - xgb_res["val"]["pr_auc"])

    if favor == "stability":
        log_score = log_val - 0.15 * log_gap - 0.12 * log_brier
        xgb_score = xgb_val - 0.45 * xgb_gap - 0.12 * xgb_brier
    else:
        log_score = log_val - 0.20 * log_gap - 0.10 * log_brier
        xgb_score = xgb_val - 0.35 * xgb_gap - 0.10 * xgb_brier

    return "logistic" if log_score >= xgb_score else "xgb"


def predict_all_logistic(pipe: Pipeline, calib: Dict[str, float], X_all_raw: pd.DataFrame) -> np.ndarray:
    p_raw = pipe.predict_proba(X_all_raw)[:, 1]
    return apply_platt(p_raw, calib)


def predict_all_xgb(pre, model, calib: Dict[str, float], X_all_raw: pd.DataFrame) -> np.ndarray:
    X_all = pre.transform(X_all_raw)
    p_raw = model.predict_proba(X_all)[:, 1]
    return apply_platt(p_raw, calib)


def step_to_cumulative(step_probs: np.ndarray) -> np.ndarray:
    surv_prev = np.cumprod(1.0 - step_probs, axis=1)
    cum = 1.0 - surv_prev
    return np.clip(cum, 0.0, 1.0)


def build_rf_prior_intensity_fallback(
    race_lap: pd.DataFrame,
    max_horizon: int,
    splits: SplitIds,
    alpha_global: float = 40.0,
    alpha_circuit: float = 30.0,
    alpha_phase: float = 20.0,
    wet_boost: float = 1.20,
    disruption_boost_scale: float = 0.15,
) -> Tuple[pd.DataFrame, Dict]:
    df = add_event_start_flags(race_lap, event="rf").copy()
    start_col = "rf_start_this_lap"

    train_mask = np.isin(df["race_id"].astype(str).to_numpy(), np.array(splits.train_races, dtype=str))
    train_df = df.loc[train_mask].copy()

    global_rate = float(train_df[start_col].mean()) if len(train_df) else 0.0

    rf_counts_per_race = (
        train_df.groupby("race_id", as_index=False)[start_col]
        .sum()
        .rename(columns={start_col: "rf_count_race"})
    )
    poisson_lambda_global = float(rf_counts_per_race["rf_count_race"].mean()) if len(rf_counts_per_race) else 0.0

    grp_circuit_phase_weather = (
        train_df.groupby(["circuit_id", "rf_phase_bucket", "rf_weather_bucket"], as_index=False)
        .agg(
            mean1=(start_col, "mean"),
            count1=(start_col, "count"),
        )
    )

    grp_circuit_phase = (
        train_df.groupby(["circuit_id", "rf_phase_bucket"], as_index=False)
        .agg(
            mean2=(start_col, "mean"),
            count2=(start_col, "count"),
        )
    )

    grp_circuit = (
        train_df.groupby(["circuit_id"], as_index=False)
        .agg(
            mean3=(start_col, "mean"),
            count3=(start_col, "count"),
        )
    )

    out = df[["race_id", "lap_number", "circuit_id", "rf_phase_bucket", "rf_weather_bucket"]].copy()
    out = out.merge(grp_circuit_phase_weather, on=["circuit_id", "rf_phase_bucket", "rf_weather_bucket"], how="left")
    out = out.merge(grp_circuit_phase, on=["circuit_id", "rf_phase_bucket"], how="left")
    out = out.merge(grp_circuit, on=["circuit_id"], how="left")

    p3 = ((out["mean3"].fillna(global_rate) * out["count3"].fillna(0.0)) + alpha_global * global_rate) / (out["count3"].fillna(0.0) + alpha_global)
    p2 = ((out["mean2"].fillna(p3) * out["count2"].fillna(0.0)) + alpha_circuit * p3) / (out["count2"].fillna(0.0) + alpha_circuit)
    p1 = ((out["mean1"].fillna(p2) * out["count1"].fillna(0.0)) + alpha_phase * p2) / (out["count1"].fillna(0.0) + alpha_phase)

    track_length = safe_numeric(df.get("track_length", pd.Series(5.0, index=df.index)), fill=5.0).to_numpy(dtype=float)
    track_length = np.where(track_length <= 0, 5.0, track_length)

    total_laps = safe_numeric(df.get("total_laps", pd.Series(60.0, index=df.index)), fill=60.0).to_numpy(dtype=float)
    total_laps = np.where(total_laps <= 0, 60.0, total_laps)

    lambda_per_lap_global = poisson_lambda_global / np.mean(total_laps) if len(total_laps) else 0.0
    poisson_step_prior = np.full(len(df), lambda_per_lap_global, dtype=float)

    wet_mult = np.where(df["rf_weather_bucket"].astype(str) == "wet", wet_boost, 1.0)

    disruption = safe_numeric(df.get("vsc_local_instability_index", pd.Series(0.0, index=df.index))).to_numpy(dtype=float)
    if np.nanstd(disruption) > 1e-12:
        disruption_z = (disruption - np.nanmean(disruption)) / np.nanstd(disruption)
    else:
        disruption_z = np.zeros(len(disruption), dtype=float)
    disruption_mult = 1.0 + disruption_boost_scale * np.clip(disruption_z, -1.5, 2.0)

    step_p = 0.70 * p1.to_numpy(dtype=float) + 0.30 * poisson_step_prior
    step_p = step_p * wet_mult * disruption_mult
    step_p = np.clip(step_p, 0.0, 0.20)

    step_mat = np.repeat(step_p.reshape(-1, 1), max_horizon, axis=1)
    cum = step_to_cumulative(step_mat)

    pred = df[["race_id", "lap_number"]].copy()
    for h in EVAL_HORIZONS_DEFAULT:
        if h <= max_horizon:
            pred[f"p_rf_next{h}"] = cum[:, h - 1]

    meta = {
        "mode": "rf_prior_poisson_fallback",
        "global_step_rate_train": global_rate,
        "poisson_lambda_race_train": poisson_lambda_global,
        "lambda_per_lap_global": float(lambda_per_lap_global),
        "alpha_global": alpha_global,
        "alpha_circuit": alpha_circuit,
        "alpha_phase": alpha_phase,
        "wet_boost": wet_boost,
        "disruption_boost_scale": disruption_boost_scale,
    }
    return pred, meta


def smooth_event_predictions(
    pred_df: pd.DataFrame,
    prefix: str,
    alpha: float,
    baseline: float,
) -> pd.DataFrame:
    out = pred_df.copy()
    cols = [c for c in out.columns if c.startswith(prefix)]
    beta = 1.0 - alpha

    for c in cols:
        out[c] = pd.to_numeric(out[c], errors="coerce")
        out[c] = (alpha * out[c] + beta * baseline).clip(0.0, 1.0)

    return out


def classification_default_rate(y_true: np.ndarray, threshold: float = 0.5) -> Dict:
    pred = (np.asarray(y_true) >= threshold).astype(int)
    return {
        "threshold": threshold,
        "positive_rate_at_threshold": float(np.mean(pred)),
    }


def horizon_specific_labels(hz: pd.DataFrame, max_horizon: int) -> pd.DataFrame:
    out = hz[["race_id", "lap_number"]].drop_duplicates().copy()
    for h in EVAL_HORIZONS_DEFAULT:
        if h > max_horizon:
            continue
        tmp = (
            hz.loc[hz["hazard_step"] <= h]
            .groupby(["race_id", "lap_number"], as_index=False)["y_hazard"]
            .max()
            .rename(columns={"y_hazard": f"y_next{h}"})
        )
        out = out.merge(tmp, on=["race_id", "lap_number"], how="left")
        out[f"y_next{h}"] = out[f"y_next{h}"].fillna(0).astype(int)
    return out


def horizon_eval_from_cumulative(pred_df: pd.DataFrame, label_df: pd.DataFrame, event: str, splits: SplitIds) -> Dict[str, Dict]:
    merged = pred_df.merge(label_df, on=["race_id", "lap_number"], how="left")
    test_mask = np.isin(merged["race_id"].astype(str).to_numpy(), np.array(splits.test_races, dtype=str))
    val_mask = np.isin(merged["race_id"].astype(str).to_numpy(), np.array(splits.val_races, dtype=str))
    train_mask = np.isin(merged["race_id"].astype(str).to_numpy(), np.array(splits.train_races, dtype=str))

    out = {}
    for h in EVAL_HORIZONS_DEFAULT:
        pcol = f"p_{event}_next{h}"
        ycol = f"y_next{h}"
        if pcol not in merged.columns or ycol not in merged.columns:
            continue

        out[f"next{h}"] = {
            "train": score_split(merged.loc[train_mask, ycol].to_numpy(dtype=int), merged.loc[train_mask, pcol].to_numpy(dtype=float)),
            "val": score_split(merged.loc[val_mask, ycol].to_numpy(dtype=int), merged.loc[val_mask, pcol].to_numpy(dtype=float)),
            "test": score_split(merged.loc[test_mask, ycol].to_numpy(dtype=int), merged.loc[test_mask, pcol].to_numpy(dtype=float)),
            "test_reliability": reliability_table(
                merged.loc[test_mask, ycol].to_numpy(dtype=int),
                merged.loc[test_mask, pcol].to_numpy(dtype=float),
                n_bins=10,
            ),
        }
    return out


def write_model_report_md(report_path: str, title: str, payload: Dict) -> None:
    lines: List[str] = []
    lines.append(f"# {title}")
    lines.append("")
    lines.append(f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
    lines.append("")

    def write_obj(name: str, value, level: int = 2) -> None:
        prefix = "#" * level
        lines.append(f"{prefix} {name}")
        lines.append("")
        if isinstance(value, (dict, list)):
            lines.append("```json")
            lines.append(json.dumps(value, indent=2))
            lines.append("```")
        else:
            lines.append(str(value))
        lines.append("")

    for key, value in payload.items():
        write_obj(key, value, level=2)

    with open(report_path, "w") as f:
        f.write("\n".join(lines))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_path", required=True)
    ap.add_argument("--outputs_dir", required=True)
    ap.add_argument("--models_dir", required=True)
    ap.add_argument("--events", default="sc,vsc,rf")
    ap.add_argument("--max_horizon", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--test_frac", type=float, default=0.20)
    ap.add_argument("--val_frac", type=float, default=0.20)
    ap.add_argument("--optuna_trials", type=int, default=60)

    ap.add_argument("--sc_optuna_trials", type=int, default=60)
    ap.add_argument("--vsc_optuna_trials", type=int, default=100)

    ap.add_argument("--rf_force_prior", type=int, default=1)
    ap.add_argument("--rf_min_test_pos_for_supervised", type=int, default=15)

    ap.add_argument("--vsc_smooth_alpha", type=float, default=0.70)
    ap.add_argument("--vsc_smooth_baseline", type=float, default=0.007)
    ap.add_argument("--sc_smooth_alpha", type=float, default=1.0)
    ap.add_argument("--sc_smooth_baseline", type=float, default=0.0)

    args = ap.parse_args()

    ensure_dir(args.outputs_dir)
    ensure_dir(args.models_dir)

    events = parse_csv_list(args.events)
    for ev in events:
        if ev not in EVENTS:
            raise ValueError(f"Unsupported event: {ev}")

    ts = now_ts()

    run_meta = {
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
        "script": "train_model2_vsc_rf_rebuild.py",
        "input_path": args.input_path,
        "outputs_dir": args.outputs_dir,
        "models_dir": args.models_dir,
        "seed": args.seed,
        "events": events,
        "max_horizon": args.max_horizon,
        "test_frac": args.test_frac,
        "val_frac": args.val_frac,
        "optuna_trials_default": args.optuna_trials,
        "sc_optuna_trials": args.sc_optuna_trials,
        "vsc_optuna_trials": args.vsc_optuna_trials,
        "rf_force_prior": int(args.rf_force_prior),
        "rf_min_test_pos_for_supervised": args.rf_min_test_pos_for_supervised,
        "vsc_smooth_alpha": args.vsc_smooth_alpha,
        "vsc_smooth_baseline": args.vsc_smooth_baseline,
        "sc_smooth_alpha": args.sc_smooth_alpha,
        "sc_smooth_baseline": args.sc_smooth_baseline,
    }
    run_meta_path = os.path.join(args.outputs_dir, f"model2_run_meta_{ts}.json")
    with open(run_meta_path, "w") as f:
        json.dump(run_meta, f, indent=2)

    master = pd.read_csv(args.input_path, low_memory=False)
    race_lap = aggregate_race_lap(master)

    splits = make_group_split(
        race_ids=race_lap["race_id"].astype(str).to_numpy(),
        seed=args.seed,
        test_frac=args.test_frac,
        val_frac=args.val_frac,
    )
    split_json_path = os.path.join(args.models_dir, f"model2_hazard_race_splits_{ts}.json")
    with open(split_json_path, "w") as f:
        json.dump(
            {
                "train_races": splits.train_races,
                "val_races": splits.val_races,
                "test_races": splits.test_races,
            },
            f,
            indent=2,
        )

    feature_sets = build_base_feature_sets(race_lap)

    race_lap_preds = race_lap[["race_id", "lap_number"]].drop_duplicates().copy()
    metrics_all: List[Dict] = []
    summary_rows: List[Dict] = []

    for event in events:
        if event == "rf" and int(args.rf_force_prior) == 1:
            rf_pred, rf_meta = build_rf_prior_intensity_fallback(
                race_lap=race_lap,
                max_horizon=args.max_horizon,
                splits=splits,
            )
            race_lap_preds = race_lap_preds.merge(rf_pred, on=["race_id", "lap_number"], how="left")
            metrics_all.append(
                {
                    "event": "rf",
                    "mode": "prior_poisson_fallback",
                    "note": "RF handled as prior plus Poisson informed fallback due to extreme scarcity and instability.",
                    "rf_meta": rf_meta,
                }
            )
            for h in EVAL_HORIZONS_DEFAULT:
                if h <= args.max_horizon:
                    summary_rows.append(
                        {
                            "event": "rf",
                            "horizon": h,
                            "winner": "prior_poisson_fallback",
                            "train_pr_auc": np.nan,
                            "val_pr_auc": np.nan,
                            "test_pr_auc": np.nan,
                            "test_brier": np.nan,
                            "test_logloss": np.nan,
                        }
                    )
            continue

        cat_cols, num_cols = feature_sets[event]

        hz = build_hazard_dataset(
            race_lap=race_lap,
            event=event,
            max_horizon=args.max_horizon,
            cat_cols=cat_cols,
            num_cols=num_cols,
        )

        feature_cols = cat_cols + num_cols
        X_all = hz[feature_cols].copy()
        for c in cat_cols:
            X_all[c] = X_all[c].astype(str)
        for c in num_cols:
            X_all[c] = pd.to_numeric(X_all[c], errors="coerce")

        y_all = hz["y_hazard"].astype(int).to_numpy()

        tr = np.isin(hz["race_id"].astype(str).to_numpy(), np.array(splits.train_races, dtype=str))
        va = np.isin(hz["race_id"].astype(str).to_numpy(), np.array(splits.val_races, dtype=str))
        te = np.isin(hz["race_id"].astype(str).to_numpy(), np.array(splits.test_races, dtype=str))

        X_train = X_all.loc[tr].copy()
        X_val = X_all.loc[va].copy()
        X_test = X_all.loc[te].copy()

        y_train = y_all[tr]
        y_val = y_all[va]
        y_test = y_all[te]

        if event == "rf":
            pos_test = int(y_test.sum())
            if pos_test < args.rf_min_test_pos_for_supervised:
                rf_pred, rf_meta = build_rf_prior_intensity_fallback(
                    race_lap=race_lap,
                    max_horizon=args.max_horizon,
                    splits=splits,
                )
                race_lap_preds = race_lap_preds.merge(rf_pred, on=["race_id", "lap_number"], how="left")
                metrics_all.append(
                    {
                        "event": "rf",
                        "mode": "prior_poisson_fallback",
                        "note": f"RF fallback used because step level test positives were only {pos_test}.",
                        "rf_meta": rf_meta,
                    }
                )
                for h in EVAL_HORIZONS_DEFAULT:
                    if h <= args.max_horizon:
                        summary_rows.append(
                            {
                                "event": "rf",
                                "horizon": h,
                                "winner": "prior_poisson_fallback",
                                "train_pr_auc": np.nan,
                                "val_pr_auc": np.nan,
                                "test_pr_auc": np.nan,
                                "test_brier": np.nan,
                                "test_logloss": np.nan,
                            }
                        )
                continue

        event_optuna_trials = args.optuna_trials
        if event == "sc":
            event_optuna_trials = args.sc_optuna_trials
        elif event == "vsc":
            event_optuna_trials = args.vsc_optuna_trials

        log_res = fit_logistic_model(
            X_train_raw=X_train,
            y_train=y_train,
            X_val_raw=X_val,
            y_val=y_val,
            X_test_raw=X_test,
            y_test=y_test,
            cat_cols=cat_cols,
            num_cols=num_cols,
        )

        xgb_res = fit_xgb_model(
            X_train_raw=X_train,
            y_train=y_train,
            X_val_raw=X_val,
            y_val=y_val,
            X_test_raw=X_test,
            y_test=y_test,
            cat_cols=cat_cols,
            num_cols=num_cols,
            seed=args.seed,
            optuna_trials=event_optuna_trials,
        )

        favor = "balanced" if event == "sc" else "stability"
        winner = choose_model(log_res, xgb_res, favor=favor)

        if winner == "xgb":
            p_all_step = predict_all_xgb(
                pre=xgb_res["preprocessor"],
                model=xgb_res["model"],
                calib=xgb_res["calibration"],
                X_all_raw=X_all,
            )
            p_test_step = xgb_res["p_test"]
        else:
            p_all_step = predict_all_logistic(
                pipe=log_res["pipeline"],
                calib=log_res["calibration"],
                X_all_raw=X_all,
            )
            p_test_step = log_res["p_test"]

        hz_pred = hz[["race_id", "lap_number", "hazard_step"]].copy()
        hz_pred["p_step"] = p_all_step

        pivot = (
            hz_pred.pivot_table(index=["race_id", "lap_number"], columns="hazard_step", values="p_step", aggfunc="first")
            .sort_index(axis=1)
        )
        for k in range(1, args.max_horizon + 1):
            if k not in pivot.columns:
                pivot[k] = np.nan
        pivot = pivot[[k for k in range(1, args.max_horizon + 1)]].fillna(0.0)

        step_mat = pivot.to_numpy(dtype=float)
        cum = step_to_cumulative(step_mat)

        pred_df = pivot.reset_index()[["race_id", "lap_number"]].copy()
        for h in EVAL_HORIZONS_DEFAULT:
            if h <= args.max_horizon:
                pred_df[f"p_{event}_next{h}"] = cum[:, h - 1]

        if event == "vsc":
            pred_df = smooth_event_predictions(
                pred_df=pred_df,
                prefix="p_vsc_",
                alpha=args.vsc_smooth_alpha,
                baseline=args.vsc_smooth_baseline,
            )
        if event == "sc" and args.sc_smooth_alpha < 0.999999:
            pred_df = smooth_event_predictions(
                pred_df=pred_df,
                prefix="p_sc_",
                alpha=args.sc_smooth_alpha,
                baseline=args.sc_smooth_baseline,
            )

        race_lap_preds = race_lap_preds.merge(pred_df, on=["race_id", "lap_number"], how="left")

        base = f"model2_hazard_{event}_{ts}"
        joblib.dump(log_res["pipeline"], os.path.join(args.models_dir, f"{base}_logistic.pkl"))
        joblib.dump(xgb_res["preprocessor"], os.path.join(args.models_dir, f"{base}_xgb_preprocessor.pkl"))
        joblib.dump(xgb_res["model"], os.path.join(args.models_dir, f"{base}_xgb.pkl"))
        with open(os.path.join(args.models_dir, f"{base}_logistic_calibration.json"), "w") as f:
            json.dump(log_res["calibration"], f, indent=2)
        with open(os.path.join(args.models_dir, f"{base}_xgb_calibration.json"), "w") as f:
            json.dump(xgb_res["calibration"], f, indent=2)

        label_df = horizon_specific_labels(hz, max_horizon=args.max_horizon)
        horizon_metrics = horizon_eval_from_cumulative(pred_df, label_df, event=event, splits=splits)

        with open(os.path.join(args.models_dir, f"{base}_meta.json"), "w") as f:
            json.dump(
                {
                    "event": event,
                    "mode": "hazard_model",
                    "winner": winner,
                    "cat_cols": cat_cols,
                    "num_cols": num_cols,
                    "feature_cols": feature_cols,
                    "max_horizon": args.max_horizon,
                    "vsc_smoothed": bool(event == "vsc"),
                    "vsc_smooth_alpha": float(args.vsc_smooth_alpha) if event == "vsc" else None,
                    "vsc_smooth_baseline": float(args.vsc_smooth_baseline) if event == "vsc" else None,
                    "sc_smoothed": bool(event == "sc" and args.sc_smooth_alpha < 0.999999),
                    "sc_smooth_alpha": float(args.sc_smooth_alpha) if event == "sc" else None,
                    "sc_smooth_baseline": float(args.sc_smooth_baseline) if event == "sc" else None,
                },
                f,
                indent=2,
            )

        event_metrics = {
            "event": event,
            "mode": "hazard_model",
            "winner": winner,
            "feature_cols": feature_cols,
            "step_level_logistic": {
                "train": log_res["train"],
                "val": log_res["val"],
                "test": log_res["test"],
            },
            "step_level_xgb": {
                "train": xgb_res["train"],
                "val": xgb_res["val"],
                "test": xgb_res["test"],
                "best_params": xgb_res["best_params"],
                "scale_pos_weight": xgb_res["scale_pos_weight"],
            },
            "winner_step_test_reliability": reliability_table(y_test, p_test_step, n_bins=10),
            "horizon_metrics": horizon_metrics,
            "postprocess": {
                "vsc_smoothed": bool(event == "vsc"),
                "vsc_alpha": float(args.vsc_smooth_alpha) if event == "vsc" else None,
                "vsc_baseline": float(args.vsc_smooth_baseline) if event == "vsc" else None,
                "sc_smoothed": bool(event == "sc" and args.sc_smooth_alpha < 0.999999),
                "sc_alpha": float(args.sc_smooth_alpha) if event == "sc" else None,
                "sc_baseline": float(args.sc_smooth_baseline) if event == "sc" else None,
            },
            "split_positive_counts_step": {
                "train": int(y_train.sum()),
                "val": int(y_val.sum()),
                "test": int(y_test.sum()),
            },
        }
        metrics_all.append(event_metrics)

        for h in EVAL_HORIZONS_DEFAULT:
            hk = f"next{h}"
            if hk not in horizon_metrics:
                continue
            summary_rows.append(
                {
                    "event": event,
                    "horizon": h,
                    "winner": winner,
                    "train_pr_auc": horizon_metrics[hk]["train"]["pr_auc"],
                    "val_pr_auc": horizon_metrics[hk]["val"]["pr_auc"],
                    "test_pr_auc": horizon_metrics[hk]["test"]["pr_auc"],
                    "test_brier": horizon_metrics[hk]["test"]["brier"],
                    "test_logloss": horizon_metrics[hk]["test"]["logloss"],
                    "test_roc_auc": horizon_metrics[hk]["test"]["roc_auc"],
                }
            )

    race_pred_path = os.path.join(args.outputs_dir, f"model2_hazard_racelap_predictions_{ts}.csv")
    race_lap_preds.to_csv(race_pred_path, index=False)

    merged = master.merge(race_lap_preds, on=["race_id", "lap_number"], how="left")
    prob_cols = [c for c in race_lap_preds.columns if c.startswith("p_")]

    for c in prob_cols:
        merged[f"{c}_was_nan"] = merged[c].isna().astype(int)
        merged[c] = pd.to_numeric(merged[c], errors="coerce")
        merged[c] = merged.groupby("race_id")[c].ffill()
        merged[c] = merged.groupby("race_id")[c].bfill()
        med = float(merged[c].median()) if merged[c].notna().any() else 0.0
        if not np.isfinite(med):
            med = 0.0
        merged[c] = merged[c].fillna(med).clip(0.0, 1.0)

    if prob_cols:
        merged["model2_event_prob_imputed_any"] = merged[[f"{c}_was_nan" for c in prob_cols]].max(axis=1).astype(int)

    merged_path = os.path.join(args.outputs_dir, f"lap_level_with_model1_model2_hazard_rebuild_{ts}.csv")
    merged.to_csv(merged_path, index=False)

    metrics_path = os.path.join(args.outputs_dir, f"model2_hazard_metrics_rebuild_{ts}.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics_all, f, indent=2)

    summary_df = pd.DataFrame(summary_rows)
    summary_csv_path = os.path.join(args.outputs_dir, f"model2_hazard_summary_rebuild_{ts}.csv")
    summary_df.to_csv(summary_csv_path, index=False)

    report_payload = {
        "run_meta": run_meta,
        "split_json_path": split_json_path,
        "race_pred_path": race_pred_path,
        "merged_path": merged_path,
        "summary_csv_path": summary_csv_path,
        "metrics_path": metrics_path,
        "metrics": metrics_all,
        "notes": [
            "SC kept as supervised hazard model with minor optional smoothing support.",
            "VSC rebuilt with local instability, gap compression, pitting surge, and disturbance features.",
            "RF handled through prior plus Poisson informed intensity fallback unless explicitly changed later.",
            "All models use race_id group split discipline to avoid race leakage.",
            "All probability outputs are merged back into the lap level master table and imputation flags are recorded.",
        ],
    }
    report_path = os.path.join(args.outputs_dir, f"model2_hazard_report_rebuild_{ts}.md")
    write_model_report_md(
        report_path=report_path,
        title="Model 2 Hazard Rebuild Report",
        payload=report_payload,
    )

    print("Saved run meta:", run_meta_path)
    print("Saved race-lap hazard predictions:", race_pred_path)
    print("Saved merged output:", merged_path)
    print("Saved metrics:", metrics_path)
    print("Saved summary CSV:", summary_csv_path)
    print("Saved report:", report_path)


if __name__ == "__main__":
    main()