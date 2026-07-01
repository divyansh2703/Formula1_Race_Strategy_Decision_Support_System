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
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
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


ALLOWED_COMPOUNDS_DEFAULT = ["SOFT", "MEDIUM", "HARD", "INTERMEDIATE", "WET"]


@dataclass
class SplitIds:
    train_races: List[str]
    val_races: List[str]
    test_races: List[str]


def now_ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def parse_horizons(s: str) -> List[int]:
    vals = sorted({int(x.strip()) for x in s.split(",") if x.strip()})
    if not vals:
        raise ValueError("No horizons provided")
    return vals


def parse_csv_list(s: str) -> List[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


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
    lr.fit(z, y_val.astype(int))
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


def add_pit_horizon_labels(df: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """
    At lap L:
    y_pit_nextH = 1 if driver pits in any of laps L+1 ... L+H
    Current lap pit is excluded to avoid same lap action leakage.
    """
    out = df.sort_values(["race_id", "driver_id", "lap_number"]).copy()

    if "pitted_this_lap_flag" not in out.columns:
        raise ValueError("Missing pitted_this_lap_flag")

    def group_label(s: pd.Series) -> pd.Series:
        arr = s.fillna(0).astype(int).to_numpy()
        n = len(arr)
        y = np.zeros(n, dtype=int)
        for i in range(n):
            lo = i + 1
            hi = min(n, i + horizon + 1)
            if lo < hi and np.any(arr[lo:hi] == 1):
                y[i] = 1
        return pd.Series(y, index=s.index)

    out[f"y_pit_next{horizon}"] = (
        out.groupby(["race_id", "driver_id"], sort=False)["pitted_this_lap_flag"]
        .apply(group_label)
        .reset_index(level=[0, 1], drop=True)
        .astype(int)
    )

    return out


def add_model3_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    numeric_candidates = [
        "lap_number",
        "laps_remaining",
        "stint_number",
        "tyre_age",
        "position",
        "gap_ahead",
        "gap_behind",
        "clean_air_flag",
        "track_temperature_used",
        "air_temperature_used",
        "rain_flag_used",
        "pit_lane_time_loss",
        "overtaking_difficulty_index",
        "field_size",
        "cars_ahead_count",
        "cars_behind_count",
        "avg_tyre_age_ahead",
        "avg_tyre_age_behind",
        "previous_lap_time",
        "lap_time",
        "expected_lap_time",
        "expected_lap_time_fresh",
        "tyre_degradation_delta",
        "tyre_degradation_per_lap",
        "lap_time_sigma",
        "lap_time_variance",
        "p_sc_next3",
        "p_sc_next5",
        "p_sc_next10",
        "p_vsc_next3",
        "p_vsc_next5",
        "p_vsc_next10",
        "p_rf_next3",
        "p_rf_next5",
        "p_rf_next10",
        "total_laps",
        "safety_car_probability_baseline",
    ]
    for c in numeric_candidates:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")

    if "tyre_compound" in out.columns:
        out["tyre_compound"] = out["tyre_compound"].astype(str).fillna("UNKNOWN")

    if "race_phase" in out.columns:
        out["race_phase"] = out["race_phase"].astype(str).fillna("unknown")

    if "race_level_weather_condition" in out.columns:
        out["race_level_weather_condition"] = out["race_level_weather_condition"].astype(str).fillna("unknown")

    # Strategy pressure style features
    if {"lap_time", "expected_lap_time"}.issubset(out.columns):
        out["pace_loss_vs_expected"] = out["lap_time"] - out["expected_lap_time"]

    if {"previous_lap_time", "expected_lap_time"}.issubset(out.columns):
        out["prev_pace_loss_vs_expected"] = out["previous_lap_time"] - out["expected_lap_time"]

    if {"expected_lap_time", "expected_lap_time_fresh"}.issubset(out.columns):
        out["fresh_vs_current_expected_gap"] = out["expected_lap_time"] - out["expected_lap_time_fresh"]

    if {"tyre_age", "tyre_degradation_per_lap"}.issubset(out.columns):
        out["deg_pressure"] = out["tyre_age"] * out["tyre_degradation_per_lap"]

    if {"tyre_age", "tyre_degradation_delta"}.issubset(out.columns):
        out["deg_delta_pressure"] = out["tyre_age"] * out["tyre_degradation_delta"]

    if {"gap_ahead", "pit_lane_time_loss"}.issubset(out.columns):
        out["ahead_within_pitloss"] = (
            pd.to_numeric(out["gap_ahead"], errors="coerce")
            <= pd.to_numeric(out["pit_lane_time_loss"], errors="coerce")
        ).astype(float)

    if {"gap_behind", "pit_lane_time_loss"}.issubset(out.columns):
        out["behind_within_pitloss"] = (
            pd.to_numeric(out["gap_behind"], errors="coerce")
            <= pd.to_numeric(out["pit_lane_time_loss"], errors="coerce")
        ).astype(float)

    if {"gap_ahead", "gap_behind"}.issubset(out.columns):
        out["surrounded_pressure"] = (
            pd.to_numeric(out["gap_ahead"], errors="coerce").fillna(999.0).clip(upper=10.0)
            + pd.to_numeric(out["gap_behind"], errors="coerce").fillna(999.0).clip(upper=10.0)
        )

    if {"cars_ahead_count", "cars_behind_count"}.issubset(out.columns):
        out["local_traffic_count"] = (
            pd.to_numeric(out["cars_ahead_count"], errors="coerce").fillna(0.0)
            + pd.to_numeric(out["cars_behind_count"], errors="coerce").fillna(0.0)
        )

    if {"avg_tyre_age_ahead", "tyre_age"}.issubset(out.columns):
        out["tyre_age_vs_ahead"] = out["tyre_age"] - out["avg_tyre_age_ahead"]

    if {"avg_tyre_age_behind", "tyre_age"}.issubset(out.columns):
        out["tyre_age_vs_behind"] = out["tyre_age"] - out["avg_tyre_age_behind"]

    if {"lap_number", "total_laps"}.issubset(out.columns):
        denom = out["total_laps"].replace(0, np.nan)
        out["race_progress_frac"] = out["lap_number"] / denom

    if {"tyre_age", "laps_remaining"}.issubset(out.columns):
        denom = (out["tyre_age"] + out["laps_remaining"]).replace(0, np.nan)
        out["stint_progress_proxy"] = out["tyre_age"] / denom

    # Neutralisation summary
    hazard_cols_short = [c for c in ["p_sc_next3", "p_vsc_next3", "p_rf_next3"] if c in out.columns]
    if hazard_cols_short:
        out["neutralisation_risk_next3_max"] = out[hazard_cols_short].max(axis=1)
        out["neutralisation_risk_next3_sum"] = out[hazard_cols_short].sum(axis=1)

    hazard_cols_med = [c for c in ["p_sc_next5", "p_vsc_next5", "p_rf_next5"] if c in out.columns]
    if hazard_cols_med:
        out["neutralisation_risk_next5_max"] = out[hazard_cols_med].max(axis=1)

    # First lap after stop is very special
    if "tyre_age" in out.columns:
        out["fresh_tyre_flag"] = (out["tyre_age"].fillna(-1) <= 1).astype(float)

    if "stint_number" in out.columns:
        out["is_first_stint_flag"] = (pd.to_numeric(out["stint_number"], errors="coerce").fillna(0) <= 1).astype(float)
        out["is_late_stint_flag"] = (pd.to_numeric(out["stint_number"], errors="coerce").fillna(0) >= 3).astype(float)

    return out


def tune_xgb_binary_with_optuna(X_train, y_train, X_val, y_val, seed: int, n_trials: int, scale_pos_weight: float) -> Dict:
    if XGBClassifier is None:
        raise RuntimeError("xgboost is not installed")
    if optuna is None:
        raise RuntimeError("optuna is not installed")

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 150, 500),
            "max_depth": trial.suggest_int("max_depth", 2, 4),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.08, log=True),
            "subsample": trial.suggest_float("subsample", 0.75, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.65, 0.95),
            "min_child_weight": trial.suggest_float("min_child_weight", 5.0, 30.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 1.0, 50.0, log=True),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
            "gamma": trial.suggest_float("gamma", 0.0, 8.0),
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

        p_train = model.predict_proba(X_train)[:, 1]
        p_val = model.predict_proba(X_val)[:, 1]

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


def tune_xgb_multiclass_with_optuna(X_train, y_train, X_val, y_val, num_class: int, seed: int, n_trials: int) -> Dict:
    if XGBClassifier is None:
        raise RuntimeError("xgboost is not installed")
    if optuna is None:
        raise RuntimeError("optuna is not installed")

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 150, 500),
            "max_depth": trial.suggest_int("max_depth", 2, 5),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.08, log=True),
            "subsample": trial.suggest_float("subsample", 0.75, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.65, 0.95),
            "min_child_weight": trial.suggest_float("min_child_weight", 3.0, 20.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 1.0, 50.0, log=True),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
            "gamma": trial.suggest_float("gamma", 0.0, 8.0),
        }

        model = XGBClassifier(
            objective="multi:softprob",
            num_class=num_class,
            eval_metric="mlogloss",
            tree_method="hist",
            random_state=seed,
            n_jobs=6,
            **params,
        )
        fit_xgb_with_early_stopping(model, X_train, y_train, X_val, y_val, rounds=35)

        p = model.predict_proba(X_val)
        eps = 1e-12
        mlogloss = -np.mean(np.log(np.clip(p[np.arange(len(y_val)), y_val], eps, 1.0)))
        return float(-mlogloss)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best = dict(study.best_params)
    best["best_objective_neg_logloss"] = float(study.best_value)
    return best


def fit_binary_xgb_model(
    X_train_raw,
    y_train,
    X_val_raw,
    y_val,
    X_test_raw,
    y_test,
    cat_cols,
    num_cols,
    seed: int,
    optuna_trials: int,
) -> Dict:
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
        best = tune_xgb_binary_with_optuna(
            X_train, y_train, X_val, y_val, seed=seed, n_trials=optuna_trials, scale_pos_weight=spw
        )
    else:
        best = {
            "n_estimators": 250,
            "max_depth": 3,
            "learning_rate": 0.03,
            "subsample": 0.85,
            "colsample_bytree": 0.80,
            "min_child_weight": 10.0,
            "reg_lambda": 8.0,
            "reg_alpha": 0.5,
            "gamma": 1.0,
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


def predict_all_binary_xgb(pre, model, calib: Dict[str, float], X_all_raw: pd.DataFrame) -> np.ndarray:
    X_all = pre.transform(X_all_raw)
    p_raw = model.predict_proba(X_all)[:, 1]
    return apply_platt(p_raw, calib)


def build_model3_feature_lists(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    cat_cols = [
        "driver_id",
        "team_id",
        "circuit_id",
        "season",
        "race_phase",
        "tyre_compound",
        "race_level_weather_condition",
    ]
    num_cols = [
        "lap_number",
        "laps_remaining",
        "stint_number",
        "tyre_age",
        "position",
        "gap_ahead",
        "gap_behind",
        "clean_air_flag",
        "track_temperature_used",
        "air_temperature_used",
        "rain_flag_used",
        "pit_lane_time_loss",
        "overtaking_difficulty_index",
        "field_size",
        "cars_ahead_count",
        "cars_behind_count",
        "avg_tyre_age_ahead",
        "avg_tyre_age_behind",
        "previous_lap_time",
        "lap_time",
        "expected_lap_time",
        "expected_lap_time_fresh",
        "tyre_degradation_delta",
        "tyre_degradation_per_lap",
        "lap_time_sigma",
        "lap_time_variance",
        "p_sc_next3",
        "p_sc_next5",
        "p_sc_next10",
        "p_vsc_next3",
        "p_vsc_next5",
        "p_vsc_next10",
        "p_rf_next3",
        "p_rf_next5",
        "p_rf_next10",
        "safety_car_probability_baseline",
        "pace_loss_vs_expected",
        "prev_pace_loss_vs_expected",
        "fresh_vs_current_expected_gap",
        "deg_pressure",
        "deg_delta_pressure",
        "ahead_within_pitloss",
        "behind_within_pitloss",
        "surrounded_pressure",
        "local_traffic_count",
        "tyre_age_vs_ahead",
        "tyre_age_vs_behind",
        "race_progress_frac",
        "stint_progress_proxy",
        "neutralisation_risk_next3_max",
        "neutralisation_risk_next3_sum",
        "neutralisation_risk_next5_max",
        "fresh_tyre_flag",
        "is_first_stint_flag",
        "is_late_stint_flag",
    ]

    cat_cols = [c for c in cat_cols if c in df.columns]
    num_cols = [c for c in num_cols if c in df.columns]

    return cat_cols, num_cols


def train_pit_timing_model_for_horizon(
    df: pd.DataFrame,
    horizon: int,
    splits: SplitIds,
    outputs_dir: str,
    models_dir: str,
    seed: int,
    optuna_trials: int,
    ts: str,
) -> Tuple[pd.DataFrame, Dict]:
    y_col = f"y_pit_next{horizon}"
    req = ["race_id", "driver_id", "lap_number", y_col]
    miss = [c for c in req if c not in df.columns]
    if miss:
        raise ValueError(f"Missing required columns for pit timing model: {miss}")

    cat_cols, num_cols = build_model3_feature_lists(df)

    use_cols = list(dict.fromkeys(["race_id", "driver_id", "lap_number"] + cat_cols + num_cols + [y_col]))
    work = df[use_cols].copy()

    train_df = work[work["race_id"].isin(splits.train_races)].copy()
    val_df = work[work["race_id"].isin(splits.val_races)].copy()
    test_df = work[work["race_id"].isin(splits.test_races)].copy()

    X_train_raw = train_df[cat_cols + num_cols].copy()
    y_train = train_df[y_col].astype(int).to_numpy()

    X_val_raw = val_df[cat_cols + num_cols].copy()
    y_val = val_df[y_col].astype(int).to_numpy()

    X_test_raw = test_df[cat_cols + num_cols].copy()
    y_test = test_df[y_col].astype(int).to_numpy()

    for c in cat_cols:
        X_train_raw[c] = X_train_raw[c].astype(str)
        X_val_raw[c] = X_val_raw[c].astype(str)
        X_test_raw[c] = X_test_raw[c].astype(str)

    for c in num_cols:
        X_train_raw[c] = pd.to_numeric(X_train_raw[c], errors="coerce")
        X_val_raw[c] = pd.to_numeric(X_val_raw[c], errors="coerce")
        X_test_raw[c] = pd.to_numeric(X_test_raw[c], errors="coerce")

    res = fit_binary_xgb_model(
        X_train_raw=X_train_raw,
        y_train=y_train,
        X_val_raw=X_val_raw,
        y_val=y_val,
        X_test_raw=X_test_raw,
        y_test=y_test,
        cat_cols=cat_cols,
        num_cols=num_cols,
        seed=seed,
        optuna_trials=optuna_trials,
    )

    # Save artifacts
    base = f"model3a_pit_next{horizon}_{ts}"
    joblib.dump(res["preprocessor"], os.path.join(models_dir, f"{base}_preprocessor.pkl"))
    joblib.dump(res["model"], os.path.join(models_dir, f"{base}_xgb.pkl"))
    with open(os.path.join(models_dir, f"{base}_calibration.json"), "w") as f:
        json.dump(res["calibration"], f, indent=2)

    metrics = {
        "target": y_col,
        "horizon": horizon,
        "mode": "binary_pit_timing",
        "cat_cols": cat_cols,
        "num_cols": num_cols,
        "scale_pos_weight": res["scale_pos_weight"],
        "best_params": res["best_params"],
        "train": res["train"],
        "val": res["val"],
        "test": res["test"],
        "reliability_test": reliability_table(y_test, res["p_test"], n_bins=10),
    }

    with open(os.path.join(outputs_dir, f"{base}_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    # Save split predictions
    def pred_df(base_df: pd.DataFrame, y: np.ndarray, p: np.ndarray, split_name: str) -> pd.DataFrame:
        out = base_df[["race_id", "driver_id", "lap_number"]].copy()
        out["split"] = split_name
        out["y_true"] = y.astype(int)
        out["p_hat"] = p.astype(float)
        return out

    train_pred = pred_df(train_df, y_train, res["p_train"], "train")
    val_pred = pred_df(val_df, y_val, res["p_val"], "val")
    test_pred = pred_df(test_df, y_test, res["p_test"], "test")

    train_pred.to_csv(os.path.join(outputs_dir, f"{base}_preds_train.csv"), index=False)
    val_pred.to_csv(os.path.join(outputs_dir, f"{base}_preds_val.csv"), index=False)
    test_pred.to_csv(os.path.join(outputs_dir, f"{base}_preds_test.csv"), index=False)

    # Inference on all rows
    X_all_raw = work[cat_cols + num_cols].copy()
    for c in cat_cols:
        X_all_raw[c] = X_all_raw[c].astype(str)
    for c in num_cols:
        X_all_raw[c] = pd.to_numeric(X_all_raw[c], errors="coerce")

    p_all = predict_all_binary_xgb(
        pre=res["preprocessor"],
        model=res["model"],
        calib=res["calibration"],
        X_all_raw=X_all_raw,
    )

    all_probs = work[["race_id", "driver_id", "lap_number"]].copy()
    all_probs[f"p_pit_next{horizon}"] = p_all.astype(float)

    return all_probs, metrics


def train_compound_choice_model(
    df: pd.DataFrame,
    splits: SplitIds,
    outputs_dir: str,
    models_dir: str,
    seed: int,
    optuna_trials: int,
    ts: str,
    allowed_compounds: List[str],
) -> Tuple[pd.DataFrame, Dict]:
    if XGBClassifier is None:
        raise RuntimeError("xgboost is not installed")

    if "pitted_this_lap_flag" not in df.columns:
        raise ValueError("Missing pitted_this_lap_flag")
    if "tyre_compound_after" not in df.columns:
        raise ValueError("Missing tyre_compound_after")

    pit = df[df["pitted_this_lap_flag"].fillna(0).astype(int) == 1].copy()
    pit["tyre_compound_after"] = pit["tyre_compound_after"].astype(str)
    pit["tyre_compound_after"] = pit["tyre_compound_after"].replace({"nan": np.nan, "None": np.nan, "": np.nan})
    pit = pit.dropna(subset=["tyre_compound_after"]).copy()

    allowed_set = set(allowed_compounds)
    pit = pit[pit["tyre_compound_after"].isin(allowed_set)].copy()

    if len(pit) < 200:
        raise ValueError(f"Too few valid pit rows for compound model: {len(pit)}")

    cat_cols, num_cols = build_model3_feature_lists(pit)

    use_cols = list(dict.fromkeys(["race_id", "driver_id", "lap_number"] + cat_cols + num_cols + ["tyre_compound_after"]))
    pit = pit[use_cols].copy()

    train_df = pit[pit["race_id"].isin(splits.train_races)].copy()
    val_df = pit[pit["race_id"].isin(splits.val_races)].copy()
    test_df = pit[pit["race_id"].isin(splits.test_races)].copy()

    classes = sorted(pd.unique(pit["tyre_compound_after"]))
    class_to_idx = {c: i for i, c in enumerate(classes)}

    X_train_raw = train_df[cat_cols + num_cols].copy()
    y_train = np.array([class_to_idx[c] for c in train_df["tyre_compound_after"].astype(str)], dtype=int)

    X_val_raw = val_df[cat_cols + num_cols].copy()
    y_val = np.array([class_to_idx[c] for c in val_df["tyre_compound_after"].astype(str)], dtype=int)

    X_test_raw = test_df[cat_cols + num_cols].copy()
    y_test = np.array([class_to_idx[c] for c in test_df["tyre_compound_after"].astype(str)], dtype=int)

    for c in cat_cols:
        X_train_raw[c] = X_train_raw[c].astype(str)
        X_val_raw[c] = X_val_raw[c].astype(str)
        X_test_raw[c] = X_test_raw[c].astype(str)

    for c in num_cols:
        X_train_raw[c] = pd.to_numeric(X_train_raw[c], errors="coerce")
        X_val_raw[c] = pd.to_numeric(X_val_raw[c], errors="coerce")
        X_test_raw[c] = pd.to_numeric(X_test_raw[c], errors="coerce")

    pre = build_preprocessor(cat_cols, num_cols)
    X_train = pre.fit_transform(X_train_raw)
    X_val = pre.transform(X_val_raw)
    X_test = pre.transform(X_test_raw)

    if optuna_trials > 0 and optuna is not None:
        best = tune_xgb_multiclass_with_optuna(
            X_train, y_train, X_val, y_val, num_class=len(classes), seed=seed, n_trials=optuna_trials
        )
    else:
        best = {
            "n_estimators": 250,
            "max_depth": 3,
            "learning_rate": 0.03,
            "subsample": 0.85,
            "colsample_bytree": 0.80,
            "min_child_weight": 8.0,
            "reg_lambda": 8.0,
            "reg_alpha": 0.5,
            "gamma": 1.0,
            "best_objective_neg_logloss": float("nan"),
        }

    model = XGBClassifier(
        objective="multi:softprob",
        num_class=len(classes),
        eval_metric="mlogloss",
        tree_method="hist",
        random_state=seed,
        n_jobs=6,
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

    p_train = model.predict_proba(X_train)
    p_val = model.predict_proba(X_val)
    p_test = model.predict_proba(X_test)

    eps = 1e-12
    train_logloss = float(-np.mean(np.log(np.clip(p_train[np.arange(len(y_train)), y_train], eps, 1.0))))
    val_logloss = float(-np.mean(np.log(np.clip(p_val[np.arange(len(y_val)), y_val], eps, 1.0))))
    test_logloss = float(-np.mean(np.log(np.clip(p_test[np.arange(len(y_test)), y_test], eps, 1.0))))

    metrics = {
        "target": "tyre_compound_after",
        "mode": "compound_choice_conditional_on_pit",
        "allowed_compounds": allowed_compounds,
        "classes": classes,
        "cat_cols": cat_cols,
        "num_cols": num_cols,
        "best_params": best,
        "n_train": int(len(train_df)),
        "n_val": int(len(val_df)),
        "n_test": int(len(test_df)),
        "train_logloss": train_logloss,
        "val_logloss": val_logloss,
        "test_logloss": test_logloss,
    }

    base = f"model3b_compound_choice_{ts}"
    joblib.dump(pre, os.path.join(models_dir, f"{base}_preprocessor.pkl"))
    joblib.dump(model, os.path.join(models_dir, f"{base}_xgb.pkl"))
    with open(os.path.join(models_dir, f"{base}_classes.json"), "w") as f:
        json.dump({"classes": classes}, f, indent=2)
    with open(os.path.join(outputs_dir, f"{base}_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    # Inference on all rows
    all_base = df[["race_id", "driver_id", "lap_number"]].copy()
    X_all_raw = df[cat_cols + num_cols].copy()

    for c in cat_cols:
        X_all_raw[c] = X_all_raw[c].astype(str)
    for c in num_cols:
        X_all_raw[c] = pd.to_numeric(X_all_raw[c], errors="coerce")

    X_all = pre.transform(X_all_raw)
    p_all = model.predict_proba(X_all)

    out = all_base.copy()
    for i, cls in enumerate(classes):
        out[f"p_compound_{cls}"] = p_all[:, i].astype(float)

    # Also give most likely compound
    pred_idx = np.argmax(p_all, axis=1)
    out["compound_choice_argmax"] = [classes[i] for i in pred_idx]

    return out, metrics


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_path", required=True, help="Latest frozen lap_level_with_model1_model2_hazard_*.csv")
    ap.add_argument("--outputs_dir", default="outputs")
    ap.add_argument("--models_dir", default="models")
    ap.add_argument("--horizons", default="1,3,5")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--test_frac", type=float, default=0.20)
    ap.add_argument("--val_frac", type=float, default=0.20)
    ap.add_argument("--optuna_trials", type=int, default=40)
    ap.add_argument("--allowed_compounds", default="SOFT,MEDIUM,HARD,INTERMEDIATE,WET")
    ap.add_argument("--merge_output_name", default="lap_level_with_model1_model2_model3.csv")
    args = ap.parse_args()

    ensure_dir(args.outputs_dir)
    ensure_dir(args.models_dir)

    horizons = parse_horizons(args.horizons)
    allowed_compounds = parse_csv_list(args.allowed_compounds)

    ts = now_ts()

    df = pd.read_csv(args.input_path, low_memory=False)
    if df.columns.duplicated().any():
        dup_names = df.columns[df.columns.duplicated()].tolist()
        print(f"Warning: dropping duplicate input columns: {dup_names}")
        df = df.loc[:, ~df.columns.duplicated()].copy()

    pk = ["race_id", "driver_id", "lap_number"]
    if df.duplicated(pk).any():
        raise ValueError("Primary key duplicates found in input dataset")

    # Sort and feature engineering
    df = df.sort_values(pk).reset_index(drop=True)
    df = add_model3_features(df)

    # Add pit timing targets
    labeled = df.copy()
    for h in horizons:
        labeled = add_pit_horizon_labels(labeled, horizon=h)

    # Group split by race
    splits = make_group_split(
        race_ids=labeled["race_id"].astype(str).to_numpy(),
        seed=args.seed,
        test_frac=args.test_frac,
        val_frac=args.val_frac,
    )

    splits_path = os.path.join(args.outputs_dir, f"model3_race_splits_{ts}.json")
    with open(splits_path, "w") as f:
        json.dump(
            {
                "seed": args.seed,
                "train_races": splits.train_races,
                "val_races": splits.val_races,
                "test_races": splits.test_races,
            },
            f,
            indent=2,
        )

    merged_preds = labeled[["race_id", "driver_id", "lap_number"]].copy()
    all_metrics: List[Dict] = []

    # Model 3A pit timing
    for h in horizons:
        probs_h, metrics_h = train_pit_timing_model_for_horizon(
            df=labeled,
            horizon=h,
            splits=splits,
            outputs_dir=args.outputs_dir,
            models_dir=args.models_dir,
            seed=args.seed,
            optuna_trials=args.optuna_trials,
            ts=ts,
        )
        merged_preds = merged_preds.merge(probs_h, on=["race_id", "driver_id", "lap_number"], how="left")
        all_metrics.append(metrics_h)

    # Model 3B compound choice conditional on pit
    compound_metrics = None
    try:
        comp_probs, compound_metrics = train_compound_choice_model(
            df=labeled,
            splits=splits,
            outputs_dir=args.outputs_dir,
            models_dir=args.models_dir,
            seed=args.seed,
            optuna_trials=args.optuna_trials,
            ts=ts,
            allowed_compounds=allowed_compounds,
        )
        # Drop any old compound columns from merged_preds if present
        old_comp_cols = [c for c in merged_preds.columns if c.startswith("p_compound_") or c == "compound_choice_argmax"]
        if old_comp_cols:
            merged_preds = merged_preds.drop(columns=old_comp_cols)
        merged_preds = merged_preds.merge(comp_probs, on=["race_id", "driver_id", "lap_number"], how="left")
        all_metrics.append(compound_metrics)
    except Exception as e:
        note_path = os.path.join(args.outputs_dir, f"model3b_skipped_{ts}.txt")
        with open(note_path, "w") as f:
            f.write(str(e))
        print(f"Compound model skipped. Reason saved to: {note_path}")

    # Merge back into full master
    out = labeled.copy()

    # remove old model3 columns if rerunning on already model3 enriched dataset
    cols_to_drop = [c for c in out.columns if c.startswith("p_pit_next") or c.startswith("p_compound_") or c == "compound_choice_argmax"]
    if cols_to_drop:
        out = out.drop(columns=cols_to_drop)

    out = out.merge(merged_preds, on=["race_id", "driver_id", "lap_number"], how="left")

    # Save merged output
    out_path = os.path.join(args.outputs_dir, args.merge_output_name)
    out.to_csv(out_path, index=False)

    # Save all metrics summary
    metrics_path = os.path.join(args.outputs_dir, f"model3_metrics_all_{ts}.json")
    with open(metrics_path, "w") as f:
        json.dump(all_metrics, f, indent=2)

    print("DONE MODEL 3")
    print(f"Input: {args.input_path}")
    print(f"Splits: {splits_path}")
    print(f"Merged output: {out_path}")
    print(f"Metrics: {metrics_path}")
    if compound_metrics is None:
        print("Compound choice model was skipped. Check the model3b_skipped file.")


if __name__ == "__main__":
    main()