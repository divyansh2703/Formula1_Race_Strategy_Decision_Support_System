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
from sklearn.linear_model import HuberRegressor, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

try:
    from xgboost import XGBRegressor
except Exception:
    XGBRegressor = None

try:
    import optuna
except Exception:
    optuna = None


@dataclass
class SplitIds:
    train_races: List[str]
    val_races: List[str]
    test_races: List[str]


def now_ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


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


def build_preprocessor(cat_cols: List[str], num_cols: List[str], scale_numeric: bool) -> ColumnTransformer:
    cat_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("ohe", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    if scale_numeric:
        num_pipe = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )
    else:
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


def score_regression(y_true: np.ndarray, y_hat: np.ndarray) -> Dict:
    mae = float(mean_absolute_error(y_true, y_hat))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_hat)))
    try:
        r2 = float(r2_score(y_true, y_hat))
    except Exception:
        r2 = float("nan")

    return {
        "n": int(len(y_true)),
        "target_mean": float(np.mean(y_true)) if len(y_true) else float("nan"),
        "target_std": float(np.std(y_true)) if len(y_true) else float("nan"),
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
    }


def residual_summary(y_true: np.ndarray, y_hat: np.ndarray) -> Dict:
    err = y_hat - y_true
    ae = np.abs(err)
    return {
        "mean_error": float(np.mean(err)),
        "median_error": float(np.median(err)),
        "p10_error": float(np.quantile(err, 0.10)),
        "p90_error": float(np.quantile(err, 0.90)),
        "mean_abs_error": float(np.mean(ae)),
        "median_abs_error": float(np.median(ae)),
        "p90_abs_error": float(np.quantile(ae, 0.90)),
    }


def score_by_band(y_true: np.ndarray, y_hat: np.ndarray, band_edges: List[float]) -> List[Dict]:
    rows = []
    y_true = np.asarray(y_true)
    y_hat = np.asarray(y_hat)

    all_edges = [-np.inf] + list(band_edges) + [np.inf]
    for lo, hi in zip(all_edges[:-1], all_edges[1:]):
        mask = (y_true > lo) & (y_true <= hi)
        n = int(mask.sum())
        label = f"({lo}, {hi}]"
        if n == 0:
            rows.append({"band": label, "n": 0, "mae": float("nan"), "rmse": float("nan")})
            continue
        yt = y_true[mask]
        yp = y_hat[mask]
        rows.append(
            {
                "band": label,
                "n": n,
                "mae": float(mean_absolute_error(yt, yp)),
                "rmse": float(np.sqrt(mean_squared_error(yt, yp))),
            }
        )
    return rows


def score_by_column_groups(
    df_eval: pd.DataFrame,
    y_true: np.ndarray,
    y_hat: np.ndarray,
    group_cols: List[str],
    top_n: int = 20,
) -> Dict[str, List[Dict]]:
    out: Dict[str, List[Dict]] = {}
    eval_df = df_eval.copy()
    eval_df["_y_true"] = y_true
    eval_df["_y_hat"] = y_hat
    eval_df["_ae"] = np.abs(eval_df["_y_hat"] - eval_df["_y_true"])
    eval_df["_se"] = (eval_df["_y_hat"] - eval_df["_y_true"]) ** 2

    for col in group_cols:
        if col not in eval_df.columns:
            continue
        grp = (
            eval_df.groupby(col, dropna=False)
            .agg(
                n=("_y_true", "size"),
                target_mean=("_y_true", "mean"),
                pred_mean=("_y_hat", "mean"),
                mae=("_ae", "mean"),
                rmse=("_se", lambda s: float(np.sqrt(np.mean(s)))),
            )
            .reset_index()
            .sort_values(["n", "mae"], ascending=[False, True])
            .head(top_n)
        )
        out[col] = grp.to_dict(orient="records")
    return out


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


def get_feature_cols(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    cat_cols = [
        "season",
        "round",
        "circuit_id",
        "team_id",
        "driver_id",
        "race_phase",
        "race_level_weather_condition",
        "tyre_compound",
        "tyre_compound_after",
    ]

    num_cols = [
        "lap_number",
        "laps_remaining",
        "position",
        "field_size",
        "gap_ahead",
        "gap_behind",
        "clean_air_flag",
        "cars_ahead_count",
        "cars_behind_count",
        "avg_tyre_age_ahead",
        "avg_tyre_age_behind",
        "local_traffic_count",
        "local_pack_density",
        "surrounded_pressure",
        "small_gap_ahead_flag",
        "small_gap_behind_flag",
        "tyre_age",
        "stint_number",
        "pit_stop_number",
        "pit_to_soft_flag",
        "pit_to_medium_flag",
        "pit_to_hard_flag",
        "pit_to_intermediate_flag",
        "pit_to_wet_flag",
        "pit_lane_time_loss",
        "pit_lane_time_sec",
        "stationary_time_sec",
        "track_length",
        "overtaking_difficulty_index",
        "track_temperature_used",
        "air_temperature_used",
        "rain_flag_used",
        "lap_time",
        "previous_lap_time",
        "expected_lap_time",
        "expected_lap_time_fresh",
        "tyre_degradation_delta",
        "tyre_degradation_per_lap",
        "lap_time_sigma",
        "lap_time_variance",
        "fresh_tyre_gain_proxy",
        "pace_loss_vs_expected",
        "prev_pace_loss_vs_expected",
        "deg_pressure",
        "deg_delta_pressure",
        "rejoin_attack_potential",
        "pitloss_minus_gapbehind",
        "pitloss_minus_gapahead",
        "safe_rejoin_ahead_of_current_behind_flag",
        "traffic_window_risk",
        "race_progress_frac",
        "likely_final_stint_if_pit_now_flag",
        "late_race_track_position_priority",
        "tyre_age_vs_ahead",
        "tyre_age_vs_behind",
        "p_sc_next3",
        "p_sc_next5",
        "p_sc_next10",
        "p_vsc_next3",
        "p_vsc_next5",
        "p_vsc_next10",
        "p_rf_next3",
        "p_rf_next5",
        "p_rf_next10",
        "p_pit_next1",
        "p_pit_next3",
        "p_pit_next5",
        "p_compound_SOFT",
        "p_compound_MEDIUM",
        "p_compound_HARD",
        "p_compound_INTERMEDIATE",
        "p_compound_WET",
        "racelap_p_pit_next1_mean",
        "racelap_p_pit_next1_max",
        "racelap_p_pit_next3_mean",
        "racelap_p_pit_next3_max",
        "racelap_p_pit_next5_mean",
        "racelap_p_pit_next5_max",
        "racelap_high_pitnext1_count",
        "racelap_high_pitnext3_count",
        "y_rejoin_clean_air_nextlap",
        "p_rejoin_clean_air_nextlap",
        "y_rejoin_position_delta_nextlap",
        "pred_rejoin_position_delta_nextlap",
    ]

    cat_cols = [c for c in cat_cols if c in df.columns]
    num_cols = [c for c in num_cols if c in df.columns]
    return cat_cols, num_cols


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    def ensure_num(col: str, fill: float = 0.0) -> np.ndarray:
        if col in out.columns:
            return pd.to_numeric(out[col], errors="coerce").fillna(fill).to_numpy(dtype=float)
        return np.full(len(out), fill, dtype=float)

    pit_lane_time_loss = ensure_num("pit_lane_time_loss")
    overtaking_difficulty_index = ensure_num("overtaking_difficulty_index")
    p_rejoin_clean_air_nextlap = ensure_num("p_rejoin_clean_air_nextlap")
    pred_rejoin_position_delta_nextlap = ensure_num("pred_rejoin_position_delta_nextlap")
    track_temperature_used = ensure_num("track_temperature_used")
    rain_flag_used = ensure_num("rain_flag_used")
    tyre_age = ensure_num("tyre_age")
    pit_stop_number = ensure_num("pit_stop_number")
    pit_to_soft_flag = ensure_num("pit_to_soft_flag")
    pit_to_hard_flag = ensure_num("pit_to_hard_flag")
    pit_to_medium_flag = ensure_num("pit_to_medium_flag")
    stationary_time_sec = ensure_num("stationary_time_sec")
    pit_lane_time_sec = ensure_num("pit_lane_time_sec")
    gap_ahead = ensure_num("gap_ahead")
    gap_behind = ensure_num("gap_behind")
    clean_air_flag = ensure_num("clean_air_flag")
    local_pack_density = ensure_num("local_pack_density")
    traffic_window_risk = ensure_num("traffic_window_risk")

    out["is_out_lap_nextlap_proxy"] = 1.0
    out["pit_lane_time_loss_x_overtaking_difficulty"] = pit_lane_time_loss * overtaking_difficulty_index
    out["p_rejoin_clean_air_x_rejoin_delta"] = p_rejoin_clean_air_nextlap * pred_rejoin_position_delta_nextlap
    out["track_temp_x_pit_to_soft"] = track_temperature_used * pit_to_soft_flag
    out["track_temp_x_pit_to_hard"] = track_temperature_used * pit_to_hard_flag
    out["track_temp_x_pit_to_medium"] = track_temperature_used * pit_to_medium_flag
    out["rain_x_rejoin_delta"] = rain_flag_used * pred_rejoin_position_delta_nextlap
    out["tyre_age_x_pit_stop_number"] = tyre_age * pit_stop_number
    out["stationary_plus_pitlane_sec"] = stationary_time_sec + pit_lane_time_sec
    out["gap_balance"] = gap_ahead - gap_behind
    out["rejoin_traffic_pressure_combo"] = (
        (1.0 - np.clip(p_rejoin_clean_air_nextlap, 0.0, 1.0))
        * np.maximum(pred_rejoin_position_delta_nextlap, 0.0)
        * (1.0 + local_pack_density)
    )
    out["pitloss_vs_gapbehind_interaction"] = pit_lane_time_loss * np.maximum(0.0, 1.5 - gap_behind)
    out["pitloss_vs_gapahead_interaction"] = pit_lane_time_loss * np.maximum(0.0, 1.5 - gap_ahead)
    out["clean_air_x_traffic_risk"] = clean_air_flag * traffic_window_risk

    return out


def filter_rows(df: pd.DataFrame, wet_mode: str) -> pd.DataFrame:
    out = df.copy()

    if wet_mode == "dry_only":
        if "rain_flag_used" in out.columns:
            out = out[pd.to_numeric(out["rain_flag_used"], errors="coerce").fillna(0) < 0.5].copy()
        if "tyre_compound_after" in out.columns:
            wet_like = {"INTERMEDIATE", "WET"}
            out = out[~out["tyre_compound_after"].astype(str).str.upper().isin(wet_like)].copy()

    return out


def build_target(
    df: pd.DataFrame,
    target_col: str,
    clip_mode: str,
    clip_upper_mode: str,
    clip_upper_value: float,
    clip_quantile: float,
    clip_lower: float,
    log_target: bool,
    train_mask: np.ndarray,
) -> Tuple[np.ndarray, Dict]:
    y_raw = pd.to_numeric(df[target_col], errors="coerce").astype(float).to_numpy()
    y_train_raw = y_raw[train_mask]

    meta: Dict = {
        "target_col": target_col,
        "clip_mode": clip_mode,
        "clip_upper_mode": clip_upper_mode,
        "clip_upper_value": clip_upper_value,
        "clip_quantile": clip_quantile,
        "clip_lower": clip_lower,
        "log_target": log_target,
    }

    if clip_mode == "none":
        y_model = y_raw.copy()
        meta["applied_clip_upper"] = None
    else:
        if clip_upper_mode == "fixed":
            upper = float(clip_upper_value)
        elif clip_upper_mode == "train_quantile":
            upper = float(np.quantile(y_train_raw, clip_quantile))
        else:
            raise ValueError(f"Unsupported clip_upper_mode: {clip_upper_mode}")

        y_model = np.clip(y_raw, clip_lower, upper)
        meta["applied_clip_upper"] = upper

    if log_target:
        if np.nanmin(y_model) < 0:
            raise ValueError("log_target requires non negative modeled target after clipping")
        y_model = np.log1p(y_model)
        meta["transform"] = "log1p"
    else:
        meta["transform"] = "identity"

    return y_model, meta


def inverse_target_transform(y_pred_model_space: np.ndarray, target_meta: Dict) -> np.ndarray:
    y = np.asarray(y_pred_model_space, dtype=float)

    if target_meta.get("log_target", False):
        y = np.expm1(y)

    lower = float(target_meta.get("clip_lower", 0.0))
    applied_upper = target_meta.get("applied_clip_upper", None)
    y = np.maximum(y, lower)

    if applied_upper is not None:
        y = np.minimum(y, float(applied_upper))

    return y


def tune_xgb_with_optuna(X_train, y_train, X_val, y_val, seed: int, n_trials: int) -> Dict:
    if XGBRegressor is None:
        raise RuntimeError("xgboost is not installed")
    if optuna is None:
        raise RuntimeError("optuna is not installed")

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 150, 650),
            "max_depth": trial.suggest_int("max_depth", 2, 6),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.08, log=True),
            "subsample": trial.suggest_float("subsample", 0.75, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.60, 0.98),
            "min_child_weight": trial.suggest_float("min_child_weight", 2.0, 25.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 1.0, 80.0, log=True),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 15.0, log=True),
            "gamma": trial.suggest_float("gamma", 0.0, 10.0),
        }

        model = XGBRegressor(
            objective="reg:squarederror",
            eval_metric="rmse",
            tree_method="hist",
            random_state=seed,
            n_jobs=6,
            **params,
        )
        fit_xgb_with_early_stopping(model, X_train, y_train, X_val, y_val, rounds=35)

        pred_train = model.predict(X_train)
        pred_val = model.predict(X_val)

        rmse_val = float(np.sqrt(mean_squared_error(y_val, pred_val)))
        rmse_train = float(np.sqrt(mean_squared_error(y_train, pred_train)))
        generalization_gap = max(0.0, rmse_val - rmse_train)
        score = -rmse_val - 0.20 * generalization_gap
        return float(score)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best = dict(study.best_params)
    best["best_objective"] = float(study.best_value)
    return best


def fit_ridge_model(
    X_train_raw,
    y_train_model,
    X_val_raw,
    y_val_model,
    X_test_raw,
    y_test_model,
    y_train_raw_eval,
    y_val_raw_eval,
    y_test_raw_eval,
    cat_cols,
    num_cols,
    target_meta,
) -> Dict:
    pre = build_preprocessor(cat_cols, num_cols, scale_numeric=True)
    pipe = Pipeline(
        steps=[
            ("preprocessor", pre),
            ("model", Ridge(alpha=1.0)),
        ]
    )
    pipe.fit(X_train_raw, y_train_model)

    pred_train_model = pipe.predict(X_train_raw)
    pred_val_model = pipe.predict(X_val_raw)
    pred_test_model = pipe.predict(X_test_raw)

    pred_train = inverse_target_transform(pred_train_model, target_meta)
    pred_val = inverse_target_transform(pred_val_model, target_meta)
    pred_test = inverse_target_transform(pred_test_model, target_meta)

    return {
        "pipeline": pipe,
        "train": score_regression(y_train_raw_eval, pred_train),
        "val": score_regression(y_val_raw_eval, pred_val),
        "test": score_regression(y_test_raw_eval, pred_test),
        "pred_train": pred_train,
        "pred_val": pred_val,
        "pred_test": pred_test,
    }


def fit_huber_model(
    X_train_raw,
    y_train_model,
    X_val_raw,
    y_val_model,
    X_test_raw,
    y_test_model,
    y_train_raw_eval,
    y_val_raw_eval,
    y_test_raw_eval,
    cat_cols,
    num_cols,
    target_meta,
) -> Dict:
    pre = build_preprocessor(cat_cols, num_cols, scale_numeric=True)
    pipe = Pipeline(
        steps=[
            ("preprocessor", pre),
            ("model", HuberRegressor(epsilon=1.35, alpha=1e-4, max_iter=300)),
        ]
    )
    pipe.fit(X_train_raw, y_train_model)

    pred_train_model = pipe.predict(X_train_raw)
    pred_val_model = pipe.predict(X_val_raw)
    pred_test_model = pipe.predict(X_test_raw)

    pred_train = inverse_target_transform(pred_train_model, target_meta)
    pred_val = inverse_target_transform(pred_val_model, target_meta)
    pred_test = inverse_target_transform(pred_test_model, target_meta)

    return {
        "pipeline": pipe,
        "train": score_regression(y_train_raw_eval, pred_train),
        "val": score_regression(y_val_raw_eval, pred_val),
        "test": score_regression(y_test_raw_eval, pred_test),
        "pred_train": pred_train,
        "pred_val": pred_val,
        "pred_test": pred_test,
    }


def fit_xgb_model(
    X_train_raw,
    y_train_model,
    X_val_raw,
    y_val_model,
    X_test_raw,
    y_test_model,
    y_train_raw_eval,
    y_val_raw_eval,
    y_test_raw_eval,
    cat_cols,
    num_cols,
    seed,
    optuna_trials,
    target_meta,
) -> Dict:
    if XGBRegressor is None:
        raise RuntimeError("xgboost is not installed")

    pre = build_preprocessor(cat_cols, num_cols, scale_numeric=False)
    X_train = pre.fit_transform(X_train_raw)
    X_val = pre.transform(X_val_raw)
    X_test = pre.transform(X_test_raw)

    if optuna_trials > 0 and optuna is not None:
        best = tune_xgb_with_optuna(X_train, y_train_model, X_val, y_val_model, seed=seed, n_trials=optuna_trials)
    else:
        best = {
            "n_estimators": 320,
            "max_depth": 3,
            "learning_rate": 0.03,
            "subsample": 0.85,
            "colsample_bytree": 0.80,
            "min_child_weight": 8.0,
            "reg_lambda": 10.0,
            "reg_alpha": 0.5,
            "gamma": 1.0,
            "best_objective": float("nan"),
        }

    model = XGBRegressor(
        objective="reg:squarederror",
        eval_metric="rmse",
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
    fit_xgb_with_early_stopping(model, X_train, y_train_model, X_val, y_val_model, rounds=35)

    pred_train_model = model.predict(X_train)
    pred_val_model = model.predict(X_val)
    pred_test_model = model.predict(X_test)

    pred_train = inverse_target_transform(pred_train_model, target_meta)
    pred_val = inverse_target_transform(pred_val_model, target_meta)
    pred_test = inverse_target_transform(pred_test_model, target_meta)

    return {
        "preprocessor": pre,
        "model": model,
        "best_params": best,
        "train": score_regression(y_train_raw_eval, pred_train),
        "val": score_regression(y_val_raw_eval, pred_val),
        "test": score_regression(y_test_raw_eval, pred_test),
        "pred_train": pred_train,
        "pred_val": pred_val,
        "pred_test": pred_test,
    }


def choose_model(ridge_res: Dict, huber_res: Dict, xgb_res: Dict) -> str:
    candidates = {
        "ridge": ridge_res,
        "huber": huber_res,
        "xgb": xgb_res,
    }

    scores = {}
    for name, res in candidates.items():
        val_rmse = res["val"]["rmse"]
        val_mae = res["val"]["mae"]
        gap = max(0.0, res["val"]["rmse"] - res["train"]["rmse"])

        if name == "xgb":
            score = -val_rmse - 0.20 * gap - 0.10 * val_mae
        else:
            score = -val_rmse - 0.10 * gap - 0.10 * val_mae
        scores[name] = score

    winner = max(scores, key=scores.get)
    return winner


def predict_all_ridge(pipe: Pipeline, X_all_raw: pd.DataFrame, target_meta: Dict) -> np.ndarray:
    pred_model = pipe.predict(X_all_raw)
    return inverse_target_transform(pred_model, target_meta)


def predict_all_xgb(pre, model, X_all_raw: pd.DataFrame, target_meta: Dict) -> np.ndarray:
    X_all = pre.transform(X_all_raw)
    pred_model = model.predict(X_all)
    return inverse_target_transform(pred_model, target_meta)


def save_feature_importance_xgb(preprocessor, model, out_path: str) -> None:
    try:
        feature_names = preprocessor.get_feature_names_out()
        gains = model.feature_importances_
        imp = pd.DataFrame({"feature": feature_names, "importance": gains})
        imp = imp.sort_values("importance", ascending=False)
        imp.to_csv(out_path, index=False)
    except Exception:
        pass


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_path", required=True, help="model4b_position_delta_scored.csv or enriched table with 4A and 4B outputs")
    ap.add_argument("--outputs_dir", default="outputs")
    ap.add_argument("--models_dir", default="models")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--test_frac", type=float, default=0.20)
    ap.add_argument("--val_frac", type=float, default=0.20)
    ap.add_argument("--optuna_trials", type=int, default=60)
    ap.add_argument("--scored_output_name", default="model4c_penalty_scored.csv")

    ap.add_argument("--clip_mode", choices=["none", "clip"], default="clip")
    ap.add_argument("--clip_upper_mode", choices=["fixed", "train_quantile"], default="train_quantile")
    ap.add_argument("--clip_upper_value", type=float, default=55.0)
    ap.add_argument("--clip_quantile", type=float, default=0.99)
    ap.add_argument("--clip_lower", type=float, default=0.0)
    ap.add_argument("--log_target", type=int, default=0)

    ap.add_argument("--wet_mode", choices=["all", "dry_only"], default="all")
    ap.add_argument("--min_target", type=float, default=0.0)
    ap.add_argument("--max_target", type=float, default=120.0)
    args = ap.parse_args()

    ensure_dir(args.outputs_dir)
    ensure_dir(args.models_dir)

    ts = now_ts()
    df = pd.read_csv(args.input_path, low_memory=False)

    required_cols = ["race_id", "driver_id", "lap_number", "y_rejoin_penalty_sec_nextlap"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df["y_rejoin_penalty_sec_nextlap"] = pd.to_numeric(df["y_rejoin_penalty_sec_nextlap"], errors="coerce")
    df = df[df["y_rejoin_penalty_sec_nextlap"].notna()].copy()
    df = df[(df["y_rejoin_penalty_sec_nextlap"] >= args.min_target) & (df["y_rejoin_penalty_sec_nextlap"] <= args.max_target)].copy()
    df = filter_rows(df, wet_mode=args.wet_mode)
    df = add_engineered_features(df)

    if len(df) == 0:
        raise ValueError("No rows left after filtering")

    splits = make_group_split(
        race_ids=df["race_id"].astype(str).to_numpy(),
        seed=args.seed,
        test_frac=args.test_frac,
        val_frac=args.val_frac,
    )

    splits_path = os.path.join(args.outputs_dir, f"model4c_penalty_race_splits_{ts}.json")
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

    cat_cols, num_cols = get_feature_cols(df)

    engineered_num_cols = [
        "is_out_lap_nextlap_proxy",
        "pit_lane_time_loss_x_overtaking_difficulty",
        "p_rejoin_clean_air_x_rejoin_delta",
        "track_temp_x_pit_to_soft",
        "track_temp_x_pit_to_hard",
        "track_temp_x_pit_to_medium",
        "rain_x_rejoin_delta",
        "tyre_age_x_pit_stop_number",
        "stationary_plus_pitlane_sec",
        "gap_balance",
        "rejoin_traffic_pressure_combo",
        "pitloss_vs_gapbehind_interaction",
        "pitloss_vs_gapahead_interaction",
        "clean_air_x_traffic_risk",
    ]
    engineered_num_cols = [c for c in engineered_num_cols if c in df.columns]

    num_cols = list(dict.fromkeys(num_cols + engineered_num_cols))
    feature_cols = cat_cols + num_cols

    X_all = df[feature_cols].copy()
    for c in cat_cols:
        X_all[c] = X_all[c].astype(str)
    for c in num_cols:
        X_all[c] = pd.to_numeric(X_all[c], errors="coerce")

    tr = np.isin(df["race_id"].astype(str).to_numpy(), np.array(splits.train_races, dtype=str))
    va = np.isin(df["race_id"].astype(str).to_numpy(), np.array(splits.val_races, dtype=str))
    te = np.isin(df["race_id"].astype(str).to_numpy(), np.array(splits.test_races, dtype=str))

    y_raw = df["y_rejoin_penalty_sec_nextlap"].astype(float).to_numpy()
    y_model, target_meta = build_target(
        df=df,
        target_col="y_rejoin_penalty_sec_nextlap",
        clip_mode=args.clip_mode,
        clip_upper_mode=args.clip_upper_mode,
        clip_upper_value=args.clip_upper_value,
        clip_quantile=args.clip_quantile,
        clip_lower=args.clip_lower,
        log_target=bool(args.log_target),
        train_mask=tr,
    )

    X_train = X_all.loc[tr].copy()
    X_val = X_all.loc[va].copy()
    X_test = X_all.loc[te].copy()

    y_train_model = y_model[tr]
    y_val_model = y_model[va]
    y_test_model = y_model[te]

    y_train_raw = y_raw[tr]
    y_val_raw = y_raw[va]
    y_test_raw = y_raw[te]

    ridge_res = fit_ridge_model(
        X_train_raw=X_train,
        y_train_model=y_train_model,
        X_val_raw=X_val,
        y_val_model=y_val_model,
        X_test_raw=X_test,
        y_test_model=y_test_model,
        y_train_raw_eval=y_train_raw,
        y_val_raw_eval=y_val_raw,
        y_test_raw_eval=y_test_raw,
        cat_cols=cat_cols,
        num_cols=num_cols,
        target_meta=target_meta,
    )

    huber_res = fit_huber_model(
        X_train_raw=X_train,
        y_train_model=y_train_model,
        X_val_raw=X_val,
        y_val_model=y_val_model,
        X_test_raw=X_test,
        y_test_model=y_test_model,
        y_train_raw_eval=y_train_raw,
        y_val_raw_eval=y_val_raw,
        y_test_raw_eval=y_test_raw,
        cat_cols=cat_cols,
        num_cols=num_cols,
        target_meta=target_meta,
    )

    xgb_res = fit_xgb_model(
        X_train_raw=X_train,
        y_train_model=y_train_model,
        X_val_raw=X_val,
        y_val_model=y_val_model,
        X_test_raw=X_test,
        y_test_model=y_test_model,
        y_train_raw_eval=y_train_raw,
        y_val_raw_eval=y_val_raw,
        y_test_raw_eval=y_test_raw,
        cat_cols=cat_cols,
        num_cols=num_cols,
        seed=args.seed,
        optuna_trials=args.optuna_trials,
        target_meta=target_meta,
    )

    winner = choose_model(ridge_res, huber_res, xgb_res)

    if winner == "xgb":
        pred_all = predict_all_xgb(
            pre=xgb_res["preprocessor"],
            model=xgb_res["model"],
            X_all_raw=X_all,
            target_meta=target_meta,
        )
        pred_test = xgb_res["pred_test"]
    elif winner == "huber":
        pred_all = predict_all_ridge(
            pipe=huber_res["pipeline"],
            X_all_raw=X_all,
            target_meta=target_meta,
        )
        pred_test = huber_res["pred_test"]
    else:
        pred_all = predict_all_ridge(
            pipe=ridge_res["pipeline"],
            X_all_raw=X_all,
            target_meta=target_meta,
        )
        pred_test = ridge_res["pred_test"]

    pred_all = np.clip(pred_all, args.min_target, args.max_target)
    pred_test = np.clip(pred_test, args.min_target, args.max_target)

    base = f"model4c_penalty_{ts}"

    if winner == "xgb":
        joblib.dump(xgb_res["preprocessor"], os.path.join(args.models_dir, f"{base}_xgb_preprocessor.pkl"))
        joblib.dump(xgb_res["model"], os.path.join(args.models_dir, f"{base}_xgb.pkl"))
        save_feature_importance_xgb(
            xgb_res["preprocessor"],
            xgb_res["model"],
            os.path.join(args.outputs_dir, f"{base}_xgb_feature_importance.csv"),
        )
    elif winner == "huber":
        joblib.dump(huber_res["pipeline"], os.path.join(args.models_dir, f"{base}_huber.pkl"))
    else:
        joblib.dump(ridge_res["pipeline"], os.path.join(args.models_dir, f"{base}_ridge.pkl"))

    test_eval_df = df.loc[te, :].copy()

    metrics = {
        "target": "y_rejoin_penalty_sec_nextlap",
        "mode": "regression_rejoin_penalty",
        "winner": winner,
        "cat_cols": cat_cols,
        "num_cols": num_cols,
        "feature_cols": feature_cols,
        "target_meta": target_meta,
        "filters": {
            "wet_mode": args.wet_mode,
            "min_target": args.min_target,
            "max_target": args.max_target,
        },
        "ridge": {
            "train": ridge_res["train"],
            "val": ridge_res["val"],
            "test": ridge_res["test"],
            "test_residual_summary": residual_summary(y_test_raw, ridge_res["pred_test"]),
            "test_band_metrics": score_by_band(y_test_raw, ridge_res["pred_test"], band_edges=[3, 8, 15, 25, 40]),
        },
        "huber": {
            "train": huber_res["train"],
            "val": huber_res["val"],
            "test": huber_res["test"],
            "test_residual_summary": residual_summary(y_test_raw, huber_res["pred_test"]),
            "test_band_metrics": score_by_band(y_test_raw, huber_res["pred_test"], band_edges=[3, 8, 15, 25, 40]),
        },
        "xgb": {
            "train": xgb_res["train"],
            "val": xgb_res["val"],
            "test": xgb_res["test"],
            "best_params": xgb_res["best_params"],
            "test_residual_summary": residual_summary(y_test_raw, xgb_res["pred_test"]),
            "test_band_metrics": score_by_band(y_test_raw, xgb_res["pred_test"], band_edges=[3, 8, 15, 25, 40]),
        },
        "winner_residual_summary_test": residual_summary(y_test_raw, pred_test),
        "winner_test_band_metrics": score_by_band(y_test_raw, pred_test, band_edges=[3, 8, 15, 25, 40]),
        "winner_group_metrics_test": score_by_column_groups(
            test_eval_df,
            y_true=y_test_raw,
            y_hat=pred_test,
            group_cols=[
                "circuit_id",
                "race_level_weather_condition",
                "tyre_compound_after",
                "team_id",
                "driver_id",
            ],
            top_n=25,
        ),
    }

    metrics_path = os.path.join(args.outputs_dir, f"{base}_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    config_path = os.path.join(args.outputs_dir, f"{base}_config.json")
    with open(config_path, "w") as f:
        json.dump(
            {
                "input_path": args.input_path,
                "seed": args.seed,
                "test_frac": args.test_frac,
                "val_frac": args.val_frac,
                "optuna_trials": args.optuna_trials,
                "scored_output_name": args.scored_output_name,
                "clip_mode": args.clip_mode,
                "clip_upper_mode": args.clip_upper_mode,
                "clip_upper_value": args.clip_upper_value,
                "clip_quantile": args.clip_quantile,
                "clip_lower": args.clip_lower,
                "log_target": int(args.log_target),
                "wet_mode": args.wet_mode,
                "min_target": args.min_target,
                "max_target": args.max_target,
            },
            f,
            indent=2,
        )

    scored = df.copy()
    scored["pred_rejoin_penalty_sec_nextlap"] = pred_all.astype(float)

    scored_path = os.path.join(args.outputs_dir, args.scored_output_name)
    scored.to_csv(scored_path, index=False)

    split_pred = df[["race_id", "driver_id", "lap_number", "y_rejoin_penalty_sec_nextlap"]].copy()
    split_pred["split"] = np.where(tr, "train", np.where(va, "val", np.where(te, "test", "other")))
    split_pred["pred_rejoin_penalty_sec_nextlap"] = pred_all.astype(float)
    split_pred_path = os.path.join(args.outputs_dir, f"{base}_all_predictions.csv")
    split_pred.to_csv(split_pred_path, index=False)

    print("DONE MODEL 4C penalty regressor")
    print(f"Winner: {winner}")
    print(f"Splits: {splits_path}")
    print(f"Metrics: {metrics_path}")
    print(f"Config: {config_path}")
    print(f"Scored output: {scored_path}")
    print(f"All predictions: {split_pred_path}")


if __name__ == "__main__":
    main()