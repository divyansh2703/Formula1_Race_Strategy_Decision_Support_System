from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

try:
    import optuna
except Exception:
    optuna = None

try:
    import xgboost as xgb
except Exception:
    xgb = None


MODEL1_FEATURES: List[str] = [
    "tyre_compound",
    "tyre_age",
    "lap_number",
    "stint_number",
    "driver_id",
    "team_id",
    "circuit_id",
    "season",
    "track_temperature_used",
    "air_temperature_used",
    "rain_flag_used",
    "gap_ahead",
    "gap_behind",
    "position",
    "speed_trap",
    "previous_lap_time",
    "clean_air_flag",
    "laps_remaining",
    "race_phase",
    "overtaking_difficulty_index",
    "pit_lane_time_loss",
]

CATEGORICAL_COLS: List[str] = [
    "tyre_compound",
    "driver_id",
    "team_id",
    "circuit_id",
    "race_phase",
]

TARGET_COL = "lap_time"
KEY_COLS = ["race_id", "driver_id", "lap_number"]


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def now_tag() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _coerce_bool01(series: pd.Series) -> pd.Series:
    if series is None:
        return series
    if series.dtype == bool:
        return series.astype(int)
    if pd.api.types.is_numeric_dtype(series):
        return series.fillna(0).astype(int).clip(0, 1)
    s = series.astype(str).str.upper()
    out = np.where(s.isin(["TRUE", "1"]), 1, np.where(s.isin(["FALSE", "0"]), 0, np.nan))
    return pd.Series(out, index=series.index).astype("float").fillna(0).astype(int)


def load_and_filter(input_path: str, min_lap_time: float = 40.0, max_lap_time: float = 200.0) -> pd.DataFrame:
    df = pd.read_csv(input_path)

    required_cols = set(KEY_COLS + [TARGET_COL, "exclude_from_laptime_model_flag"] + MODEL1_FEATURES)
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df[df["exclude_from_laptime_model_flag"] == False].copy()
    df = df.dropna(subset=[TARGET_COL]).copy()
    df = df[(df[TARGET_COL] >= min_lap_time) & (df[TARGET_COL] <= max_lap_time)].copy()

    for c in [
        "clean_air_flag",
        "rain_flag_used",
        "is_under_safety_car",
        "is_under_virtual_safety_car",
        "is_out_lap",
        "is_in_lap",
        "is_pit_lap",
    ]:
        if c in df.columns:
            df[c] = _coerce_bool01(df[c])

    return df


def split_by_race_id(
    df: pd.DataFrame, test_size: float = 0.2, val_size_within_train: float = 0.2, seed: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    races = np.array(df["race_id"].dropna().unique())
    rng = np.random.default_rng(seed)
    rng.shuffle(races)

    n_total = len(races)
    n_test = int(round(n_total * test_size))
    test_races = set(races[:n_test])
    trainval_races = set(races[n_test:])

    df_test = df[df["race_id"].isin(test_races)].copy()
    df_trainval = df[df["race_id"].isin(trainval_races)].copy()

    races_tv = np.array(df_trainval["race_id"].dropna().unique())
    rng.shuffle(races_tv)
    n_val = int(round(len(races_tv) * val_size_within_train))
    val_races = set(races_tv[:n_val])

    df_val = df_trainval[df_trainval["race_id"].isin(val_races)].copy()
    df_train = df_trainval[~df_trainval["race_id"].isin(val_races)].copy()
    return df_train, df_val, df_test


def build_preprocessor() -> ColumnTransformer:
    numeric_cols = [c for c in MODEL1_FEATURES if c not in CATEGORICAL_COLS]
    cat_cols = [c for c in CATEGORICAL_COLS if c in MODEL1_FEATURES]

    num_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler(with_mean=False)),
        ]
    )
    cat_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("ohe", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[("num", num_pipe, numeric_cols), ("cat", cat_pipe, cat_cols)],
        remainder="drop",
    )


def _monotone_constraints_for_preprocessor(pre: ColumnTransformer) -> Tuple[int, ...]:
    numeric_cols = [c for c in MODEL1_FEATURES if c not in CATEGORICAL_COLS]
    dummy = pd.DataFrame([{c: 0 for c in MODEL1_FEATURES}])
    Xt = pre.transform(dummy)
    n_out = Xt.shape[1]

    cons = [0] * n_out
    if "tyre_age" in numeric_cols:
        cons[numeric_cols.index("tyre_age")] = 1
    return tuple(cons)


def train_ridge(df_train: pd.DataFrame, df_val: pd.DataFrame, models_dir: str) -> Dict:
    pre = build_preprocessor()
    X_train = df_train[MODEL1_FEATURES].copy()
    y_train = df_train[TARGET_COL].values.astype(float)
    X_val = df_val[MODEL1_FEATURES].copy()
    y_val = df_val[TARGET_COL].values.astype(float)

    pipe = Pipeline(steps=[("pre", pre), ("ridge", Ridge(alpha=1.0, random_state=0))])
    pipe.fit(X_train, y_train)

    pred_val = pipe.predict(X_val)
    mae = float(mean_absolute_error(y_val, pred_val))

    joblib.dump(pipe, os.path.join(models_dir, "model1_ridge.pkl"))
    return {"val_mae": mae}


def train_xgb_mean_with_optuna(
    df_train: pd.DataFrame, df_val: pd.DataFrame, models_dir: str, n_trials: int = 40, seed: int = 42
) -> Dict:
    if optuna is None:
        raise RuntimeError("optuna is not installed.")
    if xgb is None:
        raise RuntimeError("xgboost is not installed.")

    pre = build_preprocessor()
    X_train = df_train[MODEL1_FEATURES].copy()
    y_train = df_train[TARGET_COL].values.astype(float)
    X_val = df_val[MODEL1_FEATURES].copy()
    y_val = df_val[TARGET_COL].values.astype(float)

    X_train_t = pre.fit_transform(X_train)
    X_val_t = pre.transform(X_val)

    mono = _monotone_constraints_for_preprocessor(pre)
    if len(mono) != X_train_t.shape[1]:
        mono = tuple([0] * X_train_t.shape[1])

    def objective(trial: "optuna.Trial") -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 300, 1400),
            "max_depth": trial.suggest_int("max_depth", 3, 9),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 30.0, log=True),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 6.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.5, 12.0),
            "gamma": trial.suggest_float("gamma", 0.0, 6.0),
        }

        model = xgb.XGBRegressor(
            objective="reg:squarederror",
            random_state=seed,
            tree_method="hist",
            n_jobs=-1,
            monotone_constraints=mono,
            **params,
        )
        model.fit(X_train_t, y_train, eval_set=[(X_val_t, y_val)], verbose=False)
        pred = model.predict(X_val_t)
        return float(mean_absolute_error(y_val, pred))

    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials)

    best_params = dict(study.best_trial.params)
    best_model = xgb.XGBRegressor(
        objective="reg:squarederror",
        random_state=seed,
        tree_method="hist",
        n_jobs=-1,
        monotone_constraints=mono,
        **best_params,
    )
    best_model.fit(X_train_t, y_train, eval_set=[(X_val_t, y_val)], verbose=False)

    joblib.dump(pre, os.path.join(models_dir, "model1_preprocessor.pkl"))
    best_model.save_model(os.path.join(models_dir, "model1_xgb_mean.json"))
    return {"best_val_mae": float(study.best_value), "best_params": best_params, "monotone_tyre_age": True}


def _load_primary_predictor(models_dir: str, model_kind: str):
    if model_kind == "ridge":
        pipe = joblib.load(os.path.join(models_dir, "model1_ridge.pkl"))
        return ("ridge", pipe)

    if model_kind == "xgb_mean":
        if xgb is None:
            raise RuntimeError("xgboost is not installed.")
        pre = joblib.load(os.path.join(models_dir, "model1_preprocessor.pkl"))
        model = xgb.XGBRegressor()
        model.load_model(os.path.join(models_dir, "model1_xgb_mean.json"))
        return ("xgb", (pre, model))

    raise ValueError(f"Unknown model_kind: {model_kind}")


def predict_with_tyre_age_override(
    df_split: pd.DataFrame, models_dir: str, model_kind: str, tyre_age_value: float | None
) -> np.ndarray:
    X = df_split[MODEL1_FEATURES].copy()
    if tyre_age_value is not None:
        X["tyre_age"] = float(tyre_age_value)

    kind, obj = _load_primary_predictor(models_dir, model_kind)
    if kind == "ridge":
        return obj.predict(X)
    pre, model = obj
    return model.predict(pre.transform(X))


def predict_with_tyre_age_minus_one(
    df_split: pd.DataFrame, models_dir: str, model_kind: str, fresh_tyre_age: float
) -> np.ndarray:
    X = df_split[MODEL1_FEATURES].copy()
    age = pd.to_numeric(X["tyre_age"], errors="coerce").fillna(fresh_tyre_age).values.astype(float)
    X["tyre_age"] = np.maximum(age - 1.0, float(fresh_tyre_age))

    kind, obj = _load_primary_predictor(models_dir, model_kind)
    if kind == "ridge":
        return obj.predict(X)
    pre, model = obj
    return model.predict(pre.transform(X))


def mask_bad_degradation_rows(df_split: pd.DataFrame) -> np.ndarray:
    mask = np.zeros(len(df_split), dtype=bool)

    for c in ["is_under_safety_car", "is_under_virtual_safety_car", "is_out_lap", "is_in_lap", "is_pit_lap"]:
        if c in df_split.columns:
            mask |= (_coerce_bool01(df_split[c]).values == 1)

    return mask


def robust_sigma_mad(resid: np.ndarray) -> float:
    resid = resid[np.isfinite(resid)]
    if resid.size < 30:
        return float(np.nan)
    med = float(np.nanmedian(resid))
    mad = float(np.nanmedian(np.abs(resid - med)))
    return float(1.4826 * mad)


def build_sigma_table(
    df_train: pd.DataFrame,
    pred_train: np.ndarray,
    min_group_n: int = 200,
    global_fallback_sigma: float = 1.5,
) -> pd.DataFrame:
    tmp = df_train[["circuit_id", "tyre_compound"]].copy()
    tmp["y"] = df_train[TARGET_COL].values.astype(float)
    tmp["p"] = np.asarray(pred_train, dtype=float)
    tmp["resid"] = tmp["y"] - tmp["p"]
    tmp = tmp[np.isfinite(tmp["resid"].values)]

    rows = []
    for (cid, comp), g in tmp.groupby(["circuit_id", "tyre_compound"], dropna=False):
        r = g["resid"].values.astype(float)
        sig = robust_sigma_mad(r)
        n = int(np.isfinite(r).sum())
        if (not np.isfinite(sig)) or n < int(min_group_n):
            continue
        rows.append({"circuit_id": cid, "tyre_compound": comp, "lap_time_sigma": float(sig), "n": n})

    tab = pd.DataFrame(rows)
    if len(tab) == 0:
        tab = pd.DataFrame(columns=["circuit_id", "tyre_compound", "lap_time_sigma", "n"])

    tab["lap_time_sigma"] = tab["lap_time_sigma"].clip(lower=0.2, upper=6.0)

    global_sig = robust_sigma_mad(tmp["resid"].values.astype(float))
    if not np.isfinite(global_sig):
        global_sig = float(global_fallback_sigma)
    global_sig = float(np.clip(global_sig, 0.2, 6.0))

    tab.attrs["global_sigma"] = global_sig
    return tab


def attach_sigma(df_split: pd.DataFrame, sigma_table: pd.DataFrame) -> np.ndarray:
    global_sig = float(sigma_table.attrs.get("global_sigma", 1.5))

    if len(sigma_table) == 0:
        return np.full(len(df_split), global_sig, dtype=float)

    tmp = df_split[["circuit_id", "tyre_compound"]].copy()
    tmp = tmp.merge(sigma_table[["circuit_id", "tyre_compound", "lap_time_sigma"]], on=["circuit_id", "tyre_compound"], how="left")
    sig = tmp["lap_time_sigma"].values.astype(float)
    sig = np.where(np.isfinite(sig), sig, global_sig)
    return sig


def learn_deg_floor_table(
    df_train: pd.DataFrame,
    per_lap_train: np.ndarray,
    q: float,
    min_age: float,
    max_age: float,
    global_min_floor: float,
    clip_high_quantile: float = 0.95,
) -> pd.DataFrame:
    tmp = pd.DataFrame(
        {
            "circuit_id": df_train["circuit_id"].values,
            "tyre_compound": df_train["tyre_compound"].values,
            "tyre_age": pd.to_numeric(df_train["tyre_age"], errors="coerce").values.astype(float),
            "perlap": np.asarray(per_lap_train, dtype=float),
        }
    )
    tmp = tmp.dropna(subset=["circuit_id", "tyre_compound", "tyre_age", "perlap"])
    tmp = tmp[np.isfinite(tmp["perlap"].values)]
    tmp = tmp[(tmp["tyre_age"] >= float(min_age)) & (tmp["tyre_age"] <= float(max_age))]
    tmp = tmp[tmp["perlap"] > 0.0]

    if len(tmp) == 0:
        return pd.DataFrame(columns=["circuit_id", "tyre_compound", "deg_floor"])

    high = float(np.nanquantile(tmp["perlap"].values, clip_high_quantile))
    tmp["perlap"] = tmp["perlap"].clip(upper=high)

    floors = (
        tmp.groupby(["circuit_id", "tyre_compound"], dropna=False)["perlap"]
        .quantile(float(q))
        .reset_index()
        .rename(columns={"perlap": "deg_floor"})
    )
    floors["deg_floor"] = floors["deg_floor"].clip(lower=float(global_min_floor))
    return floors


def apply_deg_floor_conditionally(
    df_split: pd.DataFrame,
    per_lap: np.ndarray,
    floors: pd.DataFrame,
    apply_only_if_below: float,
) -> np.ndarray:
    out = np.asarray(per_lap, dtype=float).copy()

    if floors is None or len(floors) == 0:
        return out

    tmp = df_split[["circuit_id", "tyre_compound"]].copy()
    tmp["perlap"] = out
    tmp = tmp.merge(floors, on=["circuit_id", "tyre_compound"], how="left")

    floor = tmp["deg_floor"].values.astype(float)

    use = np.isfinite(out) & np.isfinite(floor) & (out < float(apply_only_if_below))
    out[use] = np.maximum(out[use], floor[use])
    return out


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--input_path", type=str, required=True)
    ap.add_argument("--outputs_dir", type=str, default="outputs")
    ap.add_argument("--models_dir", type=str, default="models")

    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--min_lap_time", type=float, default=40.0)
    ap.add_argument("--max_lap_time", type=float, default=200.0)

    ap.add_argument("--test_size", type=float, default=0.2)
    ap.add_argument("--val_size_within_train", type=float, default=0.2)

    ap.add_argument("--run_ridge", type=int, default=1)
    ap.add_argument("--run_xgb", type=int, default=1)
    ap.add_argument("--optuna_trials", type=int, default=40)

    ap.add_argument("--fresh_tyre_age", type=float, default=0.0)

    ap.add_argument("--deg_per_lap_cap", type=float, default=1.0)

    ap.add_argument("--apply_deg_floor", type=int, default=1)
    ap.add_argument("--deg_floor_quantile", type=float, default=0.05)
    ap.add_argument("--deg_floor_min_age", type=float, default=3.0)
    ap.add_argument("--deg_floor_max_age", type=float, default=25.0)
    ap.add_argument("--deg_floor_global_min", type=float, default=0.01)
    ap.add_argument("--deg_floor_apply_below", type=float, default=0.005)

    ap.add_argument("--sigma_min_group_n", type=int, default=200)
    ap.add_argument("--sigma_global_fallback", type=float, default=1.5)

    args = ap.parse_args()

    ensure_dir(args.outputs_dir)
    ensure_dir(args.models_dir)
    tag = now_tag()

    df = load_and_filter(args.input_path, args.min_lap_time, args.max_lap_time)
    df_train, df_val, df_test = split_by_race_id(df, args.test_size, args.val_size_within_train, args.seed)

    metrics: Dict = {
        "rows_after_filter": int(len(df)),
        "train_rows": int(len(df_train)),
        "val_rows": int(len(df_val)),
        "test_rows": int(len(df_test)),
        "unique_races_total": int(df["race_id"].nunique()),
        "unique_races_train": int(df_train["race_id"].nunique()),
        "unique_races_val": int(df_val["race_id"].nunique()),
        "unique_races_test": int(df_test["race_id"].nunique()),
        "fresh_tyre_age": float(args.fresh_tyre_age),
        "deg_settings": {
            "per_lap_cap": float(args.deg_per_lap_cap),
            "apply_floor": int(args.apply_deg_floor),
            "floor_quantile": float(args.deg_floor_quantile),
            "floor_age_band": [float(args.deg_floor_min_age), float(args.deg_floor_max_age)],
            "floor_global_min": float(args.deg_floor_global_min),
            "floor_apply_below": float(args.deg_floor_apply_below),
        },
        "sigma_settings": {
            "min_group_n": int(args.sigma_min_group_n),
            "global_fallback": float(args.sigma_global_fallback),
        },
    }

    if args.run_ridge == 1:
        metrics["ridge"] = train_ridge(df_train, df_val, args.models_dir)
    if args.run_xgb == 1:
        metrics["xgb_mean"] = train_xgb_mean_with_optuna(df_train, df_val, args.models_dir, args.optuna_trials, args.seed)

    primary = "xgb_mean" if args.run_xgb == 1 else "ridge"

    pred_train = predict_with_tyre_age_override(df_train, args.models_dir, primary, tyre_age_value=None)
    pred_val = predict_with_tyre_age_override(df_val, args.models_dir, primary, tyre_age_value=None)
    pred_test = predict_with_tyre_age_override(df_test, args.models_dir, primary, tyre_age_value=None)

    pred_train_fresh = predict_with_tyre_age_override(df_train, args.models_dir, primary, tyre_age_value=args.fresh_tyre_age)
    pred_val_fresh = predict_with_tyre_age_override(df_val, args.models_dir, primary, tyre_age_value=args.fresh_tyre_age)
    pred_test_fresh = predict_with_tyre_age_override(df_test, args.models_dir, primary, tyre_age_value=args.fresh_tyre_age)

    pred_train_minus1 = predict_with_tyre_age_minus_one(df_train, args.models_dir, primary, args.fresh_tyre_age)
    pred_val_minus1 = predict_with_tyre_age_minus_one(df_val, args.models_dir, primary, args.fresh_tyre_age)
    pred_test_minus1 = predict_with_tyre_age_minus_one(df_test, args.models_dir, primary, args.fresh_tyre_age)

    deg_delta_raw_train = np.maximum(pred_train - pred_train_fresh, 0.0)
    deg_delta_raw_val = np.maximum(pred_val - pred_val_fresh, 0.0)
    deg_delta_raw_test = np.maximum(pred_test - pred_test_fresh, 0.0)

    per_lap_train = np.maximum(pred_train - pred_train_minus1, 0.0)
    per_lap_val = np.maximum(pred_val - pred_val_minus1, 0.0)
    per_lap_test = np.maximum(pred_test - pred_test_minus1, 0.0)

    cap = float(args.deg_per_lap_cap)
    if cap > 0:
        per_lap_train = np.minimum(per_lap_train, cap)
        per_lap_val = np.minimum(per_lap_val, cap)
        per_lap_test = np.minimum(per_lap_test, cap)

    mask_train = mask_bad_degradation_rows(df_train)
    mask_val = mask_bad_degradation_rows(df_val)
    mask_test = mask_bad_degradation_rows(df_test)

    per_lap_train = np.where(mask_train, np.nan, per_lap_train)
    per_lap_val = np.where(mask_val, np.nan, per_lap_val)
    per_lap_test = np.where(mask_test, np.nan, per_lap_test)

    deg_delta_raw_train = np.where(mask_train, np.nan, deg_delta_raw_train)
    deg_delta_raw_val = np.where(mask_val, np.nan, deg_delta_raw_val)
    deg_delta_raw_test = np.where(mask_test, np.nan, deg_delta_raw_test)

    floors = None
    if args.apply_deg_floor == 1:
        floors = learn_deg_floor_table(
            df_train=df_train,
            per_lap_train=per_lap_train,
            q=args.deg_floor_quantile,
            min_age=args.deg_floor_min_age,
            max_age=args.deg_floor_max_age,
            global_min_floor=args.deg_floor_global_min,
        )
        floors_path = os.path.join(args.outputs_dir, f"model1_deg_floor_{tag}.csv")
        floors.to_csv(floors_path, index=False)
        metrics["deg_floor_table_path"] = floors_path

        per_lap_train = apply_deg_floor_conditionally(df_train, per_lap_train, floors, args.deg_floor_apply_below)
        per_lap_val = apply_deg_floor_conditionally(df_val, per_lap_val, floors, args.deg_floor_apply_below)
        per_lap_test = apply_deg_floor_conditionally(df_test, per_lap_test, floors, args.deg_floor_apply_below)

        per_lap_train = np.where(mask_train, np.nan, per_lap_train)
        per_lap_val = np.where(mask_val, np.nan, per_lap_val)
        per_lap_test = np.where(mask_test, np.nan, per_lap_test)

    age_train = pd.to_numeric(df_train["tyre_age"], errors="coerce").fillna(args.fresh_tyre_age).values.astype(float)
    age_val = pd.to_numeric(df_val["tyre_age"], errors="coerce").fillna(args.fresh_tyre_age).values.astype(float)
    age_test = pd.to_numeric(df_test["tyre_age"], errors="coerce").fillna(args.fresh_tyre_age).values.astype(float)

    dist_train = np.maximum(age_train - float(args.fresh_tyre_age), 0.0)
    dist_val = np.maximum(age_val - float(args.fresh_tyre_age), 0.0)
    dist_test = np.maximum(age_test - float(args.fresh_tyre_age), 0.0)

    deg_delta_train = np.where(np.isfinite(per_lap_train), per_lap_train * dist_train, np.nan)
    deg_delta_val = np.where(np.isfinite(per_lap_val), per_lap_val * dist_val, np.nan)
    deg_delta_test = np.where(np.isfinite(per_lap_test), per_lap_test * dist_test, np.nan)

    deg_delta_train = np.where(np.isfinite(deg_delta_raw_train), np.maximum(deg_delta_train, deg_delta_raw_train), deg_delta_train)
    deg_delta_val = np.where(np.isfinite(deg_delta_raw_val), np.maximum(deg_delta_val, deg_delta_raw_val), deg_delta_val)
    deg_delta_test = np.where(np.isfinite(deg_delta_raw_test), np.maximum(deg_delta_test, deg_delta_raw_test), deg_delta_test)

    sigma_table = build_sigma_table(
        df_train=df_train,
        pred_train=pred_train,
        min_group_n=args.sigma_min_group_n,
        global_fallback_sigma=args.sigma_global_fallback,
    )
    sigma_path = os.path.join(args.outputs_dir, f"model1_sigma_table_{tag}.csv")
    sigma_table.to_csv(sigma_path, index=False)
    metrics["sigma_table_path"] = sigma_path
    metrics["sigma_global"] = float(sigma_table.attrs.get("global_sigma", args.sigma_global_fallback))

    sigma_train = attach_sigma(df_train, sigma_table)
    sigma_val = attach_sigma(df_val, sigma_table)
    sigma_test = attach_sigma(df_test, sigma_table)

    metrics["mae_train"] = float(mean_absolute_error(df_train[TARGET_COL].values, pred_train))
    metrics["mae_val"] = float(mean_absolute_error(df_val[TARGET_COL].values, pred_val))
    metrics["mae_test"] = float(mean_absolute_error(df_test[TARGET_COL].values, pred_test))

    preds_all = pd.concat(
        [
            pd.DataFrame({**{c: df_train[c].values for c in KEY_COLS}, "expected_lap_time": pred_train}),
            pd.DataFrame({**{c: df_val[c].values for c in KEY_COLS}, "expected_lap_time": pred_val}),
            pd.DataFrame({**{c: df_test[c].values for c in KEY_COLS}, "expected_lap_time": pred_test}),
        ],
        ignore_index=True,
    )

    extra_all = pd.concat(
        [
            pd.DataFrame(
                {
                    "race_id": df_train["race_id"].values,
                    "driver_id": df_train["driver_id"].values,
                    "lap_number": df_train["lap_number"].values,
                    "expected_lap_time_fresh": pred_train_fresh,
                    "tyre_degradation_delta": deg_delta_train,
                    "tyre_degradation_per_lap": per_lap_train,
                    "lap_time_sigma": sigma_train,
                    "lap_time_variance": sigma_train ** 2,
                }
            ),
            pd.DataFrame(
                {
                    "race_id": df_val["race_id"].values,
                    "driver_id": df_val["driver_id"].values,
                    "lap_number": df_val["lap_number"].values,
                    "expected_lap_time_fresh": pred_val_fresh,
                    "tyre_degradation_delta": deg_delta_val,
                    "tyre_degradation_per_lap": per_lap_val,
                    "lap_time_sigma": sigma_val,
                    "lap_time_variance": sigma_val ** 2,
                }
            ),
            pd.DataFrame(
                {
                    "race_id": df_test["race_id"].values,
                    "driver_id": df_test["driver_id"].values,
                    "lap_number": df_test["lap_number"].values,
                    "expected_lap_time_fresh": pred_test_fresh,
                    "tyre_degradation_delta": deg_delta_test,
                    "tyre_degradation_per_lap": per_lap_test,
                    "lap_time_sigma": sigma_test,
                    "lap_time_variance": sigma_test ** 2,
                }
            ),
        ],
        ignore_index=True,
    )

    full = pd.read_csv(args.input_path)
    merged = full.merge(preds_all, on=KEY_COLS, how="left").merge(extra_all, on=KEY_COLS, how="left")
    out_path = os.path.join(args.outputs_dir, f"lap_level_with_model1_{tag}.csv")
    merged.to_csv(out_path, index=False)
    metrics["merged_output_path"] = out_path
    metrics["primary_model"] = primary

    metrics_path = os.path.join(args.outputs_dir, f"model1_metrics_{tag}.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print("Done")
    print(f"Merged dataset: {out_path}")
    print(f"Metrics: {metrics_path}")
    print(f"Sigma table: {sigma_path}")
    if args.apply_deg_floor == 1:
        print(f"Floor table: {metrics.get('deg_floor_table_path')}")


if __name__ == "__main__":
    main()