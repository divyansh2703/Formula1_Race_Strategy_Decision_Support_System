from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
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


def residual_bucket_summary(y_true: np.ndarray, y_hat: np.ndarray) -> Dict:
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


def get_feature_cols(df: pd.DataFrame) -> tuple[list[str], list[str]]:
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
    ]

    cat_cols = [c for c in cat_cols if c in df.columns]
    num_cols = [c for c in num_cols if c in df.columns]
    return cat_cols, num_cols


def tune_xgb_with_optuna(X_train, y_train, X_val, y_val, seed: int, n_trials: int) -> Dict:
    if XGBRegressor is None:
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
        overfit_penalty = max(0.0, rmse_train - rmse_val)
        score = -rmse_val - 0.15 * abs(overfit_penalty)
        return float(score)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best = dict(study.best_params)
    best["best_objective"] = float(study.best_value)
    return best


def fit_ridge_model(X_train_raw, y_train, X_val_raw, y_val, X_test_raw, y_test, cat_cols, num_cols) -> Dict:
    pre = build_preprocessor(cat_cols, num_cols, scale_numeric=True)
    pipe = Pipeline(
        steps=[
            ("preprocessor", pre),
            ("model", Ridge(alpha=1.0)),
        ]
    )
    pipe.fit(X_train_raw, y_train)

    pred_train = pipe.predict(X_train_raw)
    pred_val = pipe.predict(X_val_raw)
    pred_test = pipe.predict(X_test_raw)

    return {
        "pipeline": pipe,
        "train": score_regression(y_train, pred_train),
        "val": score_regression(y_val, pred_val),
        "test": score_regression(y_test, pred_test),
        "pred_train": pred_train,
        "pred_val": pred_val,
        "pred_test": pred_test,
    }


def fit_xgb_model(X_train_raw, y_train, X_val_raw, y_val, X_test_raw, y_test, cat_cols, num_cols, seed: int, optuna_trials: int) -> Dict:
    if XGBRegressor is None:
        raise RuntimeError("xgboost is not installed")

    pre = build_preprocessor(cat_cols, num_cols, scale_numeric=False)
    X_train = pre.fit_transform(X_train_raw)
    X_val = pre.transform(X_val_raw)
    X_test = pre.transform(X_test_raw)

    if optuna_trials > 0 and optuna is not None:
        best = tune_xgb_with_optuna(X_train, y_train, X_val, y_val, seed=seed, n_trials=optuna_trials)
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
    fit_xgb_with_early_stopping(model, X_train, y_train, X_val, y_val, rounds=35)

    pred_train = model.predict(X_train)
    pred_val = model.predict(X_val)
    pred_test = model.predict(X_test)

    return {
        "preprocessor": pre,
        "model": model,
        "best_params": best,
        "train": score_regression(y_train, pred_train),
        "val": score_regression(y_val, pred_val),
        "test": score_regression(y_test, pred_test),
        "pred_train": pred_train,
        "pred_val": pred_val,
        "pred_test": pred_test,
    }


def choose_model(ridge_res: Dict, xgb_res: Dict) -> str:
    ridge_val_rmse = ridge_res["val"]["rmse"]
    xgb_val_rmse = xgb_res["val"]["rmse"]

    ridge_gap = max(0.0, ridge_res["val"]["rmse"] - ridge_res["train"]["rmse"])
    xgb_gap = max(0.0, xgb_res["val"]["rmse"] - xgb_res["train"]["rmse"])

    ridge_score = -ridge_val_rmse - 0.10 * ridge_gap
    xgb_score = -xgb_val_rmse - 0.20 * xgb_gap

    return "ridge" if ridge_score >= xgb_score else "xgb"


def predict_all_ridge(pipe: Pipeline, X_all_raw: pd.DataFrame) -> np.ndarray:
    return pipe.predict(X_all_raw)


def predict_all_xgb(pre, model, X_all_raw: pd.DataFrame) -> np.ndarray:
    X_all = pre.transform(X_all_raw)
    return model.predict(X_all)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_path", required=True, help="model4a_clean_air_scored.csv or training table with clean air predictions")
    ap.add_argument("--outputs_dir", default="outputs")
    ap.add_argument("--models_dir", default="models")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--test_frac", type=float, default=0.20)
    ap.add_argument("--val_frac", type=float, default=0.20)
    ap.add_argument("--optuna_trials", type=int, default=40)
    ap.add_argument("--scored_output_name", default="model4b_position_delta_scored.csv")
    args = ap.parse_args()

    ensure_dir(args.outputs_dir)
    ensure_dir(args.models_dir)

    ts = now_ts()

    df = pd.read_csv(args.input_path, low_memory=False)

    required_cols = ["race_id", "driver_id", "lap_number", "y_rejoin_position_delta_nextlap"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df.copy()
    df["y_rejoin_position_delta_nextlap"] = pd.to_numeric(df["y_rejoin_position_delta_nextlap"], errors="coerce")
    df = df[df["y_rejoin_position_delta_nextlap"].notna()].copy()

    splits = make_group_split(
        race_ids=df["race_id"].astype(str).to_numpy(),
        seed=args.seed,
        test_frac=args.test_frac,
        val_frac=args.val_frac,
    )

    splits_path = os.path.join(args.outputs_dir, f"model4b_position_delta_race_splits_{ts}.json")
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
    feature_cols = cat_cols + num_cols

    X_all = df[feature_cols].copy()
    for c in cat_cols:
        X_all[c] = X_all[c].astype(str)
    for c in num_cols:
        X_all[c] = pd.to_numeric(X_all[c], errors="coerce")

    y_all = df["y_rejoin_position_delta_nextlap"].astype(float).to_numpy()

    tr = np.isin(df["race_id"].astype(str).to_numpy(), np.array(splits.train_races, dtype=str))
    va = np.isin(df["race_id"].astype(str).to_numpy(), np.array(splits.val_races, dtype=str))
    te = np.isin(df["race_id"].astype(str).to_numpy(), np.array(splits.test_races, dtype=str))

    X_train = X_all.loc[tr].copy()
    X_val = X_all.loc[va].copy()
    X_test = X_all.loc[te].copy()

    y_train = y_all[tr]
    y_val = y_all[va]
    y_test = y_all[te]

    ridge_res = fit_ridge_model(
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
        optuna_trials=args.optuna_trials,
    )

    winner = choose_model(ridge_res, xgb_res)

    if winner == "xgb":
        pred_all = predict_all_xgb(
            pre=xgb_res["preprocessor"],
            model=xgb_res["model"],
            X_all_raw=X_all,
        )
        pred_test = xgb_res["pred_test"]
    else:
        pred_all = predict_all_ridge(
            pipe=ridge_res["pipeline"],
            X_all_raw=X_all,
        )
        pred_test = ridge_res["pred_test"]

    base = f"model4b_position_delta_{ts}"

    if winner == "xgb":
        joblib.dump(xgb_res["preprocessor"], os.path.join(args.models_dir, f"{base}_xgb_preprocessor.pkl"))
        joblib.dump(xgb_res["model"], os.path.join(args.models_dir, f"{base}_xgb.pkl"))
    else:
        joblib.dump(ridge_res["pipeline"], os.path.join(args.models_dir, f"{base}_ridge.pkl"))

    metrics = {
        "target": "y_rejoin_position_delta_nextlap",
        "mode": "regression_position_delta",
        "winner": winner,
        "cat_cols": cat_cols,
        "num_cols": num_cols,
        "feature_cols": feature_cols,
        "ridge": {
            "train": ridge_res["train"],
            "val": ridge_res["val"],
            "test": ridge_res["test"],
            "test_residual_summary": residual_bucket_summary(y_test, ridge_res["pred_test"]),
        },
        "xgb": {
            "train": xgb_res["train"],
            "val": xgb_res["val"],
            "test": xgb_res["test"],
            "best_params": xgb_res["best_params"],
            "test_residual_summary": residual_bucket_summary(y_test, xgb_res["pred_test"]),
        },
        "winner_residual_summary_test": residual_bucket_summary(y_test, pred_test),
    }

    metrics_path = os.path.join(args.outputs_dir, f"{base}_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    scored = df.copy()
    scored["pred_rejoin_position_delta_nextlap"] = pred_all.astype(float)

    scored_path = os.path.join(args.outputs_dir, args.scored_output_name)
    scored.to_csv(scored_path, index=False)

    split_pred = df[["race_id", "driver_id", "lap_number", "y_rejoin_position_delta_nextlap"]].copy()
    split_pred["split"] = np.where(tr, "train", np.where(va, "val", np.where(te, "test", "other")))
    split_pred["pred_rejoin_position_delta_nextlap"] = pred_all.astype(float)
    split_pred_path = os.path.join(args.outputs_dir, f"{base}_all_predictions.csv")
    split_pred.to_csv(split_pred_path, index=False)

    print("DONE MODEL 4B position delta regressor")
    print(f"Splits: {splits_path}")
    print(f"Metrics: {metrics_path}")
    print(f"Scored output: {scored_path}")
    print(f"All predictions: {split_pred_path}")


if __name__ == "__main__":
    main()