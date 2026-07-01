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


@dataclass
class SplitIds:
    train_races: List[str]
    val_races: List[str]
    test_races: List[str]


def now_ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


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
    ]

    cat_cols = [c for c in cat_cols if c in df.columns]
    num_cols = [c for c in num_cols if c in df.columns]
    return cat_cols, num_cols


def tune_xgb_with_optuna(X_train, y_train, X_val, y_val, seed: int, n_trials: int, scale_pos_weight: float) -> Dict:
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


def choose_model(log_res: Dict, xgb_res: Dict) -> str:
    log_val = log_res["val"]["pr_auc"]
    xgb_val = xgb_res["val"]["pr_auc"]

    log_brier = log_res["val"]["brier"]
    xgb_brier = xgb_res["val"]["brier"]

    log_gap = max(0.0, log_res["train"]["pr_auc"] - log_res["val"]["pr_auc"])
    xgb_gap = max(0.0, xgb_res["train"]["pr_auc"] - xgb_res["val"]["pr_auc"])

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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_path", required=True, help="model4_pit_event_training_table.csv")
    ap.add_argument("--outputs_dir", default="outputs")
    ap.add_argument("--models_dir", default="models")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--test_frac", type=float, default=0.20)
    ap.add_argument("--val_frac", type=float, default=0.20)
    ap.add_argument("--optuna_trials", type=int, default=40)
    ap.add_argument("--scored_output_name", default="model4a_clean_air_scored.csv")
    args = ap.parse_args()

    ensure_dir(args.outputs_dir)
    ensure_dir(args.models_dir)

    ts = now_ts()

    df = pd.read_csv(args.input_path, low_memory=False)

    required_cols = ["race_id", "driver_id", "lap_number", "y_rejoin_clean_air_nextlap"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df.copy()
    df["y_rejoin_clean_air_nextlap"] = pd.to_numeric(df["y_rejoin_clean_air_nextlap"], errors="coerce")
    df = df[df["y_rejoin_clean_air_nextlap"].notna()].copy()
    df["y_rejoin_clean_air_nextlap"] = df["y_rejoin_clean_air_nextlap"].astype(int)

    splits = make_group_split(
        race_ids=df["race_id"].astype(str).to_numpy(),
        seed=args.seed,
        test_frac=args.test_frac,
        val_frac=args.val_frac,
    )

    splits_path = os.path.join(args.outputs_dir, f"model4a_clean_air_race_splits_{ts}.json")
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

    y_all = df["y_rejoin_clean_air_nextlap"].astype(int).to_numpy()

    tr = np.isin(df["race_id"].astype(str).to_numpy(), np.array(splits.train_races, dtype=str))
    va = np.isin(df["race_id"].astype(str).to_numpy(), np.array(splits.val_races, dtype=str))
    te = np.isin(df["race_id"].astype(str).to_numpy(), np.array(splits.test_races, dtype=str))

    X_train = X_all.loc[tr].copy()
    X_val = X_all.loc[va].copy()
    X_test = X_all.loc[te].copy()

    y_train = y_all[tr]
    y_val = y_all[va]
    y_test = y_all[te]

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
        optuna_trials=args.optuna_trials,
    )

    winner = choose_model(log_res, xgb_res)

    if winner == "xgb":
        p_all = predict_all_xgb(
            pre=xgb_res["preprocessor"],
            model=xgb_res["model"],
            calib=xgb_res["calibration"],
            X_all_raw=X_all,
        )
        p_test = xgb_res["p_test"]
    else:
        p_all = predict_all_logistic(
            pipe=log_res["pipeline"],
            calib=log_res["calibration"],
            X_all_raw=X_all,
        )
        p_test = log_res["p_test"]

    pred_class = (p_all >= 0.50).astype(int)

    base = f"model4a_clean_air_{ts}"

    if winner == "xgb":
        joblib.dump(xgb_res["preprocessor"], os.path.join(args.models_dir, f"{base}_xgb_preprocessor.pkl"))
        joblib.dump(xgb_res["model"], os.path.join(args.models_dir, f"{base}_xgb.pkl"))
        with open(os.path.join(args.models_dir, f"{base}_calibration.json"), "w") as f:
            json.dump(xgb_res["calibration"], f, indent=2)
    else:
        joblib.dump(log_res["pipeline"], os.path.join(args.models_dir, f"{base}_logistic.pkl"))
        with open(os.path.join(args.models_dir, f"{base}_calibration.json"), "w") as f:
            json.dump(log_res["calibration"], f, indent=2)

    metrics = {
        "target": "y_rejoin_clean_air_nextlap",
        "mode": "binary_clean_air_classifier",
        "winner": winner,
        "cat_cols": cat_cols,
        "num_cols": num_cols,
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
        "winner_test_reliability": reliability_table(y_test, p_test, n_bins=10),
    }

    metrics_path = os.path.join(args.outputs_dir, f"{base}_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    scored = df.copy()
    scored["p_rejoin_clean_air_nextlap"] = p_all.astype(float)
    scored["pred_rejoin_clean_air_nextlap"] = pred_class.astype(int)

    scored_path = os.path.join(args.outputs_dir, args.scored_output_name)
    scored.to_csv(scored_path, index=False)

    # Save split predictions for inspection
    split_pred = df[["race_id", "driver_id", "lap_number", "y_rejoin_clean_air_nextlap"]].copy()
    split_pred["split"] = np.where(tr, "train", np.where(va, "val", np.where(te, "test", "other")))
    split_pred["p_rejoin_clean_air_nextlap"] = p_all.astype(float)
    split_pred["pred_rejoin_clean_air_nextlap"] = pred_class.astype(int)
    split_pred_path = os.path.join(args.outputs_dir, f"{base}_all_predictions.csv")
    split_pred.to_csv(split_pred_path, index=False)

    print("DONE MODEL 4A clean air classifier")
    print(f"Splits: {splits_path}")
    print(f"Metrics: {metrics_path}")
    print(f"Scored output: {scored_path}")
    print(f"All predictions: {split_pred_path}")


if __name__ == "__main__":
    main()