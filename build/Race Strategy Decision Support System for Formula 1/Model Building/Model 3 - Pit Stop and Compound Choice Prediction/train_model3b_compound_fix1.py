from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from typing import Dict, List

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import confusion_matrix
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


DRY_CLASSES = ["SOFT", "MEDIUM", "HARD"]


def now_ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


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


def load_splits(splits_path: str) -> Dict[str, List[str]]:
    with open(splits_path, "r") as f:
        splits = json.load(f)

    for k in ["train_races", "val_races", "test_races"]:
        if k not in splits:
            raise ValueError(f"Missing key in splits file: {k}")

    return splits


def add_compound_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    numeric_candidates = [
        "lap_number",
        "laps_remaining",
        "stint_number",
        "pit_stop_number",
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
        "total_laps",
        "is_abnormally_long_stop",
        "lap1_stop_flag",
        "stationary_missing_flag",
        "model2_event_prob_imputed_any",
    ]
    for c in numeric_candidates:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")

    for c in [
        "driver_id",
        "team_id",
        "circuit_id",
        "season",
        "race_phase",
        "tyre_compound",
        "tyre_compound_before",
        "race_level_weather_condition",
    ]:
        if c in out.columns:
            out[c] = out[c].astype(str).fillna("UNKNOWN")

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

    if "tyre_age" in out.columns:
        out["fresh_tyre_flag"] = (out["tyre_age"].fillna(-1) <= 1).astype(float)

    if "stint_number" in out.columns:
        stint_num = pd.to_numeric(out["stint_number"], errors="coerce").fillna(0)
        out["is_first_stint_flag"] = (stint_num <= 1).astype(float)
        out["is_late_stint_flag"] = (stint_num >= 3).astype(float)

    if "laps_remaining" in out.columns:
        laps_remaining = pd.to_numeric(out["laps_remaining"], errors="coerce")
        out["likely_final_stop_flag"] = (laps_remaining <= 25).astype(float)

    hazard_cols_short = [c for c in ["p_sc_next3", "p_vsc_next3", "p_rf_next3"] if c in out.columns]
    if hazard_cols_short:
        out["neutralisation_risk_next3_max"] = out[hazard_cols_short].max(axis=1)
        out["neutralisation_risk_next3_sum"] = out[hazard_cols_short].sum(axis=1)

    return out


def get_feature_lists(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    cat_cols = [
        "driver_id",
        "team_id",
        "circuit_id",
        "season",
        "race_phase",
        "tyre_compound",
        "tyre_compound_before",
        "race_level_weather_condition",
    ]

    num_cols = [
        "lap_number",
        "laps_remaining",
        "stint_number",
        "pit_stop_number",
        "tyre_age",
        "position",
        "gap_ahead",
        "gap_behind",
        "clean_air_flag",
        "track_temperature_used",
        "air_temperature_used",
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
        "fresh_tyre_flag",
        "is_first_stint_flag",
        "is_late_stint_flag",
        "likely_final_stop_flag",
        "neutralisation_risk_next3_max",
        "neutralisation_risk_next3_sum",
    ]

    cat_cols = [c for c in cat_cols if c in df.columns]
    num_cols = [c for c in num_cols if c in df.columns]
    return cat_cols, num_cols


def optuna_tune_multiclass(
    X_train,
    y_train,
    X_val,
    y_val,
    num_class: int,
    seed: int,
    trials: int,
    sample_weight_train=None,
    sample_weight_val=None,
) -> Dict:
    if optuna is None:
        raise RuntimeError("optuna is not installed")
    if XGBClassifier is None:
        raise RuntimeError("xgboost is not installed")

    def objective(trial: optuna.Trial) -> float:
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

        fit_kwargs = {"sample_weight": sample_weight_train} if sample_weight_train is not None else {}
        eval_set = [(X_val, y_val)]
        try:
            model.fit(
                X_train,
                y_train,
                eval_set=eval_set,
                sample_weight_eval_set=[sample_weight_val] if sample_weight_val is not None else None,
                verbose=False,
                early_stopping_rounds=35,
                **fit_kwargs,
            )
        except TypeError:
            fit_xgb_with_early_stopping(model, X_train, y_train, X_val, y_val, rounds=35)

        p = model.predict_proba(X_val)
        eps = 1e-12
        mlogloss = -np.mean(np.log(np.clip(p[np.arange(len(y_val)), y_val], eps, 1.0)))
        return float(-mlogloss)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=trials, show_progress_bar=False)

    best = dict(study.best_params)
    best["best_objective_neg_logloss"] = float(study.best_value)
    return best


def compute_class_weights(y: np.ndarray, num_class: int) -> np.ndarray:
    counts = np.bincount(y, minlength=num_class).astype(float)
    counts[counts == 0] = 1.0
    weights = counts.sum() / (num_class * counts)
    return weights


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_path", required=True)
    ap.add_argument("--outputs_dir", default="outputs")
    ap.add_argument("--models_dir", default="models")
    ap.add_argument("--splits_path", required=True, help="Use the same race split JSON as Model 3")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--optuna_trials", type=int, default=60)
    ap.add_argument("--merge_output_name", default="lap_level_with_model1_model2_model3_compoundfixed.csv")
    ap.add_argument("--exclude_abnormal_long_stops", type=int, default=1)
    ap.add_argument("--exclude_lap1_stops", type=int, default=1)
    ap.add_argument("--dry_rain_flag_value", type=int, default=0)
    args = ap.parse_args()

    ensure_dir(args.outputs_dir)
    ensure_dir(args.models_dir)

    if XGBClassifier is None:
        raise RuntimeError("xgboost is not installed")

    ts = now_ts()
    splits = load_splits(args.splits_path)

    df = pd.read_csv(args.input_path, low_memory=False)
    pk = ["race_id", "driver_id", "lap_number"]
    if df.duplicated(pk).any():
        raise ValueError("Primary key duplicates found")

    df = add_compound_features(df)

    required = ["pitted_this_lap_flag", "tyre_compound_after", "rain_flag_used"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    pit = df[df["pitted_this_lap_flag"].fillna(0).astype(int) == 1].copy()

    pit["tyre_compound_after"] = pit["tyre_compound_after"].astype(str)
    pit["tyre_compound_after"] = pit["tyre_compound_after"].replace({"nan": np.nan, "None": np.nan, "": np.nan})
    pit = pit.dropna(subset=["tyre_compound_after"]).copy()

    # Dry regime only
    pit = pit[pd.to_numeric(pit["rain_flag_used"], errors="coerce").fillna(0).astype(int) == args.dry_rain_flag_value].copy()

    # Only dry compounds
    pit = pit[pit["tyre_compound_after"].isin(DRY_CLASSES)].copy()

    # Remove noisy non strategic rows
    if args.exclude_abnormal_long_stops == 1 and "is_abnormally_long_stop" in pit.columns:
        pit = pit[pd.to_numeric(pit["is_abnormally_long_stop"], errors="coerce").fillna(0).astype(int) == 0].copy()

    if args.exclude_lap1_stops == 1 and "lap1_stop_flag" in pit.columns:
        pit = pit[pd.to_numeric(pit["lap1_stop_flag"], errors="coerce").fillna(0).astype(int) == 0].copy()

    if len(pit) < 300:
        raise ValueError(f"Too few dry strategic pit rows after filtering: {len(pit)}")

    cat_cols, num_cols = get_feature_lists(pit)
    use_cols = list(dict.fromkeys(["race_id", "driver_id", "lap_number"] + cat_cols + num_cols + ["tyre_compound_after"]))
    pit = pit[use_cols].copy()

    train_df = pit[pit["race_id"].isin(splits["train_races"])].copy()
    val_df = pit[pit["race_id"].isin(splits["val_races"])].copy()
    test_df = pit[pit["race_id"].isin(splits["test_races"])].copy()

    classes = [c for c in DRY_CLASSES if c in set(pit["tyre_compound_after"].astype(str))]
    if len(classes) < 3:
        raise ValueError(f"Expected 3 dry classes but found: {classes}")

    class_to_idx = {c: i for i, c in enumerate(classes)}

    X_train_raw = train_df[cat_cols + num_cols].copy()
    X_val_raw = val_df[cat_cols + num_cols].copy()
    X_test_raw = test_df[cat_cols + num_cols].copy()

    for c in cat_cols:
        X_train_raw[c] = X_train_raw[c].astype(str)
        X_val_raw[c] = X_val_raw[c].astype(str)
        X_test_raw[c] = X_test_raw[c].astype(str)

    for c in num_cols:
        X_train_raw[c] = pd.to_numeric(X_train_raw[c], errors="coerce")
        X_val_raw[c] = pd.to_numeric(X_val_raw[c], errors="coerce")
        X_test_raw[c] = pd.to_numeric(X_test_raw[c], errors="coerce")

    y_train = np.array([class_to_idx[c] for c in train_df["tyre_compound_after"].astype(str)], dtype=int)
    y_val = np.array([class_to_idx[c] for c in val_df["tyre_compound_after"].astype(str)], dtype=int)
    y_test = np.array([class_to_idx[c] for c in test_df["tyre_compound_after"].astype(str)], dtype=int)

    pre = build_preprocessor(cat_cols, num_cols)
    X_train = pre.fit_transform(X_train_raw)
    X_val = pre.transform(X_val_raw)
    X_test = pre.transform(X_test_raw)

    class_weights = compute_class_weights(y_train, num_class=len(classes))
    sample_weight_train = class_weights[y_train]
    sample_weight_val = class_weights[y_val]

    if args.optuna_trials > 0 and optuna is not None:
        best = optuna_tune_multiclass(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            num_class=len(classes),
            seed=args.seed,
            trials=args.optuna_trials,
            sample_weight_train=sample_weight_train,
            sample_weight_val=sample_weight_val,
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
        random_state=args.seed,
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

    try:
        model.fit(
            X_train,
            y_train,
            sample_weight=sample_weight_train,
            eval_set=[(X_val, y_val)],
            sample_weight_eval_set=[sample_weight_val],
            verbose=False,
            early_stopping_rounds=35,
        )
    except TypeError:
        fit_xgb_with_early_stopping(model, X_train, y_train, X_val, y_val, rounds=35)

    p_train = model.predict_proba(X_train)
    p_val = model.predict_proba(X_val)
    p_test = model.predict_proba(X_test)

    eps = 1e-12
    train_logloss = float(-np.mean(np.log(np.clip(p_train[np.arange(len(y_train)), y_train], eps, 1.0))))
    val_logloss = float(-np.mean(np.log(np.clip(p_val[np.arange(len(y_val)), y_val], eps, 1.0))))
    test_logloss = float(-np.mean(np.log(np.clip(p_test[np.arange(len(y_test)), y_test], eps, 1.0))))

    pred_test_top1 = np.argmax(p_test, axis=1)
    top1_acc = float(np.mean(pred_test_top1 == y_test))

    top2_idx = np.argsort(-p_test, axis=1)[:, :2]
    top2_acc = float(np.mean([y_test[i] in top2_idx[i] for i in range(len(y_test))]))

    cm = confusion_matrix(y_test, pred_test_top1, labels=list(range(len(classes))))

    metrics = {
        "timestamp": ts,
        "mode": "dry_compound_choice_conditional_on_pit",
        "classes": classes,
        "n_rows_pit_used": int(len(pit)),
        "n_train": int(len(train_df)),
        "n_val": int(len(val_df)),
        "n_test": int(len(test_df)),
        "exclude_abnormal_long_stops": bool(args.exclude_abnormal_long_stops == 1),
        "exclude_lap1_stops": bool(args.exclude_lap1_stops == 1),
        "cat_cols": cat_cols,
        "num_cols": num_cols,
        "best_params": best,
        "train_logloss": train_logloss,
        "val_logloss": val_logloss,
        "test_logloss": test_logloss,
        "top1_accuracy_test": top1_acc,
        "top2_accuracy_test": top2_acc,
        "class_weights": {classes[i]: float(class_weights[i]) for i in range(len(classes))},
        "confusion_matrix_test": {
            "labels": classes,
            "matrix": cm.tolist(),
        },
    }

    metrics_path = os.path.join(args.outputs_dir, f"model3b_dry_metrics_{ts}.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    pre_path = os.path.join(args.models_dir, f"model3b_dry_preprocessor_{ts}.pkl")
    joblib.dump(pre, pre_path)

    model_path = os.path.join(args.models_dir, f"model3b_dry_xgb_{ts}.pkl")
    joblib.dump(model, model_path)

    classes_path = os.path.join(args.models_dir, f"model3b_dry_classes_{ts}.json")
    with open(classes_path, "w") as f:
        json.dump({"classes": classes}, f, indent=2)

    # Predict on all rows
    X_all_raw = df[cat_cols + num_cols].copy()
    for c in cat_cols:
        X_all_raw[c] = X_all_raw[c].astype(str)
    for c in num_cols:
        X_all_raw[c] = pd.to_numeric(X_all_raw[c], errors="coerce")

    X_all = pre.transform(X_all_raw)
    p_all = model.predict_proba(X_all)

    probs = df[["race_id", "driver_id", "lap_number"]].copy()

    # initialize all compound columns
    probs["p_compound_SOFT"] = 0.0
    probs["p_compound_MEDIUM"] = 0.0
    probs["p_compound_HARD"] = 0.0
    probs["p_compound_INTERMEDIATE"] = 0.0
    probs["p_compound_WET"] = 0.0

    for i, cls in enumerate(classes):
        probs[f"p_compound_{cls}"] = p_all[:, i].astype(float)

    # Dry model only. Wet classes kept at zero.
    probs["compound_choice_argmax"] = np.array(classes, dtype=object)[np.argmax(p_all, axis=1)]

    # Drop old compound columns from original df
    out = df.copy()
    old_cols = [c for c in out.columns if c.startswith("p_compound_") or c == "compound_choice_argmax"]
    if old_cols:
        out = out.drop(columns=old_cols)

    out = out.merge(probs, on=["race_id", "driver_id", "lap_number"], how="left")

    out_path = os.path.join(args.outputs_dir, args.merge_output_name)
    out.to_csv(out_path, index=False)

    print("DONE fixed Model 3B dry compound retrain")
    print(f"Metrics: {metrics_path}")
    print(f"Saved: {pre_path}")
    print(f"Saved: {model_path}")
    print(f"Saved: {classes_path}")
    print(f"Merged output: {out_path}")


if __name__ == "__main__":
    main()