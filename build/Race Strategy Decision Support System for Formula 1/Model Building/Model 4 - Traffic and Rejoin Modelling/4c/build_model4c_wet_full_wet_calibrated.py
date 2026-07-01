from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd


def now_ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def safe_numeric(series: pd.Series, fill: float = 0.0) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").fillna(fill)


def get_col(df: pd.DataFrame, col: str, fill: float = 0.0) -> np.ndarray:
    if col in df.columns:
        return safe_numeric(df[col], fill=fill).to_numpy(dtype=float)
    return np.full(len(df), fill, dtype=float)


def score_regression(y_true: np.ndarray, y_hat: np.ndarray) -> dict:
    err = y_hat - y_true
    ae = np.abs(err)
    mse = np.mean((err) ** 2) if len(err) else np.nan
    rmse = float(np.sqrt(mse)) if len(err) else float("nan")

    ss_res = float(np.sum((y_true - y_hat) ** 2)) if len(y_true) else float("nan")
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2)) if len(y_true) else float("nan")
    r2 = float(1.0 - ss_res / ss_tot) if len(y_true) > 1 and ss_tot > 0 else float("nan")

    return {
        "n": int(len(y_true)),
        "target_mean": float(np.mean(y_true)) if len(y_true) else float("nan"),
        "target_std": float(np.std(y_true)) if len(y_true) else float("nan"),
        "mae": float(np.mean(ae)) if len(ae) else float("nan"),
        "rmse": rmse,
        "r2": r2,
        "mean_error": float(np.mean(err)) if len(err) else float("nan"),
        "median_error": float(np.median(err)) if len(err) else float("nan"),
        "median_abs_error": float(np.median(ae)) if len(ae) else float("nan"),
    }


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    out["pred_rejoin_position_delta_nextlap_num"] = get_col(out, "pred_rejoin_position_delta_nextlap", 0.0)
    out["p_rejoin_clean_air_nextlap_num"] = get_col(out, "p_rejoin_clean_air_nextlap", 0.5)
    out["local_pack_density_num"] = get_col(out, "local_pack_density", 0.0)
    out["traffic_window_risk_num"] = get_col(out, "traffic_window_risk", 0.0)
    out["pit_lane_time_loss_num"] = get_col(out, "pit_lane_time_loss", 0.0)
    out["pit_lane_time_sec_num"] = get_col(out, "pit_lane_time_sec", 0.0)
    out["stationary_time_sec_num"] = get_col(out, "stationary_time_sec", 0.0)
    out["gap_ahead_num"] = get_col(out, "gap_ahead", 0.0)
    out["gap_behind_num"] = get_col(out, "gap_behind", 0.0)
    out["clean_air_flag_num"] = get_col(out, "clean_air_flag", 0.0)
    out["track_temperature_used_num"] = get_col(out, "track_temperature_used", 0.0)
    out["rain_flag_used_num"] = get_col(out, "rain_flag_used", 1.0)

    out["stationary_plus_pitlane_sec_num"] = out["pit_lane_time_sec_num"] + out["stationary_time_sec_num"]
    out["rejoin_traffic_pressure_combo_num"] = (
        (1.0 - np.clip(out["p_rejoin_clean_air_nextlap_num"], 0.0, 1.0))
        * np.maximum(out["pred_rejoin_position_delta_nextlap_num"], 0.0)
        * (1.0 + out["local_pack_density_num"])
    )
    out["gap_balance_num"] = out["gap_ahead_num"] - out["gap_behind_num"]

    return out


def fit_linear_calibration(df: pd.DataFrame, target_col: str) -> dict:
    y = safe_numeric(df[target_col]).to_numpy(dtype=float)

    x_cols = [
        "pred_rejoin_position_delta_nextlap_num",
        "p_rejoin_clean_air_nextlap_num",
        "local_pack_density_num",
        "traffic_window_risk_num",
        "pit_lane_time_loss_num",
        "stationary_plus_pitlane_sec_num",
        "gap_balance_num",
        "track_temperature_used_num",
    ]

    X = df[x_cols].to_numpy(dtype=float)

    x_mean = X.mean(axis=0) if len(X) else np.zeros(len(x_cols))
    x_std = X.std(axis=0)
    x_std = np.where(x_std <= 1e-12, 1.0, x_std)

    Xs = (X - x_mean) / x_std

    # Strong ridge because n is tiny
    alpha = 8.0
    X_design = np.column_stack([np.ones(len(Xs)), Xs])
    I = np.eye(X_design.shape[1])
    I[0, 0] = 0.0

    beta = np.linalg.solve(X_design.T @ X_design + alpha * I, X_design.T @ y)

    intercept = float(beta[0])
    coefs = beta[1:]

    raw_pred = intercept + Xs @ coefs

    # Blend with robust central estimate because n=4 is extremely small
    base_median = float(np.median(y))
    blend = 0.65
    pred = blend * raw_pred + (1.0 - blend) * base_median

    calibration = {
        "feature_names": x_cols,
        "x_mean": x_mean.tolist(),
        "x_std": x_std.tolist(),
        "intercept": intercept,
        "coefs": coefs.tolist(),
        "alpha": alpha,
        "blend_with_median": blend,
        "base_median": base_median,
        "target_summary": {
            "n": int(len(y)),
            "mean": float(np.mean(y)),
            "std": float(np.std(y)),
            "min": float(np.min(y)),
            "p25": float(np.quantile(y, 0.25)),
            "p50": float(np.quantile(y, 0.50)),
            "p75": float(np.quantile(y, 0.75)),
            "max": float(np.max(y)),
        },
    }

    return calibration


def predict_with_calibration(df: pd.DataFrame, calibration: dict, min_target: float, max_target: float) -> np.ndarray:
    x_cols = calibration["feature_names"]
    X = df[x_cols].to_numpy(dtype=float)
    x_mean = np.array(calibration["x_mean"], dtype=float)
    x_std = np.array(calibration["x_std"], dtype=float)
    intercept = float(calibration["intercept"])
    coefs = np.array(calibration["coefs"], dtype=float)
    blend = float(calibration["blend_with_median"])
    base_median = float(calibration["base_median"])

    Xs = (X - x_mean) / x_std
    raw_pred = intercept + Xs @ coefs
    pred = blend * raw_pred + (1.0 - blend) * base_median
    pred = np.clip(pred, min_target, max_target)
    return pred.astype(float)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_path", required=True, help="model4c_wet_full_wet_table_*.csv")
    ap.add_argument("--outputs_dir", default="outputs")
    ap.add_argument("--models_dir", default="models")
    ap.add_argument("--min_target", type=float, default=0.0)
    ap.add_argument("--max_target", type=float, default=120.0)
    ap.add_argument("--scored_output_name", default="model4c_wet_full_wet_calibrated_scored.csv")
    args = ap.parse_args()

    ensure_dir(args.outputs_dir)
    ensure_dir(args.models_dir)

    ts = now_ts()
    df = pd.read_csv(args.input_path, low_memory=False)

    required_cols = ["race_id", "driver_id", "lap_number", "y_rejoin_penalty_sec_nextlap"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df["y_rejoin_penalty_sec_nextlap"] = safe_numeric(df["y_rejoin_penalty_sec_nextlap"], fill=np.nan)
    df = df[df["y_rejoin_penalty_sec_nextlap"].notna()].copy()
    df = df[
        (df["y_rejoin_penalty_sec_nextlap"] >= args.min_target)
        & (df["y_rejoin_penalty_sec_nextlap"] <= args.max_target)
    ].copy()

    if len(df) == 0:
        raise ValueError("No FULL_WET rows left after filtering")

    df = build_features(df)

    calibration = fit_linear_calibration(df, "y_rejoin_penalty_sec_nextlap")
    pred = predict_with_calibration(df, calibration, args.min_target, args.max_target)

    metrics = {
        "target": "y_rejoin_penalty_sec_nextlap",
        "mode": "wet_full_wet_calibrated",
        "filters": {
            "min_target": args.min_target,
            "max_target": args.max_target,
            "regime": "FULL_WET",
        },
        "calibration": calibration,
        "in_sample_metrics": score_regression(
            df["y_rejoin_penalty_sec_nextlap"].to_numpy(dtype=float),
            pred,
        ),
        "rows": int(len(df)),
        "notes": [
            "This is a calibrated formula branch, not a supervised ML branch, because FULL_WET sample size is too small for reliable training.",
            "Prediction is based on a strongly regularized linear calibration blended with the observed FULL_WET median.",
            "This branch should be routed only when tyre_compound_after == WET.",
        ],
    }

    base = f"model4c_wet_full_wet_calibrated_{ts}"

    metrics_path = os.path.join(args.outputs_dir, f"{base}_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    model_path = os.path.join(args.models_dir, f"{base}.json")
    with open(model_path, "w") as f:
        json.dump(calibration, f, indent=2)

    config_path = os.path.join(args.outputs_dir, f"{base}_config.json")
    with open(config_path, "w") as f:
        json.dump(
            {
                "input_path": args.input_path,
                "min_target": args.min_target,
                "max_target": args.max_target,
                "scored_output_name": args.scored_output_name,
                "regime": "FULL_WET",
            },
            f,
            indent=2,
        )

    scored = df.copy()
    scored["pred_rejoin_penalty_sec_nextlap"] = pred
    scored_path = os.path.join(args.outputs_dir, args.scored_output_name)
    scored.to_csv(scored_path, index=False)

    pred_export = df[["race_id", "driver_id", "lap_number", "y_rejoin_penalty_sec_nextlap"]].copy()
    pred_export["pred_rejoin_penalty_sec_nextlap"] = pred
    pred_path = os.path.join(args.outputs_dir, f"{base}_all_predictions.csv")
    pred_export.to_csv(pred_path, index=False)

    print("DONE MODEL 4C wet FULL_WET calibrated branch")
    print(f"Model JSON: {model_path}")
    print(f"Metrics: {metrics_path}")
    print(f"Config: {config_path}")
    print(f"Scored output: {scored_path}")
    print(f"All predictions: {pred_path}")


if __name__ == "__main__":
    main()