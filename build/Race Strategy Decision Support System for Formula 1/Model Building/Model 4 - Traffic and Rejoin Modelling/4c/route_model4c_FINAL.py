from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from typing import Dict, List, Tuple

import joblib
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


def add_common_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    pit_lane_time_loss = get_col(out, "pit_lane_time_loss", 0.0)
    overtaking_difficulty_index = get_col(out, "overtaking_difficulty_index", 0.0)
    p_rejoin_clean_air_nextlap = get_col(out, "p_rejoin_clean_air_nextlap", 0.0)
    pred_rejoin_position_delta_nextlap = get_col(out, "pred_rejoin_position_delta_nextlap", 0.0)
    track_temperature_used = get_col(out, "track_temperature_used", 0.0)
    rain_flag_used = get_col(out, "rain_flag_used", 0.0)
    tyre_age = get_col(out, "tyre_age", 0.0)
    pit_stop_number = get_col(out, "pit_stop_number", 0.0)
    pit_to_soft_flag = get_col(out, "pit_to_soft_flag", 0.0)
    pit_to_hard_flag = get_col(out, "pit_to_hard_flag", 0.0)
    pit_to_medium_flag = get_col(out, "pit_to_medium_flag", 0.0)
    pit_to_intermediate_flag = get_col(out, "pit_to_intermediate_flag", 0.0)
    pit_to_wet_flag = get_col(out, "pit_to_wet_flag", 0.0)
    stationary_time_sec = get_col(out, "stationary_time_sec", 0.0)
    pit_lane_time_sec = get_col(out, "pit_lane_time_sec", 0.0)
    gap_ahead = get_col(out, "gap_ahead", 0.0)
    gap_behind = get_col(out, "gap_behind", 0.0)
    clean_air_flag = get_col(out, "clean_air_flag", 0.0)
    local_pack_density = get_col(out, "local_pack_density", 0.0)
    traffic_window_risk = get_col(out, "traffic_window_risk", 0.0)

    out["is_out_lap_nextlap_proxy"] = 1.0
    out["pit_lane_time_loss_x_overtaking_difficulty"] = pit_lane_time_loss * overtaking_difficulty_index
    out["p_rejoin_clean_air_x_rejoin_delta"] = p_rejoin_clean_air_nextlap * pred_rejoin_position_delta_nextlap
    out["track_temp_x_pit_to_soft"] = track_temperature_used * pit_to_soft_flag
    out["track_temp_x_pit_to_hard"] = track_temperature_used * pit_to_hard_flag
    out["track_temp_x_pit_to_medium"] = track_temperature_used * pit_to_medium_flag
    out["track_temp_x_pit_to_intermediate"] = track_temperature_used * pit_to_intermediate_flag
    out["track_temp_x_pit_to_wet"] = track_temperature_used * pit_to_wet_flag
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

    y_penalty = safe_numeric(out["y_rejoin_penalty_sec_nextlap"], fill=0.0) if "y_rejoin_penalty_sec_nextlap" in out.columns else pd.Series(0.0, index=out.index)
    out["tail_low_flag"] = (y_penalty <= 15.0).astype(float)
    out["tail_high_flag"] = (y_penalty > 25.0).astype(float)

    return out


def infer_dry_mask(df: pd.DataFrame) -> np.ndarray:
    tyre_after = df["tyre_compound_after"].astype(str).str.upper() if "tyre_compound_after" in df.columns else pd.Series("", index=df.index)
    race_weather = df["race_level_weather_condition"].astype(str).str.lower() if "race_level_weather_condition" in df.columns else pd.Series("", index=df.index)
    rain_flag = safe_numeric(df["rain_flag_used"], fill=0.0) if "rain_flag_used" in df.columns else pd.Series(0.0, index=df.index)

    wet_mask = (
        tyre_after.isin(["INTERMEDIATE", "WET"])
        | race_weather.eq("rain")
        | (rain_flag > 0)
    )
    return (~wet_mask).to_numpy()


def infer_intermediate_mask(df: pd.DataFrame) -> np.ndarray:
    if "tyre_compound_after" not in df.columns:
        return np.zeros(len(df), dtype=bool)
    return df["tyre_compound_after"].astype(str).str.upper().eq("INTERMEDIATE").to_numpy()


def infer_full_wet_mask(df: pd.DataFrame) -> np.ndarray:
    if "tyre_compound_after" not in df.columns:
        return np.zeros(len(df), dtype=bool)
    return df["tyre_compound_after"].astype(str).str.upper().eq("WET").to_numpy()


def infer_rain_other_mask(df: pd.DataFrame) -> np.ndarray:
    dry_mask = infer_dry_mask(df)
    intermediate_mask = infer_intermediate_mask(df)
    full_wet_mask = infer_full_wet_mask(df)
    return ~(dry_mask | intermediate_mask | full_wet_mask)


def predict_sklearn_pipeline(pipe, X: pd.DataFrame) -> np.ndarray:
    return pipe.predict(X)


def predict_xgb_branch(preprocessor, model, X: pd.DataFrame) -> np.ndarray:
    X_t = preprocessor.transform(X)
    return model.predict(X_t)


def predict_xgb_with_optional_inverse(preprocessor, model, X: pd.DataFrame, target_meta: Dict | None = None) -> np.ndarray:
    pred = predict_xgb_branch(preprocessor, model, X)
    if target_meta is not None:
        if target_meta.get("log_target", False):
            pred = np.expm1(pred)
        lower = float(target_meta.get("clip_lower", 0.0))
        upper = target_meta.get("applied_clip_upper", None)
        pred = np.maximum(pred, lower)
        if upper is not None:
            pred = np.minimum(pred, float(upper))
    return np.asarray(pred, dtype=float)


def predict_full_wet_calibrated(df: pd.DataFrame, calibration: Dict, min_target: float, max_target: float) -> np.ndarray:
    work = df.copy()

    def ensure_num(col: str, fill: float = 0.0) -> np.ndarray:
        if col in work.columns:
            return pd.to_numeric(work[col], errors="coerce").fillna(fill).to_numpy(dtype=float)
        return np.full(len(work), fill, dtype=float)

    work["pred_rejoin_position_delta_nextlap_num"] = ensure_num("pred_rejoin_position_delta_nextlap", 0.0)
    work["p_rejoin_clean_air_nextlap_num"] = ensure_num("p_rejoin_clean_air_nextlap", 0.5)
    work["local_pack_density_num"] = ensure_num("local_pack_density", 0.0)
    work["traffic_window_risk_num"] = ensure_num("traffic_window_risk", 0.0)
    work["pit_lane_time_loss_num"] = ensure_num("pit_lane_time_loss", 0.0)
    work["pit_lane_time_sec_num"] = ensure_num("pit_lane_time_sec", 0.0)
    work["stationary_time_sec_num"] = ensure_num("stationary_time_sec", 0.0)
    work["gap_ahead_num"] = ensure_num("gap_ahead", 0.0)
    work["gap_behind_num"] = ensure_num("gap_behind", 0.0)
    work["track_temperature_used_num"] = ensure_num("track_temperature_used", 0.0)

    work["stationary_plus_pitlane_sec_num"] = work["pit_lane_time_sec_num"] + work["stationary_time_sec_num"]
    work["gap_balance_num"] = work["gap_ahead_num"] - work["gap_behind_num"]

    x_cols = calibration["feature_names"]
    X = work[x_cols].to_numpy(dtype=float)

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
    ap.add_argument("--input_path", required=True)
    ap.add_argument("--outputs_dir", default="outputs")

    ap.add_argument("--dry_model_type", choices=["xgb", "ridge", "huber"], required=True)
    ap.add_argument("--dry_model_path", required=True)
    ap.add_argument("--dry_preprocessor_path", default="")
    ap.add_argument("--dry_target_meta_json", default="")

    ap.add_argument("--wet_intermediate_model_type", choices=["xgb", "ridge", "huber"], required=True)
    ap.add_argument("--wet_intermediate_model_path", required=True)
    ap.add_argument("--wet_intermediate_preprocessor_path", default="")

    ap.add_argument("--wet_rain_other_model_type", choices=["xgb", "ridge", "huber"], required=True)
    ap.add_argument("--wet_rain_other_model_path", required=True)
    ap.add_argument("--wet_rain_other_preprocessor_path", default="")
    ap.add_argument("--wet_rain_other_target_meta_json", default="")

    ap.add_argument("--wet_full_wet_calibration_json", required=True)

    ap.add_argument("--min_target", type=float, default=0.0)
    ap.add_argument("--max_target", type=float, default=120.0)
    ap.add_argument("--scored_output_name", default="model4c_final_routed_scored.csv")
    args = ap.parse_args()

    ensure_dir(args.outputs_dir)
    ts = now_ts()

    df = pd.read_csv(args.input_path, low_memory=False)
    df = add_common_engineered_features(df)

    n = len(df)
    pred_final = np.full(n, np.nan, dtype=float)
    route_label = np.full(n, "", dtype=object)

    dry_mask = infer_dry_mask(df)
    intermediate_mask = infer_intermediate_mask(df)
    full_wet_mask = infer_full_wet_mask(df)
    rain_other_mask = infer_rain_other_mask(df)

    # DRY
    if dry_mask.sum() > 0:
        X_sub = df.loc[dry_mask].copy()

        if args.dry_model_type == "xgb":
            pre = joblib.load(args.dry_preprocessor_path)
            model = joblib.load(args.dry_model_path)
            target_meta = None
            if args.dry_target_meta_json:
                with open(args.dry_target_meta_json, "r") as f:
                    maybe_metrics = json.load(f)
                target_meta = maybe_metrics.get("target_meta", maybe_metrics)
            pred = predict_xgb_with_optional_inverse(pre, model, X_sub, target_meta)
        else:
            pipe = joblib.load(args.dry_model_path)
            pred = predict_sklearn_pipeline(pipe, X_sub)

        pred_final[dry_mask] = pred
        route_label[dry_mask] = "dry"

    # WET INTERMEDIATE
    if intermediate_mask.sum() > 0:
        X_sub = df.loc[intermediate_mask].copy()

        if args.wet_intermediate_model_type == "xgb":
            pre = joblib.load(args.wet_intermediate_preprocessor_path)
            model = joblib.load(args.wet_intermediate_model_path)
            pred = predict_xgb_branch(pre, model, X_sub)
        else:
            pipe = joblib.load(args.wet_intermediate_model_path)
            pred = predict_sklearn_pipeline(pipe, X_sub)

        pred_final[intermediate_mask] = pred
        route_label[intermediate_mask] = "wet_intermediate"

       # WET FULL WET
    if full_wet_mask.sum() > 0:
        X_sub = df.loc[full_wet_mask].copy()
        with open(args.wet_full_wet_calibration_json, "r") as f:
            full_wet_json = json.load(f)

        calibration = full_wet_json.get("calibration", full_wet_json)
        pred = predict_full_wet_calibrated(X_sub, calibration, args.min_target, args.max_target)
        pred_final[full_wet_mask] = pred
        route_label[full_wet_mask] = "wet_full_wet_calibrated"
    # WET RAIN OTHER
    if rain_other_mask.sum() > 0:
        X_sub = df.loc[rain_other_mask].copy()

        if args.wet_rain_other_model_type == "xgb":
            pre = joblib.load(args.wet_rain_other_preprocessor_path)
            model = joblib.load(args.wet_rain_other_model_path)
            target_meta = None
            if args.wet_rain_other_target_meta_json:
                with open(args.wet_rain_other_target_meta_json, "r") as f:
                    maybe_metrics = json.load(f)
                target_meta = maybe_metrics.get("target_meta", maybe_metrics)
            pred = predict_xgb_with_optional_inverse(pre, model, X_sub, target_meta)
        else:
            pipe = joblib.load(args.wet_rain_other_model_path)
            pred = predict_sklearn_pipeline(pipe, X_sub)

        pred_final[rain_other_mask] = pred
        route_label[rain_other_mask] = "wet_rain_other_weighted"

    pred_final = np.clip(pred_final, args.min_target, args.max_target)

    routed = df.copy()
    routed["model4c_route"] = route_label
    routed["pred_rejoin_penalty_sec_nextlap"] = pred_final

    route_counts = pd.Series(route_label).value_counts(dropna=False).to_dict()

    metrics = {
        "mode": "model4c_final_router",
        "route_counts": route_counts,
        "filters": {
            "min_target": args.min_target,
            "max_target": args.max_target,
        },
        "branches": {
            "dry_model_type": args.dry_model_type,
            "wet_intermediate_model_type": args.wet_intermediate_model_type,
            "wet_rain_other_model_type": args.wet_rain_other_model_type,
            "wet_full_wet_calibration_json": args.wet_full_wet_calibration_json,
        },
    }

    if "y_rejoin_penalty_sec_nextlap" in routed.columns:
        y_true = pd.to_numeric(routed["y_rejoin_penalty_sec_nextlap"], errors="coerce")
        valid = y_true.notna() & np.isfinite(routed["pred_rejoin_penalty_sec_nextlap"])
        if valid.sum() > 0:
            yt = y_true.loc[valid].to_numpy(dtype=float)
            yp = routed.loc[valid, "pred_rejoin_penalty_sec_nextlap"].to_numpy(dtype=float)

            err = yp - yt
            ae = np.abs(err)
            ss_res = float(np.sum((yt - yp) ** 2))
            ss_tot = float(np.sum((yt - np.mean(yt)) ** 2))
            r2 = float(1.0 - ss_res / ss_tot) if len(yt) > 1 and ss_tot > 0 else float("nan")

            metrics["overall_with_available_truth"] = {
                "n": int(len(yt)),
                "mae": float(np.mean(ae)),
                "rmse": float(np.sqrt(np.mean((yt - yp) ** 2))),
                "r2": r2,
                "mean_error": float(np.mean(err)),
                "median_abs_error": float(np.median(ae)),
            }

            per_route = []
            for route in sorted(pd.unique(routed.loc[valid, "model4c_route"])):
                mask = valid & routed["model4c_route"].eq(route)
                if mask.sum() == 0:
                    continue
                yrt = y_true.loc[mask].to_numpy(dtype=float)
                yrp = routed.loc[mask, "pred_rejoin_penalty_sec_nextlap"].to_numpy(dtype=float)
                route_err = yrp - yrt
                route_ae = np.abs(route_err)
                route_ss_res = float(np.sum((yrt - yrp) ** 2))
                route_ss_tot = float(np.sum((yrt - np.mean(yrt)) ** 2))
                route_r2 = float(1.0 - route_ss_res / route_ss_tot) if len(yrt) > 1 and route_ss_tot > 0 else float("nan")

                per_route.append(
                    {
                        "route": route,
                        "n": int(len(yrt)),
                        "mae": float(np.mean(route_ae)),
                        "rmse": float(np.sqrt(np.mean((yrt - yrp) ** 2))),
                        "r2": route_r2,
                        "mean_error": float(np.mean(route_err)),
                    }
                )
            metrics["per_route_with_available_truth"] = per_route

    base = f"model4c_final_router_{ts}"

    scored_path = os.path.join(args.outputs_dir, args.scored_output_name)
    routed.to_csv(scored_path, index=False)

    metrics_path = os.path.join(args.outputs_dir, f"{base}_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    route_counts_path = os.path.join(args.outputs_dir, f"{base}_route_counts.json")
    with open(route_counts_path, "w") as f:
        json.dump(route_counts, f, indent=2)

    print("DONE MODEL 4C FINAL ROUTER")
    print(f"Scored output: {scored_path}")
    print(f"Metrics: {metrics_path}")
    print(f"Route counts: {route_counts_path}")


if __name__ == "__main__":
    main()