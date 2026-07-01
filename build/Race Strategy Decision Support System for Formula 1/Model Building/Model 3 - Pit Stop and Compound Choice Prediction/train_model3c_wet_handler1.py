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


def normalize_series(s: pd.Series, lo: float = None, hi: float = None) -> pd.Series:
    x = pd.to_numeric(s, errors="coerce")
    if lo is None:
        lo = float(np.nanpercentile(x, 5)) if x.notna().any() else 0.0
    if hi is None:
        hi = float(np.nanpercentile(x, 95)) if x.notna().any() else 1.0
    if not np.isfinite(lo):
        lo = 0.0
    if not np.isfinite(hi) or hi <= lo:
        hi = lo + 1.0
    z = (x - lo) / (hi - lo)
    return z.clip(0.0, 1.0).fillna(0.0)


def safe_text_col(df: pd.DataFrame, col: str, default: str = "") -> pd.Series:
    if col not in df.columns:
        return pd.Series([default] * len(df), index=df.index)
    return df[col].astype(str).fillna(default)


def safe_num_col(df: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
    if col not in df.columns:
        return pd.Series([default] * len(df), index=df.index, dtype=float)
    return pd.to_numeric(df[col], errors="coerce").fillna(default)


def build_wetness_score(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    rain_flag = safe_num_col(out, "rain_flag_used", 0.0).clip(0.0, 1.0)
    weather_txt = safe_text_col(out, "race_level_weather_condition", "").str.lower()
    tyre_now = safe_text_col(out, "tyre_compound", "").str.upper()

    track_temp = safe_num_col(out, "track_temperature_used", 30.0)
    air_temp = safe_num_col(out, "air_temperature_used", 20.0)
    laps_remaining = safe_num_col(out, "laps_remaining", 0.0)
    sc_flag = safe_num_col(out, "is_under_safety_car", 0.0).clip(0.0, 1.0)
    vsc_flag = safe_num_col(out, "is_under_virtual_safety_car", 0.0).clip(0.0, 1.0)

    # Weather text signals
    weather_wet = weather_txt.str.contains("wet|rain|shower|storm", regex=True).astype(float)
    weather_damp = weather_txt.str.contains("damp|mixed|changeable", regex=True).astype(float)

    # Temperature effect
    # colder conditions sustain WET tyres more easily
    cold_track = (1.0 - normalize_series(track_temp, lo=10.0, hi=45.0)).clip(0.0, 1.0)
    cold_air = (1.0 - normalize_series(air_temp, lo=5.0, hi=30.0)).clip(0.0, 1.0)

    # Current tyre signal
    on_wet = (tyre_now == "WET").astype(float)
    on_inter = (tyre_now == "INTERMEDIATE").astype(float)
    on_dry = tyre_now.isin(["SOFT", "MEDIUM", "HARD"]).astype(float)

    # Neutralization can accompany very wet running, but weak signal only
    neutralization = np.maximum(sc_flag, vsc_flag)

    # Wetness score
    wetness_score = (
        0.40 * rain_flag
        + 0.25 * weather_wet
        + 0.10 * weather_damp
        + 0.08 * cold_track
        + 0.05 * cold_air
        + 0.07 * on_wet
        + 0.03 * neutralization
        + 0.02 * (laps_remaining > 10).astype(float)
    ).clip(0.0, 1.0)

    # Crossover score
    crossover_score = (
        0.35 * rain_flag
        + 0.25 * weather_damp
        + 0.10 * weather_wet
        + 0.10 * on_inter
        + 0.10 * on_dry
        + 0.05 * cold_track
        + 0.05 * neutralization
    ).clip(0.0, 1.0)

    out["wetness_score"] = wetness_score
    out["crossover_score"] = crossover_score

    return out


def compute_wet_probabilities(df: pd.DataFrame) -> pd.DataFrame:
    out = build_wetness_score(df)

    rain_flag = safe_num_col(out, "rain_flag_used", 0.0).clip(0.0, 1.0)
    tyre_now = safe_text_col(out, "tyre_compound", "").str.upper()
    weather_txt = safe_text_col(out, "race_level_weather_condition", "").str.lower()

    wetness = out["wetness_score"]
    crossover = out["crossover_score"]

    # Base wet probabilities
    p_wet = np.where(
        wetness >= 0.70,
        0.75 + 0.20 * (wetness - 0.70) / 0.30,
        np.where(
            wetness >= 0.45,
            0.20 + 0.55 * (wetness - 0.45) / 0.25,
            0.02 + 0.18 * wetness / 0.45,
        ),
    )

    p_inter = np.where(
        wetness >= 0.70,
        0.20 - 0.10 * (wetness - 0.70) / 0.30,
        np.where(
            wetness >= 0.35,
            0.45 + 0.35 * (wetness - 0.35) / 0.35,
            0.05 + 0.40 * wetness / 0.35,
        ),
    )

    # If currently on wets and still clearly wet, bias harder toward WET
    p_wet += 0.10 * ((tyre_now == "WET") & (wetness >= 0.55)).astype(float)

    # If currently on intermediates in crossover, bias toward INTERMEDIATE
    p_inter += 0.10 * ((tyre_now == "INTERMEDIATE") & (crossover >= 0.40)).astype(float)

    # If weather text strongly says damp or mixed, pull toward INTERMEDIATE
    p_inter += 0.08 * weather_txt.str.contains("damp|mixed|changeable", regex=True).astype(float)

    # If no rain signal and no wet text, keep wet tyres near zero
    no_wet_signal = ((rain_flag <= 0) & (~weather_txt.str.contains("wet|rain|shower|storm|damp|mixed", regex=True))).astype(float)
    p_wet *= (1.0 - 0.95 * no_wet_signal)
    p_inter *= (1.0 - 0.90 * no_wet_signal)

    p_wet = np.clip(p_wet, 0.0, 1.0)
    p_inter = np.clip(p_inter, 0.0, 1.0)

    wet_total = p_wet + p_inter
    wet_total = np.where(wet_total > 1.0, wet_total, 1.0)

    p_wet = p_wet / wet_total
    p_inter = p_inter / wet_total

    out["p_compound_INTERMEDIATE_wetlogic"] = p_inter
    out["p_compound_WET_wetlogic"] = p_wet

    # Choose wet argmax only within wet regime
    out["wet_logic_argmax"] = np.where(
        p_wet >= p_inter,
        "WET",
        "INTERMEDIATE",
    )

    return out


def merge_with_existing_compound_probs(df: pd.DataFrame) -> pd.DataFrame:
    out = compute_wet_probabilities(df)

    # Ensure base compound columns exist
    for c in ["p_compound_SOFT", "p_compound_MEDIUM", "p_compound_HARD", "p_compound_INTERMEDIATE", "p_compound_WET"]:
        if c not in out.columns:
            out[c] = 0.0

    rain_flag = safe_num_col(out, "rain_flag_used", 0.0).clip(0.0, 1.0)
    weather_txt = safe_text_col(out, "race_level_weather_condition", "").str.lower()

    wet_regime = (
        (rain_flag > 0)
        | weather_txt.str.contains("wet|rain|shower|storm|damp|mixed", regex=True)
    ).astype(float)

    p_inter_wet = out["p_compound_INTERMEDIATE_wetlogic"]
    p_wet_wet = out["p_compound_WET_wetlogic"]

    dry_sum = (
        safe_num_col(out, "p_compound_SOFT", 0.0)
        + safe_num_col(out, "p_compound_MEDIUM", 0.0)
        + safe_num_col(out, "p_compound_HARD", 0.0)
    )

    # In wet regime, override dry probabilities heavily
    out["p_compound_INTERMEDIATE"] = np.where(wet_regime > 0, p_inter_wet, safe_num_col(out, "p_compound_INTERMEDIATE", 0.0))
    out["p_compound_WET"] = np.where(wet_regime > 0, p_wet_wet, safe_num_col(out, "p_compound_WET", 0.0))

    out["p_compound_SOFT"] = np.where(wet_regime > 0, 0.0, safe_num_col(out, "p_compound_SOFT", 0.0))
    out["p_compound_MEDIUM"] = np.where(wet_regime > 0, 0.0, safe_num_col(out, "p_compound_MEDIUM", 0.0))
    out["p_compound_HARD"] = np.where(wet_regime > 0, 0.0, safe_num_col(out, "p_compound_HARD", 0.0))

    # Final argmax over all compounds
    prob_cols = [
        "p_compound_SOFT",
        "p_compound_MEDIUM",
        "p_compound_HARD",
        "p_compound_INTERMEDIATE",
        "p_compound_WET",
    ]
    probs_mat = out[prob_cols].to_numpy(dtype=float)
    argmax_idx = np.argmax(probs_mat, axis=1)
    labels = np.array(["SOFT", "MEDIUM", "HARD", "INTERMEDIATE", "WET"], dtype=object)
    out["compound_choice_argmax"] = labels[argmax_idx]

    out["wet_handler_used_flag"] = wet_regime.astype(int)

    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_path", required=True, help="Dataset after Model 3A + dry Model 3B")
    ap.add_argument("--outputs_dir", default="outputs")
    ap.add_argument("--merge_output_name", default="lap_level_with_model1_model2_model3_final.csv")
    args = ap.parse_args()

    ensure_dir(args.outputs_dir)
    ts = now_ts()

    df = pd.read_csv(args.input_path, low_memory=False)
    pk = ["race_id", "driver_id", "lap_number"]
    if df.duplicated(pk).any():
        raise ValueError("Primary key duplicates found")

    out = merge_with_existing_compound_probs(df)

    metrics = {
        "timestamp": ts,
        "mode": "wet_compound_handler_rule_based",
        "input_path": args.input_path,
        "n_rows": int(len(out)),
        "n_wet_handler_used": int(out["wet_handler_used_flag"].sum()),
        "mean_p_intermediate_when_used": float(out.loc[out["wet_handler_used_flag"] == 1, "p_compound_INTERMEDIATE"].mean()) if (out["wet_handler_used_flag"] == 1).any() else 0.0,
        "mean_p_wet_when_used": float(out.loc[out["wet_handler_used_flag"] == 1, "p_compound_WET"].mean()) if (out["wet_handler_used_flag"] == 1).any() else 0.0,
    }

    metrics_path = os.path.join(args.outputs_dir, f"model3c_wet_handler_metrics_{ts}.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    out_path = os.path.join(args.outputs_dir, args.merge_output_name)
    out.to_csv(out_path, index=False)

    print("DONE MODEL 3C wet compound handler")
    print(f"Metrics: {metrics_path}")
    print(f"Merged output: {out_path}")


if __name__ == "__main__":
    main()