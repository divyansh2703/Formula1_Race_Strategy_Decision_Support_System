from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


EVENT_ACTIVE_COLS = {
    "sc": "is_under_safety_car_event",
    "vsc": "is_under_vsc_event",
    "rf": "is_under_red_flag_event",
}


@dataclass
class SplitIds:
    train_races: List[str]
    val_races: List[str]
    test_races: List[str]


def now_ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def safe_numeric(s: pd.Series, fill: float = 0.0) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").fillna(fill)


def parse_csv_list(s: str) -> List[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


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


def safe_roc_auc(y_true: np.ndarray, p_hat: np.ndarray) -> float:
    try:
        if len(np.unique(y_true)) < 2:
            return float("nan")
        from sklearn.metrics import roc_auc_score
        return float(roc_auc_score(y_true, p_hat))
    except Exception:
        return float("nan")


def safe_pr_auc(y_true: np.ndarray, p_hat: np.ndarray) -> float:
    try:
        if len(np.unique(y_true)) < 2:
            return float("nan")
        from sklearn.metrics import average_precision_score
        return float(average_precision_score(y_true, p_hat))
    except Exception:
        return float("nan")


def safe_brier(y_true: np.ndarray, p_hat: np.ndarray) -> float:
    try:
        if len(np.unique(y_true)) < 2:
            return float("nan")
        from sklearn.metrics import brier_score_loss
        return float(brier_score_loss(y_true, p_hat))
    except Exception:
        return float("nan")


def safe_logloss(y_true: np.ndarray, p_hat: np.ndarray) -> float:
    try:
        if len(np.unique(y_true)) < 2:
            return float("nan")
        from sklearn.metrics import log_loss
        return float(log_loss(y_true, np.clip(p_hat, 1e-6, 1.0 - 1e-6)))
    except Exception:
        return float("nan")


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


def write_model_report_md(report_path: str, title: str, payload: Dict) -> None:
    lines: List[str] = []
    lines.append(f"# {title}")
    lines.append("")
    lines.append(f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
    lines.append("")

    for key, value in payload.items():
        lines.append(f"## {key}")
        lines.append("")
        if isinstance(value, (dict, list)):
            lines.append("```json")
            lines.append(json.dumps(value, indent=2))
            lines.append("```")
        else:
            lines.append(str(value))
        lines.append("")

    with open(report_path, "w") as f:
        f.write("\n".join(lines))


def step_to_cumulative(step_probs: np.ndarray) -> np.ndarray:
    surv = np.cumprod(1.0 - step_probs, axis=1)
    cum = 1.0 - surv
    return np.clip(cum, 0.0, 1.0)


def aggregate_race_lap(master: pd.DataFrame) -> pd.DataFrame:
    df = master.copy()
    df = df.sort_values(["race_id", "driver_id", "lap_number"]).reset_index(drop=True)

    required = ["race_id", "driver_id", "lap_number"]
    for c in required:
        if c not in df.columns:
            raise ValueError(f"Missing required column: {c}")

    numeric_try = [
        "gap_ahead",
        "gap_behind",
        "tyre_age",
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
        "field_size",
        "lap_time",
    ]
    for c in numeric_try:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    for c in EVENT_ACTIVE_COLS.values():
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)

    if "is_under_safety_car" in df.columns:
        df["is_under_safety_car"] = pd.to_numeric(df["is_under_safety_car"], errors="coerce").fillna(0).astype(int)
    if "is_under_virtual_safety_car" in df.columns:
        df["is_under_virtual_safety_car"] = pd.to_numeric(df["is_under_virtual_safety_car"], errors="coerce").fillna(0).astype(int)

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
        c for c in [
            "is_under_safety_car",
            "is_under_virtual_safety_car",
            "is_under_safety_car_event",
            "is_under_vsc_event",
            "is_under_red_flag_event",
        ] if c in df.columns
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
        race_lap[c] = safe_numeric(race_lap[c], fill=0.0)

    race_lap = add_vsc_instability_features(race_lap)
    race_lap = add_vsc_start_flags(race_lap)
    return race_lap


def add_vsc_start_flags(race_lap: pd.DataFrame) -> pd.DataFrame:
    df = race_lap.copy()
    active_col = "is_under_vsc_event"
    if active_col not in df.columns:
        raise ValueError("Missing is_under_vsc_event for VSC prior model")

    cur = safe_numeric(df[active_col], fill=0).astype(int)
    prev = df.groupby("race_id", sort=False)[active_col].shift(1).fillna(0).astype(int)
    df["vsc_start_this_lap"] = ((cur == 1) & (prev == 0)).astype(int)
    return df


def add_vsc_instability_features(race_lap: pd.DataFrame) -> pd.DataFrame:
    df = race_lap.copy()

    def col(name: str) -> pd.Series:
        if name in df.columns:
            return safe_numeric(df[name], fill=0.0)
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

    df["close_pack_x_pitting"] = col("share_close_ahead_1s") * col("share_pitting")
    df["close_pack_x_old_tyres"] = col("share_close_ahead_15s") * col("share_old_tyres_15")
    df["wet_x_instability"] = col("rain_flag_used") * df["vsc_local_instability_index"].fillna(0.0)
    df["pit_surge_x_gap_compression"] = df["delta_share_pitting"].fillna(0.0) * df["gap_compression_index"].fillna(0.0)
    df["pack_pressure_balance"] = col("share_close_ahead_1s") + col("share_close_behind_1s") - col("share_clean_air")

    total = safe_numeric(df.get("total_laps", pd.Series(np.nan, index=df.index)), fill=np.nan).replace(0, np.nan)
    frac = safe_numeric(df.get("lap_number", pd.Series(np.nan, index=df.index)), fill=np.nan) / total
    df["vsc_phase_bucket"] = np.where(frac <= 0.33, "early", np.where(frac <= 0.66, "mid", "late"))
    df["vsc_weather_bucket"] = np.where(col("rain_flag_used") > 0, "wet", "dry")

    return df


def build_train_side_tables(df: pd.DataFrame, train_mask: np.ndarray) -> Dict:
    train_df = df.loc[train_mask].copy()

    global_step_rate = float(train_df["vsc_start_this_lap"].mean()) if len(train_df) else 0.0

    grp1 = (
        train_df.groupby(["circuit_id", "vsc_phase_bucket", "vsc_weather_bucket"], as_index=False)
        .agg(mean1=("vsc_start_this_lap", "mean"), count1=("vsc_start_this_lap", "count"))
    )

    grp2 = (
        train_df.groupby(["circuit_id", "vsc_phase_bucket"], as_index=False)
        .agg(mean2=("vsc_start_this_lap", "mean"), count2=("vsc_start_this_lap", "count"))
    )

    grp3 = (
        train_df.groupby(["circuit_id"], as_index=False)
        .agg(mean3=("vsc_start_this_lap", "mean"), count3=("vsc_start_this_lap", "count"))
    )

    # Quantiles based on train only for stable scaling
    inst = safe_numeric(train_df["vsc_local_instability_index"], fill=0.0)
    q = {
        "inst_q10": float(np.quantile(inst, 0.10)) if len(inst) else 0.0,
        "inst_q50": float(np.quantile(inst, 0.50)) if len(inst) else 0.0,
        "inst_q90": float(np.quantile(inst, 0.90)) if len(inst) else 1.0,
    }

    return {
        "global_step_rate": global_step_rate,
        "grp1": grp1,
        "grp2": grp2,
        "grp3": grp3,
        "quantiles": q,
    }


def apply_hierarchical_prior(df: pd.DataFrame, tables: Dict, alpha_global: float, alpha_circuit: float, alpha_phase: float) -> pd.DataFrame:
    out = df.copy()
    global_rate = float(tables["global_step_rate"])

    out = out.merge(tables["grp1"], on=["circuit_id", "vsc_phase_bucket", "vsc_weather_bucket"], how="left")
    out = out.merge(tables["grp2"], on=["circuit_id", "vsc_phase_bucket"], how="left")
    out = out.merge(tables["grp3"], on=["circuit_id"], how="left")

    p3 = ((out["mean3"].fillna(global_rate) * out["count3"].fillna(0.0)) + alpha_global * global_rate) / (out["count3"].fillna(0.0) + alpha_global)
    p2 = ((out["mean2"].fillna(p3) * out["count2"].fillna(0.0)) + alpha_circuit * p3) / (out["count2"].fillna(0.0) + alpha_circuit)
    p1 = ((out["mean1"].fillna(p2) * out["count1"].fillna(0.0)) + alpha_phase * p2) / (out["count1"].fillna(0.0) + alpha_phase)

    out["vsc_hier_prior_step"] = np.clip(p1, 0.0, 1.0)
    return out


def build_instability_multiplier(
    df: pd.DataFrame,
    q: Dict,
    wet_boost: float,
    instability_scale: float,
    pitting_scale: float,
    compression_scale: float,
    track_prior_scale: float,
) -> np.ndarray:
    inst = safe_numeric(df["vsc_local_instability_index"], fill=0.0).to_numpy(dtype=float)
    q10 = float(q["inst_q10"])
    q50 = float(q["inst_q50"])
    q90 = float(q["inst_q90"])

    denom = max(q90 - q10, 1e-6)
    inst_scaled = (inst - q50) / denom
    inst_scaled = np.clip(inst_scaled, -1.5, 2.0)

    wet_flag = np.where(safe_numeric(df.get("rain_flag_used", pd.Series(0.0, index=df.index)), fill=0.0).to_numpy(dtype=float) > 0, wet_boost, 1.0)

    share_pitting = safe_numeric(df.get("share_pitting", pd.Series(0.0, index=df.index)), fill=0.0).to_numpy(dtype=float)
    gap_compression = safe_numeric(df.get("gap_compression_index", pd.Series(0.0, index=df.index)), fill=0.0).to_numpy(dtype=float)
    gap_comp_scaled = np.clip(gap_compression, -2.0, 3.0)
    safety_car_baseline = safe_numeric(df.get("safety_car_probability_baseline", pd.Series(0.0, index=df.index)), fill=0.0).to_numpy(dtype=float)

    mult = (
        1.0
        + instability_scale * inst_scaled
        + pitting_scale * np.clip(share_pitting, 0.0, 1.0)
        + compression_scale * gap_comp_scaled
        + track_prior_scale * np.clip(safety_car_baseline, 0.0, 1.0)
    )

    mult = mult * wet_flag
    mult = np.clip(mult, 0.35, 2.50)
    return mult.astype(float)


def build_vsc_step_prior(
    race_lap: pd.DataFrame,
    splits: SplitIds,
    alpha_global: float,
    alpha_circuit: float,
    alpha_phase: float,
    wet_boost: float,
    instability_scale: float,
    pitting_scale: float,
    compression_scale: float,
    track_prior_scale: float,
    smooth_alpha: float,
    smooth_baseline: float,
) -> Tuple[pd.DataFrame, Dict]:
    df = race_lap.copy()

    train_mask = np.isin(df["race_id"].astype(str).to_numpy(), np.array(splits.train_races, dtype=str))
    tables = build_train_side_tables(df, train_mask=train_mask)

    scored = apply_hierarchical_prior(
        df=df,
        tables=tables,
        alpha_global=alpha_global,
        alpha_circuit=alpha_circuit,
        alpha_phase=alpha_phase,
    )

    mult = build_instability_multiplier(
        df=scored,
        q=tables["quantiles"],
        wet_boost=wet_boost,
        instability_scale=instability_scale,
        pitting_scale=pitting_scale,
        compression_scale=compression_scale,
        track_prior_scale=track_prior_scale,
    )

    raw_step = scored["vsc_hier_prior_step"].to_numpy(dtype=float) * mult
    raw_step = np.clip(raw_step, 0.0, 0.20)

    smoothed = smooth_alpha * raw_step + (1.0 - smooth_alpha) * smooth_baseline
    smoothed = np.clip(smoothed, 0.0, 0.20)

    scored["p_vsc_step_prior"] = smoothed

    meta = {
        "mode": "vsc_instability_prior",
        "global_step_rate_train": float(tables["global_step_rate"]),
        "alpha_global": float(alpha_global),
        "alpha_circuit": float(alpha_circuit),
        "alpha_phase": float(alpha_phase),
        "wet_boost": float(wet_boost),
        "instability_scale": float(instability_scale),
        "pitting_scale": float(pitting_scale),
        "compression_scale": float(compression_scale),
        "track_prior_scale": float(track_prior_scale),
        "smooth_alpha": float(smooth_alpha),
        "smooth_baseline": float(smooth_baseline),
        "train_quantiles": tables["quantiles"],
    }
    return scored, meta


def build_vsc_horizon_predictions(scored: pd.DataFrame, max_horizon: int, horizons: List[int]) -> pd.DataFrame:
    base = scored[["race_id", "lap_number", "p_vsc_step_prior"]].copy()

    pivot = (
        base.pivot_table(index=["race_id", "lap_number"], values="p_vsc_step_prior", aggfunc="first")
        .reset_index()
    )

    pred = pivot[["race_id", "lap_number"]].copy()
    step_p = pivot["p_vsc_step_prior"].to_numpy(dtype=float)

    step_mat = np.repeat(step_p.reshape(-1, 1), max_horizon, axis=1)
    cum = step_to_cumulative(step_mat)

    for h in horizons:
        if h <= max_horizon:
            pred[f"p_vsc_next{h}"] = cum[:, h - 1]

    return pred


def build_vsc_horizon_truth(race_lap: pd.DataFrame, max_horizon: int, horizons: List[int]) -> pd.DataFrame:
    df = race_lap.copy()
    start_col = "vsc_start_this_lap"

    future_start = df[["race_id", "lap_number", start_col]].copy()
    base = df[["race_id", "lap_number", "total_laps"]].copy()

    out = base[["race_id", "lap_number"]].drop_duplicates().copy()

    for h in horizons:
        parts = []
        for k in range(1, h + 1):
            tmp = base[["race_id", "lap_number", "total_laps"]].copy()
            tmp["future_lap_number"] = tmp["lap_number"] + k
            tmp = tmp.loc[tmp["future_lap_number"] <= tmp["total_laps"]].copy()
            tmp = tmp.merge(
                future_start.rename(columns={"lap_number": "future_lap_number", start_col: "y_future"}),
                on=["race_id", "future_lap_number"],
                how="left",
            )
            tmp["y_future"] = tmp["y_future"].fillna(0).astype(int)
            parts.append(tmp[["race_id", "lap_number", "y_future"]])

        if parts:
            allp = pd.concat(parts, axis=0, ignore_index=True)
            y = (
                allp.groupby(["race_id", "lap_number"], as_index=False)["y_future"]
                .max()
                .rename(columns={"y_future": f"y_vsc_next{h}"})
            )
            out = out.merge(y, on=["race_id", "lap_number"], how="left")
            out[f"y_vsc_next{h}"] = out[f"y_vsc_next{h}"].fillna(0).astype(int)

    return out


def evaluate_by_horizon(pred_df: pd.DataFrame, truth_df: pd.DataFrame, splits: SplitIds, horizons: List[int]) -> Dict:
    merged = pred_df.merge(truth_df, on=["race_id", "lap_number"], how="left")

    train_mask = np.isin(merged["race_id"].astype(str).to_numpy(), np.array(splits.train_races, dtype=str))
    val_mask = np.isin(merged["race_id"].astype(str).to_numpy(), np.array(splits.val_races, dtype=str))
    test_mask = np.isin(merged["race_id"].astype(str).to_numpy(), np.array(splits.test_races, dtype=str))

    out: Dict[str, Dict] = {}
    for h in horizons:
        pcol = f"p_vsc_next{h}"
        ycol = f"y_vsc_next{h}"
        if pcol not in merged.columns or ycol not in merged.columns:
            continue

        yt_train = merged.loc[train_mask, ycol].to_numpy(dtype=int)
        yp_train = merged.loc[train_mask, pcol].to_numpy(dtype=float)

        yt_val = merged.loc[val_mask, ycol].to_numpy(dtype=int)
        yp_val = merged.loc[val_mask, pcol].to_numpy(dtype=float)

        yt_test = merged.loc[test_mask, ycol].to_numpy(dtype=int)
        yp_test = merged.loc[test_mask, pcol].to_numpy(dtype=float)

        out[f"next{h}"] = {
            "train": score_split(yt_train, yp_train),
            "val": score_split(yt_val, yp_val),
            "test": score_split(yt_test, yp_test),
            "test_reliability": reliability_table(yt_test, yp_test, n_bins=10),
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_path", required=True)
    ap.add_argument("--outputs_dir", required=True)
    ap.add_argument("--models_dir", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--test_frac", type=float, default=0.20)
    ap.add_argument("--val_frac", type=float, default=0.20)
    ap.add_argument("--max_horizon", type=int, default=10)
    ap.add_argument("--horizons", default="3,5,10")

    ap.add_argument("--alpha_global", type=float, default=40.0)
    ap.add_argument("--alpha_circuit", type=float, default=30.0)
    ap.add_argument("--alpha_phase", type=float, default=20.0)

    ap.add_argument("--wet_boost", type=float, default=1.15)
    ap.add_argument("--instability_scale", type=float, default=0.22)
    ap.add_argument("--pitting_scale", type=float, default=0.18)
    ap.add_argument("--compression_scale", type=float, default=0.08)
    ap.add_argument("--track_prior_scale", type=float, default=0.10)

    ap.add_argument("--smooth_alpha", type=float, default=0.80)
    ap.add_argument("--smooth_baseline", type=float, default=0.007)

    args = ap.parse_args()

    ensure_dir(args.outputs_dir)
    ensure_dir(args.models_dir)

    ts = now_ts()
    horizons = parse_csv_list(args.horizons)

    run_meta = {
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
        "script": "build_model2_vsc_prior.py",
        "input_path": args.input_path,
        "outputs_dir": args.outputs_dir,
        "models_dir": args.models_dir,
        "seed": args.seed,
        "test_frac": args.test_frac,
        "val_frac": args.val_frac,
        "max_horizon": args.max_horizon,
        "horizons": horizons,
        "alpha_global": args.alpha_global,
        "alpha_circuit": args.alpha_circuit,
        "alpha_phase": args.alpha_phase,
        "wet_boost": args.wet_boost,
        "instability_scale": args.instability_scale,
        "pitting_scale": args.pitting_scale,
        "compression_scale": args.compression_scale,
        "track_prior_scale": args.track_prior_scale,
        "smooth_alpha": args.smooth_alpha,
        "smooth_baseline": args.smooth_baseline,
    }
    run_meta_path = os.path.join(args.outputs_dir, f"model2_vsc_prior_run_meta_{ts}.json")
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
    split_json_path = os.path.join(args.models_dir, f"model2_vsc_prior_race_splits_{ts}.json")
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

    scored, vsc_meta = build_vsc_step_prior(
        race_lap=race_lap,
        splits=splits,
        alpha_global=args.alpha_global,
        alpha_circuit=args.alpha_circuit,
        alpha_phase=args.alpha_phase,
        wet_boost=args.wet_boost,
        instability_scale=args.instability_scale,
        pitting_scale=args.pitting_scale,
        compression_scale=args.compression_scale,
        track_prior_scale=args.track_prior_scale,
        smooth_alpha=args.smooth_alpha,
        smooth_baseline=args.smooth_baseline,
    )

    pred_df = build_vsc_horizon_predictions(
        scored=scored,
        max_horizon=args.max_horizon,
        horizons=horizons,
    )

    truth_df = build_vsc_horizon_truth(
        race_lap=scored,
        max_horizon=args.max_horizon,
        horizons=horizons,
    )

    horizon_metrics = evaluate_by_horizon(
        pred_df=pred_df,
        truth_df=truth_df,
        splits=splits,
        horizons=horizons,
    )

    race_pred_path = os.path.join(args.outputs_dir, f"model2_vsc_prior_racelap_predictions_{ts}.csv")
    pred_df.to_csv(race_pred_path, index=False)

    merged = master.merge(pred_df, on=["race_id", "lap_number"], how="left")
    prob_cols = [c for c in pred_df.columns if c.startswith("p_vsc_")]
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
        merged["model2_vsc_prob_imputed_any"] = merged[[f"{c}_was_nan" for c in prob_cols]].max(axis=1).astype(int)

    merged_path = os.path.join(args.outputs_dir, f"lap_level_with_model2_vsc_prior_{ts}.csv")
    merged.to_csv(merged_path, index=False)

    metrics = {
        "event": "vsc",
        "mode": "vsc_instability_prior",
        "vsc_meta": vsc_meta,
        "horizon_metrics": horizon_metrics,
        "route_counts": {
            "rows_total": int(len(scored)),
            "rows_with_truth_next3": int(truth_df["y_vsc_next3"].notna().sum()) if "y_vsc_next3" in truth_df.columns else 0,
            "rows_with_truth_next5": int(truth_df["y_vsc_next5"].notna().sum()) if "y_vsc_next5" in truth_df.columns else 0,
            "rows_with_truth_next10": int(truth_df["y_vsc_next10"].notna().sum()) if "y_vsc_next10" in truth_df.columns else 0,
        },
        "notes": [
            "This is a structured prior model, not a supervised classifier.",
            "Base VSC risk is estimated from hierarchical train-only rates using circuit, phase, and weather.",
            "The base prior is then adjusted by local instability, pitting, gap compression, and wetness.",
            "This branch is designed to be more stable and paper-defensible than a weak supervised VSC classifier.",
        ],
    }

    metrics_path = os.path.join(args.outputs_dir, f"model2_vsc_prior_metrics_{ts}.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    model_json_path = os.path.join(args.models_dir, f"model2_vsc_prior_model_{ts}.json")
    with open(model_json_path, "w") as f:
        json.dump(vsc_meta, f, indent=2)

    summary_rows = []
    for h in horizons:
        hk = f"next{h}"
        if hk not in horizon_metrics:
            continue
        summary_rows.append(
            {
                "event": "vsc",
                "horizon": h,
                "mode": "vsc_instability_prior",
                "train_pr_auc": horizon_metrics[hk]["train"]["pr_auc"],
                "val_pr_auc": horizon_metrics[hk]["val"]["pr_auc"],
                "test_pr_auc": horizon_metrics[hk]["test"]["pr_auc"],
                "test_roc_auc": horizon_metrics[hk]["test"]["roc_auc"],
                "test_brier": horizon_metrics[hk]["test"]["brier"],
                "test_logloss": horizon_metrics[hk]["test"]["logloss"],
            }
        )
    summary_df = pd.DataFrame(summary_rows)
    summary_csv_path = os.path.join(args.outputs_dir, f"model2_vsc_prior_summary_{ts}.csv")
    summary_df.to_csv(summary_csv_path, index=False)

    report_payload = {
        "run_meta": run_meta,
        "split_json_path": split_json_path,
        "model_json_path": model_json_path,
        "race_pred_path": race_pred_path,
        "merged_path": merged_path,
        "summary_csv_path": summary_csv_path,
        "metrics_path": metrics_path,
        "metrics": metrics,
        "notes": [
            "VSC modeled as an instability prior rather than a supervised hazard classifier.",
            "Use this branch if it is more stable and interpretable than the supervised VSC alternative.",
            "Compare this report directly against the supervised VSC report before final freeze.",
        ],
    }
    report_path = os.path.join(args.outputs_dir, f"model2_vsc_prior_report_{ts}.md")
    write_model_report_md(
        report_path=report_path,
        title="Model 2 VSC Prior Report",
        payload=report_payload,
    )

    print("Saved run meta:", run_meta_path)
    print("Saved split json:", split_json_path)
    print("Saved VSC model json:", model_json_path)
    print("Saved race-lap predictions:", race_pred_path)
    print("Saved merged output:", merged_path)
    print("Saved metrics:", metrics_path)
    print("Saved summary CSV:", summary_csv_path)
    print("Saved report:", report_path)


if __name__ == "__main__":
    main()