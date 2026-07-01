from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd


# ============================================================
# Model 6: Final Pipeline Runner + Risk-Aware Recommendation
# ============================================================
#
# Pipeline mode:
#   Model 6 runs Model 5B first, stores Model 5B output inside
#   the Model 6 run folder, then creates final recommendation.
#
# Existing-result mode:
#   If --model5b_metrics is provided, Model 6 copies that existing
#   Model 5B run into its own folder and then produces the Model 6
#   recommendation.
#
# Output structure:
#
# outputs/model6_runs/<run_name>/
#   model5b_run/
#       results.csv
#       metrics.json
#       config.json
#       report.md
#
#   model6_result/
#       model6_ranked_strategies.csv
#       model6_recommendation.json
#       model6_technical_report.md
#       model6_layman_report.md
#       model6_combined_report.md
#
# ============================================================


# ============================================================
# Utility helpers
# ============================================================


def now_tag() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def safe_name(x: str) -> str:
    x = str(x).strip()
    x = re.sub(r"[^A-Za-z0-9_]+", "_", x)
    x = re.sub(r"_+", "_", x)
    return x.strip("_")


def ensure_dir(path: str | Path) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def clean_for_json(obj):
    if isinstance(obj, dict):
        return {str(k): clean_for_json(v) for k, v in obj.items()}

    if isinstance(obj, list):
        return [clean_for_json(v) for v in obj]

    if isinstance(obj, tuple):
        return [clean_for_json(v) for v in obj]

    if isinstance(obj, np.integer):
        return int(obj)

    if isinstance(obj, np.floating):
        x = float(obj)
        if not np.isfinite(x):
            return None
        return x

    if isinstance(obj, float):
        if not np.isfinite(obj):
            return None
        return obj

    try:
        if pd.isna(obj):
            return None
    except Exception:
        pass

    return obj


def safe_float(x, default: float = 0.0) -> float:
    try:
        value = float(x)

        if not np.isfinite(value):
            return float(default)

        return value
    except Exception:
        return float(default)


def format_pct(x: float) -> str:
    return f"{100.0 * safe_float(x, 0.0):.1f}%"


def format_num(x: float, digits: int = 2) -> str:
    return f"{safe_float(x, 0.0):.{digits}f}"


def minmax(series: pd.Series, higher_is_better: bool = True) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce").astype(float)

    if s.isna().all():
        return pd.Series(0.5, index=series.index, dtype=float)

    s = s.fillna(s.median())

    mn = float(s.min())
    mx = float(s.max())

    if abs(mx - mn) < 1e-12:
        return pd.Series(0.5, index=series.index, dtype=float)

    scaled = (s - mn) / (mx - mn)

    if higher_is_better:
        return scaled.clip(0.0, 1.0)

    return (1.0 - scaled).clip(0.0, 1.0)


# ============================================================
# Model 5B orchestration
# ============================================================


def build_model6_root_dir(args, tag: str) -> Path:
    if args.model6_run_dir:
        return Path(args.model6_run_dir)

    if args.model5b_metrics:
        name = f"existing_model5b_{args.risk_appetite}_{tag}"
    else:
        race_safe = safe_name(args.race_id)
        driver_safe = safe_name(args.driver_id)
        scenario_safe = safe_name(args.scenario)

        name = (
            f"{race_safe}_{driver_safe}_L{int(args.decision_lap)}_"
            f"{scenario_safe}_{args.risk_appetite}_{tag}"
        )

    return Path(args.outputs_dir) / "model6_runs" / name


def validate_pipeline_args(args) -> None:
    if args.model5b_metrics:
        if not Path(args.model5b_metrics).exists():
            raise FileNotFoundError(f"Model 5B metrics file not found: {args.model5b_metrics}")
        return

    required = {
        "--model5b_script": args.model5b_script,
        "--input_path": args.input_path,
        "--model5a_strategy_table": args.model5a_strategy_table,
        "--race_id": args.race_id,
        "--driver_id": args.driver_id,
        "--decision_lap": args.decision_lap,
        "--scenario": args.scenario,
    }

    missing = [k for k, v in required.items() if v is None or str(v).strip() == ""]

    if missing:
        raise ValueError(
            "Missing required arguments for full Model 6 pipeline mode: "
            + ", ".join(missing)
        )

    if not Path(args.model5b_script).exists():
        raise FileNotFoundError(f"Model 5B script not found: {args.model5b_script}")

    if not Path(args.input_path).exists():
        raise FileNotFoundError(f"Input path not found: {args.input_path}")

    if not Path(args.model5a_strategy_table).exists():
        raise FileNotFoundError(f"Model 5A strategy table not found: {args.model5a_strategy_table}")


def find_latest_model5b_run(tmp_outputs_dir: Path) -> Path:
    model5b_runs_dir = tmp_outputs_dir / "model5b_runs"

    if not model5b_runs_dir.exists():
        raise FileNotFoundError(
            f"Model 5B did not create expected folder: {model5b_runs_dir}"
        )

    metrics_files = list(model5b_runs_dir.glob("*/metrics.json"))

    if not metrics_files:
        raise FileNotFoundError(
            f"No Model 5B metrics.json found under: {model5b_runs_dir}"
        )

    latest_metrics = max(metrics_files, key=lambda p: p.stat().st_mtime)

    return latest_metrics.parent


def add_optional_model5b_args(cmd: list[str], args) -> list[str]:
    optional_map = {
        "ranking_mode": args.model5b_ranking_mode,
        "dry_on_wet_hard_loss_sec": args.dry_on_wet_hard_loss_sec,
        "intermediate_on_dry_hard_loss_sec": args.intermediate_on_dry_hard_loss_sec,
        "wet_on_dry_hard_loss_sec": args.wet_on_dry_hard_loss_sec,
        "intermediate_warmup_laps": args.intermediate_warmup_laps,
        "wet_warmup_laps": args.wet_warmup_laps,
        "intermediate_warmup_loss_sec": args.intermediate_warmup_loss_sec,
        "wet_warmup_loss_sec": args.wet_warmup_loss_sec,
        "rain_crossover_delay_laps": args.rain_crossover_delay_laps,
        "model3_low_confidence_penalty": args.model3_low_confidence_penalty,
        "model3_zero_probability_penalty": args.model3_zero_probability_penalty,
        "early_wet_pit_track_position_penalty": args.early_wet_pit_track_position_penalty,
        "rival_reacts_to_rain": args.rival_reacts_to_rain,
    }

    for key, value in optional_map.items():
        if value is not None:
            cmd.extend([f"--{key}", str(value)])

    return cmd


def run_model5b_inside_model6(args, model6_root: Path) -> Path:
    """
    Runs Model 5B and stores its actual output inside:

        model6_root/model5b_run/

    Returns:
        model6_root/model5b_run/metrics.json
    """

    tmp_outputs_dir = model6_root / "_tmp_model5b_outputs"
    final_model5b_dir = model6_root / "model5b_run"

    if tmp_outputs_dir.exists():
        shutil.rmtree(tmp_outputs_dir)

    if final_model5b_dir.exists():
        shutil.rmtree(final_model5b_dir)

    ensure_dir(tmp_outputs_dir)

    cmd = [
        sys.executable,
        args.model5b_script,
        "--input_path",
        args.input_path,
        "--model5a_strategy_table",
        args.model5a_strategy_table,
        "--outputs_dir",
        str(tmp_outputs_dir),
        "--race_id",
        str(args.race_id),
        "--driver_id",
        str(args.driver_id),
        "--decision_lap",
        str(args.decision_lap),
        "--scenario",
        str(args.scenario),
        "--strategies",
        str(args.strategies),
        "--n_sim",
        str(args.n_sim),
        "--seed",
        str(args.seed),
    ]

    cmd = add_optional_model5b_args(cmd, args)

    print("")
    print("Running Model 5B from Model 6")
    print("-" * 90)
    print(" ".join(cmd))
    print("-" * 90)
    print("")

    completed = subprocess.run(
        cmd,
        text=True,
        capture_output=True,
    )

    model5b_log_path = model6_root / "model5b_run_terminal_log.txt"

    with open(model5b_log_path, "w") as f:
        f.write("COMMAND:\n")
        f.write(" ".join(cmd))
        f.write("\n\nSTDOUT:\n")
        f.write(completed.stdout)
        f.write("\n\nSTDERR:\n")
        f.write(completed.stderr)

    if completed.returncode != 0:
        print(completed.stdout)
        print(completed.stderr)

        raise RuntimeError(
            f"Model 5B failed. Check log: {model5b_log_path}"
        )

    actual_model5b_dir = find_latest_model5b_run(tmp_outputs_dir)

    shutil.copytree(actual_model5b_dir, final_model5b_dir)

    if not int(args.keep_temp):
        shutil.rmtree(tmp_outputs_dir, ignore_errors=True)

    copied_metrics = final_model5b_dir / "metrics.json"

    if not copied_metrics.exists():
        raise FileNotFoundError(f"Copied Model 5B metrics not found: {copied_metrics}")

    return copied_metrics


def copy_existing_model5b_to_model6_folder(model5b_metrics: str, model6_root: Path) -> Path:
    src_metrics = Path(model5b_metrics)

    if not src_metrics.exists():
        raise FileNotFoundError(f"Model 5B metrics file not found: {src_metrics}")

    src_dir = src_metrics.parent
    dst_dir = model6_root / "model5b_run"

    if dst_dir.exists():
        shutil.rmtree(dst_dir)

    shutil.copytree(src_dir, dst_dir)

    copied_metrics = dst_dir / "metrics.json"

    if not copied_metrics.exists():
        raise FileNotFoundError(f"Copied Model 5B metrics not found: {copied_metrics}")

    return copied_metrics


# ============================================================
# Model 5B result loading
# ============================================================


def require_columns(df: pd.DataFrame) -> pd.DataFrame:
    defaults = {
        "strategy": "unknown",
        "scenario": "unknown",
        "current_position": np.nan,
        "current_compound": "UNKNOWN",
        "strategy_compound": "UNKNOWN",
        "compound_probability": 0.0,
        "mean_projected_race_time": np.nan,
        "median_projected_race_time": np.nan,
        "p90_projected_race_time": np.nan,
        "cvar90_projected_race_time": np.nan,
        "mean_finish_position": np.nan,
        "median_finish_position": np.nan,
        "p10_finish_position": np.nan,
        "p90_finish_position": np.nan,
        "mean_points": 0.0,
        "p_points": 0.0,
        "p_win": 0.0,
        "p_podium": 0.0,
        "p_top5": 0.0,
        "p_top10": 0.0,
        "mean_expected_overtakes": 0.0,
        "mean_raw_rivals_ahead_before_overtake_layer": 0.0,
        "pit_cost_used": np.nan,
        "pit_cost_source": "unknown",
        "target_mean_stint_cost": np.nan,
        "target_mean_warmup_crossover_cost": 0.0,
        "target_mean_model3_confidence_cost": 0.0,
        "target_mean_neutralisation": 0.0,
        "target_mean_pit_flag": 0.0,
        "model1_deg_per_lap_used": np.nan,
        "model1_pace_loss_used": np.nan,
        "model1_sigma_used": np.nan,
        "field_size": np.nan,
    }

    out = df.copy()

    for col, default in defaults.items():
        if col not in out.columns:
            out[col] = default

    numeric_cols = [
        "current_position",
        "compound_probability",
        "mean_projected_race_time",
        "median_projected_race_time",
        "p90_projected_race_time",
        "cvar90_projected_race_time",
        "mean_finish_position",
        "median_finish_position",
        "p10_finish_position",
        "p90_finish_position",
        "mean_points",
        "p_points",
        "p_win",
        "p_podium",
        "p_top5",
        "p_top10",
        "mean_expected_overtakes",
        "mean_raw_rivals_ahead_before_overtake_layer",
        "pit_cost_used",
        "target_mean_stint_cost",
        "target_mean_warmup_crossover_cost",
        "target_mean_model3_confidence_cost",
        "target_mean_neutralisation",
        "target_mean_pit_flag",
        "model1_deg_per_lap_used",
        "model1_pace_loss_used",
        "model1_sigma_used",
        "field_size",
    ]

    for col in numeric_cols:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    probability_cols = [
        "compound_probability",
        "p_points",
        "p_win",
        "p_podium",
        "p_top5",
        "p_top10",
        "target_mean_neutralisation",
        "target_mean_pit_flag",
    ]

    for col in probability_cols:
        out[col] = out[col].fillna(0.0).clip(0.0, 1.0)

    out["strategy"] = out["strategy"].astype(str)
    out["scenario"] = out["scenario"].astype(str)
    out["current_compound"] = out["current_compound"].astype(str)
    out["strategy_compound"] = out["strategy_compound"].astype(str)

    return out


def load_model5b_metrics(metrics_path: Path) -> Tuple[pd.DataFrame, Dict]:
    with open(metrics_path, "r") as f:
        payload = json.load(f)

    if "results" in payload and isinstance(payload["results"], list):
        df = pd.DataFrame(payload["results"])
    elif "best_strategy" in payload:
        df = pd.DataFrame([payload["best_strategy"]])
    else:
        raise ValueError("Model 5B metrics.json does not contain results or best_strategy.")

    df = require_columns(df)

    if len(df) == 0:
        raise ValueError("No strategy rows found in Model 5B results.")

    return df, payload


def infer_race_context(df: pd.DataFrame, payload: Dict) -> Dict:
    first = df.iloc[0]

    current_position = safe_float(first.get("current_position", np.nan), np.nan)
    field_size = safe_float(first.get("field_size", np.nan), np.nan)

    if not np.isfinite(current_position):
        current_position = (
            float(df["current_position"].dropna().median())
            if df["current_position"].notna().any()
            else 10.0
        )

    if not np.isfinite(field_size):
        field_size = (
            float(df["field_size"].dropna().median())
            if df["field_size"].notna().any()
            else 20.0
        )

    scenario = str(first.get("scenario", "unknown"))
    scenario_lower = scenario.lower()

    if current_position <= 3:
        race_role = "front_runner"
    elif current_position <= 10:
        race_role = "points_fight"
    else:
        race_role = "recovery_drive"

    if "wet" in scenario_lower or "rain" in scenario_lower:
        scenario_type = "wet_or_rain"
    elif "vsc" in scenario_lower:
        scenario_type = "virtual_safety_car"
    elif "sc" in scenario_lower:
        scenario_type = "safety_car"
    elif "no_neutralisation" in scenario_lower:
        scenario_type = "pure_green_flag"
    else:
        scenario_type = "base"

    query = payload.get("query", {}) if isinstance(payload, dict) else {}

    return {
        "race_id": query.get("race_id", "unknown"),
        "driver_id": query.get("driver_id", "unknown"),
        "requested_decision_lap": query.get("requested_decision_lap", None),
        "actual_lap_used": query.get("actual_lap_used", None),
        "scenario": scenario,
        "scenario_type": scenario_type,
        "current_position": float(current_position),
        "field_size": float(field_size),
        "race_role": race_role,
    }


# ============================================================
# Model 6 scoring
# ============================================================


def get_weights(race_role: str, risk_appetite: str) -> Dict[str, float]:
    if race_role == "front_runner":
        base = {
            "mean_points": 1.40,
            "finish_position": 1.25,
            "p_win": 1.25,
            "p_podium": 1.15,
            "p_top5": 0.40,
            "p_top10": 0.15,
            "downside_position": 1.15,
            "tail_time": 0.70,
            "bad_result_risk": 1.10,
            "compound_confidence": 0.25,
        }
    elif race_role == "points_fight":
        base = {
            "mean_points": 1.55,
            "finish_position": 1.10,
            "p_win": 0.15,
            "p_podium": 0.35,
            "p_top5": 0.70,
            "p_top10": 1.30,
            "downside_position": 1.05,
            "tail_time": 0.55,
            "bad_result_risk": 1.25,
            "compound_confidence": 0.35,
        }
    else:
        base = {
            "mean_points": 1.25,
            "finish_position": 0.85,
            "p_win": 0.20,
            "p_podium": 0.45,
            "p_top5": 1.00,
            "p_top10": 1.35,
            "downside_position": 0.75,
            "tail_time": 0.45,
            "bad_result_risk": 0.90,
            "compound_confidence": 0.30,
        }

    risk_appetite = str(risk_appetite).lower()

    if risk_appetite == "conservative":
        base["downside_position"] *= 1.45
        base["tail_time"] *= 1.35
        base["bad_result_risk"] *= 1.55
        base["p_win"] *= 0.70
        base["p_podium"] *= 0.85
        base["compound_confidence"] *= 1.35
    elif risk_appetite == "aggressive":
        base["p_win"] *= 1.55
        base["p_podium"] *= 1.35
        base["p_top5"] *= 1.25
        base["mean_points"] *= 1.10
        base["downside_position"] *= 0.70
        base["tail_time"] *= 0.75
        base["bad_result_risk"] *= 0.70
        base["compound_confidence"] *= 0.75
    elif risk_appetite == "balanced":
        pass
    else:
        raise ValueError("risk_appetite must be conservative, balanced, or aggressive.")

    return base


def add_strategy_features(df: pd.DataFrame, context: Dict) -> pd.DataFrame:
    out = df.copy()

    out["downside_position_spread"] = (
        out["p90_finish_position"] - out["mean_finish_position"]
    ).clip(lower=0.0)

    out["upside_position_gain"] = (
        out["mean_finish_position"] - out["p10_finish_position"]
    ).clip(lower=0.0)

    out["time_uncertainty_spread"] = (
        out["p90_projected_race_time"] - out["median_projected_race_time"]
    ).clip(lower=0.0)

    out["tail_time_risk"] = (
        out["cvar90_projected_race_time"] - out["mean_projected_race_time"]
    ).clip(lower=0.0)

    race_role = context["race_role"]

    if race_role == "front_runner":
        out["bad_result_risk"] = 1.0 - out["p_podium"]
        out["success_probability_primary"] = out["p_podium"]
        out["primary_success_name"] = "podium"
    elif race_role == "points_fight":
        out["bad_result_risk"] = 1.0 - out["p_top10"]
        out["success_probability_primary"] = out["p_top10"]
        out["primary_success_name"] = "points finish"
    else:
        out["bad_result_risk"] = 1.0 - out["p_points"]
        out["success_probability_primary"] = out["p_points"]
        out["primary_success_name"] = "points finish"

    out["bad_result_risk"] = out["bad_result_risk"].clip(0.0, 1.0)

    out["compound_confidence_score"] = out["compound_probability"].clip(0.0, 1.0)

    out["uses_pit"] = out["strategy"].str.contains("pit", case=False, regex=False).astype(int)
    out["is_stay_out"] = out["strategy"].eq("stay_out").astype(int)
    out["is_wet_tyre"] = out["strategy_compound"].str.upper().isin(["INTERMEDIATE", "WET"]).astype(int)
    out["is_dry_tyre"] = out["strategy_compound"].str.upper().isin(["SOFT", "MEDIUM", "HARD"]).astype(int)

    return out


def score_strategies(df: pd.DataFrame, context: Dict, risk_appetite: str) -> Tuple[pd.DataFrame, Dict]:
    out = add_strategy_features(df, context)
    weights = get_weights(context["race_role"], risk_appetite)

    out["score_mean_points"] = minmax(out["mean_points"], higher_is_better=True)
    out["score_finish_position"] = minmax(out["mean_finish_position"], higher_is_better=False)
    out["score_p_win"] = out["p_win"].fillna(0.0).clip(0.0, 1.0)
    out["score_p_podium"] = out["p_podium"].fillna(0.0).clip(0.0, 1.0)
    out["score_p_top5"] = out["p_top5"].fillna(0.0).clip(0.0, 1.0)
    out["score_p_top10"] = out["p_top10"].fillna(0.0).clip(0.0, 1.0)
    out["score_downside_position"] = minmax(out["downside_position_spread"], higher_is_better=False)
    out["score_tail_time"] = minmax(out["tail_time_risk"], higher_is_better=False)
    out["score_bad_result_risk"] = 1.0 - out["bad_result_risk"].fillna(1.0).clip(0.0, 1.0)
    out["score_compound_confidence"] = out["compound_confidence_score"].fillna(0.0).clip(0.0, 1.0)

    out["model6_utility"] = (
        weights["mean_points"] * out["score_mean_points"]
        + weights["finish_position"] * out["score_finish_position"]
        + weights["p_win"] * out["score_p_win"]
        + weights["p_podium"] * out["score_p_podium"]
        + weights["p_top5"] * out["score_p_top5"]
        + weights["p_top10"] * out["score_p_top10"]
        + weights["downside_position"] * out["score_downside_position"]
        + weights["tail_time"] * out["score_tail_time"]
        + weights["bad_result_risk"] * out["score_bad_result_risk"]
        + weights["compound_confidence"] * out["score_compound_confidence"]
    )

    out["model6_utility"] = pd.to_numeric(out["model6_utility"], errors="coerce").fillna(-999.0)

    out = out.sort_values(
        [
            "model6_utility",
            "mean_points",
            "p_top10",
            "p_top5",
            "p_podium",
            "p_win",
            "mean_finish_position",
        ],
        ascending=[False, False, False, False, False, False, True],
    ).reset_index(drop=True)

    out["model6_rank"] = np.arange(1, len(out) + 1)

    return out, weights


def rank_for_profile(df: pd.DataFrame, context: Dict, risk_appetite: str) -> pd.DataFrame:
    ranked, _ = score_strategies(df, context, risk_appetite)
    return ranked


def select_aggressive_distinct(df: pd.DataFrame, context: Dict, selected_strategy: str) -> Dict:
    aggressive_ranked = rank_for_profile(df, context, "aggressive")

    sub = aggressive_ranked[
        aggressive_ranked["strategy"].astype(str) != str(selected_strategy)
    ].copy()

    if len(sub) == 0:
        return aggressive_ranked.iloc[0].to_dict()

    return sub.iloc[0].to_dict()


def select_named_recommendations(df: pd.DataFrame, context: Dict, selected_risk: str) -> Dict:
    selected = rank_for_profile(df, context, selected_risk)
    balanced = rank_for_profile(df, context, "balanced")
    conservative = rank_for_profile(df, context, "conservative")
    aggressive = rank_for_profile(df, context, "aggressive")

    selected_strategy = str(selected.iloc[0]["strategy"])

    fastest = df.sort_values(
        ["mean_projected_race_time", "mean_finish_position", "mean_points"],
        ascending=[True, True, False],
    ).reset_index(drop=True)

    best_points = df.sort_values(
        ["mean_points", "p_top10", "p_top5", "p_podium", "mean_finish_position"],
        ascending=[False, False, False, False, True],
    ).reset_index(drop=True)

    best_position = df.sort_values(
        ["mean_finish_position", "mean_points", "p_top10"],
        ascending=[True, False, False],
    ).reset_index(drop=True)

    stay_out = df[df["strategy"].astype(str) == "stay_out"].copy()

    if len(stay_out) > 0:
        baseline = stay_out.iloc[0].to_dict()
        baseline_name = "stay_out"
    else:
        baseline = selected.iloc[-1].to_dict()
        baseline_name = "lowest_ranked_available_strategy"

    safe_strategy = conservative.iloc[0].to_dict()

    aggressive_strategy = select_aggressive_distinct(
        df=df,
        context=context,
        selected_strategy=selected_strategy,
    )

    return {
        "selected": selected.iloc[0].to_dict(),
        "balanced": balanced.iloc[0].to_dict(),
        "conservative_safe": safe_strategy,
        "aggressive_upside": aggressive_strategy,
        "fastest_time": fastest.iloc[0].to_dict(),
        "best_points": best_points.iloc[0].to_dict(),
        "best_position": best_position.iloc[0].to_dict(),
        "baseline": baseline,
        "baseline_name": baseline_name,
        "ranked_selected": selected,
        "ranked_balanced": balanced,
        "ranked_conservative": conservative,
        "ranked_aggressive": aggressive,
    }


# ============================================================
# Reporting helpers
# ============================================================


def strategy_to_plain_english(strategy: str) -> str:
    strategy = str(strategy)

    if strategy == "stay_out":
        return "Stay out and do not pit."

    m = re.match(r"pit_now_to_(.+)", strategy)

    if m:
        tyre = m.group(1).replace("_", " ").title()
        return f"Pit now and switch to {tyre} tyres."

    m = re.match(r"wait_(\d+)_then_pit_to_(.+)", strategy)

    if m:
        laps = int(m.group(1))
        tyre = m.group(2).replace("_", " ").title()

        if laps == 1:
            return f"Stay out for 1 more lap, then pit for {tyre} tyres."

        return f"Stay out for {laps} more laps, then pit for {tyre} tyres."

    return strategy.replace("_", " ")


def confidence_label(ranked: pd.DataFrame) -> Tuple[str, float, str]:
    if len(ranked) <= 1:
        return "LOW", 0.0, "Only one strategy was available."

    top = safe_float(ranked.iloc[0]["model6_utility"], 0.0)
    second = safe_float(ranked.iloc[1]["model6_utility"], 0.0)
    gap = top - second

    top_points = safe_float(ranked.iloc[0]["mean_points"], 0.0)
    second_points = safe_float(ranked.iloc[1]["mean_points"], 0.0)
    points_gap = top_points - second_points

    if gap >= 0.35 or points_gap >= 1.5:
        return "HIGH", gap, "The top strategy is clearly ahead of the next option."

    if gap >= 0.12 or points_gap >= 0.5:
        return "MEDIUM", gap, "The top strategy is better, but there are close alternatives."

    return "LOW", gap, "The top strategies are very close, so this is a marginal call."


def compute_delta(row: Dict, baseline: Dict) -> Dict:
    row_points = safe_float(row.get("mean_points", 0.0))
    base_points = safe_float(baseline.get("mean_points", 0.0))

    row_pos = safe_float(row.get("mean_finish_position", np.nan), np.nan)
    base_pos = safe_float(baseline.get("mean_finish_position", np.nan), np.nan)

    position_gain = None

    if np.isfinite(row_pos) and np.isfinite(base_pos):
        position_gain = base_pos - row_pos

    return {
        "mean_points_gain_vs_baseline": row_points - base_points,
        "mean_finish_position_gain_vs_baseline": position_gain,
        "p_top10_gain_vs_baseline": safe_float(row.get("p_top10", 0.0)) - safe_float(baseline.get("p_top10", 0.0)),
        "p_top5_gain_vs_baseline": safe_float(row.get("p_top5", 0.0)) - safe_float(baseline.get("p_top5", 0.0)),
        "p_podium_gain_vs_baseline": safe_float(row.get("p_podium", 0.0)) - safe_float(baseline.get("p_podium", 0.0)),
    }


def decision_label(row: Dict, context: Dict) -> str:
    role = context["race_role"]

    p_win = safe_float(row.get("p_win", 0.0))
    p_podium = safe_float(row.get("p_podium", 0.0))
    p_top5 = safe_float(row.get("p_top5", 0.0))
    p_top10 = safe_float(row.get("p_top10", 0.0))

    if role == "front_runner":
        if p_win >= 0.35:
            return "Race-winning recommendation"
        if p_podium >= 0.80:
            return "Podium-protection recommendation"
        return "Front-runner risk-managed recommendation"

    if role == "points_fight":
        if p_top10 >= 0.80 and p_top5 >= 0.25:
            return "Strong points and top-five upside recommendation"
        if p_top10 >= 0.70:
            return "Points-protection recommendation"
        return "Risky points-fight recommendation"

    if p_top10 >= 0.40:
        return "Recovery-drive points opportunity"

    return "Low-upside recovery recommendation"


def make_top_table(ranked: pd.DataFrame, n: int = 8) -> str:
    cols = [
        "model6_rank",
        "strategy",
        "strategy_compound",
        "model6_utility",
        "mean_finish_position",
        "p90_finish_position",
        "mean_points",
        "p_win",
        "p_podium",
        "p_top5",
        "p_top10",
        "bad_result_risk",
        "downside_position_spread",
        "compound_probability",
        "target_mean_warmup_crossover_cost",
        "target_mean_model3_confidence_cost",
        "pit_cost_source",
    ]

    cols = [c for c in cols if c in ranked.columns]
    top = ranked[cols].head(n).copy()

    try:
        return top.to_markdown(index=False)
    except Exception:
        return top.to_string(index=False)


def build_recommendation_payload(
    ranked: pd.DataFrame,
    model5b_payload: Dict,
    context: Dict,
    selected_risk: str,
    weights: Dict,
    selections: Dict,
    output_paths: Dict,
    model5b_metrics_used: Path,
) -> Dict:
    selected = selections["selected"]
    baseline = selections["baseline"]

    confidence, utility_gap, confidence_reason = confidence_label(selections["ranked_selected"])
    deltas = compute_delta(selected, baseline)

    payload = {
        "timestamp": now_tag(),
        "mode": "model6_full_pipeline_runner_and_risk_aware_recommendation_layer",
        "input_model5b_mode": model5b_payload.get("mode", "unknown"),
        "model5b_metrics_used": str(model5b_metrics_used),
        "context": context,
        "risk_appetite_used": selected_risk,
        "weights_used": weights,
        "final_recommendation": {
            "strategy": selected.get("strategy"),
            "plain_english_action": strategy_to_plain_english(selected.get("strategy")),
            "decision_label": decision_label(selected, context),
            "confidence": confidence,
            "confidence_reason": confidence_reason,
            "utility_gap_to_second": utility_gap,
            "mean_finish_position": safe_float(selected.get("mean_finish_position")),
            "median_finish_position": safe_float(selected.get("median_finish_position")),
            "p90_finish_position": safe_float(selected.get("p90_finish_position")),
            "mean_points": safe_float(selected.get("mean_points")),
            "p_win": safe_float(selected.get("p_win")),
            "p_podium": safe_float(selected.get("p_podium")),
            "p_top5": safe_float(selected.get("p_top5")),
            "p_top10": safe_float(selected.get("p_top10")),
            "bad_result_risk": safe_float(selected.get("bad_result_risk")),
            "downside_position_spread": safe_float(selected.get("downside_position_spread")),
            "strategy_compound": selected.get("strategy_compound"),
            "compound_probability": safe_float(selected.get("compound_probability")),
            "pit_cost_used": safe_float(selected.get("pit_cost_used")),
            "pit_cost_source": selected.get("pit_cost_source"),
            "model6_utility": safe_float(selected.get("model6_utility")),
            "deltas_vs_baseline": deltas,
        },
        "alternative_recommendations": {
            "safe_strategy": {
                "strategy": selections["conservative_safe"].get("strategy"),
                "plain_english_action": strategy_to_plain_english(selections["conservative_safe"].get("strategy")),
                "mean_finish_position": safe_float(selections["conservative_safe"].get("mean_finish_position")),
                "mean_points": safe_float(selections["conservative_safe"].get("mean_points")),
                "p_top10": safe_float(selections["conservative_safe"].get("p_top10")),
                "p_top5": safe_float(selections["conservative_safe"].get("p_top5")),
                "p_podium": safe_float(selections["conservative_safe"].get("p_podium")),
            },
            "aggressive_strategy": {
                "strategy": selections["aggressive_upside"].get("strategy"),
                "plain_english_action": strategy_to_plain_english(selections["aggressive_upside"].get("strategy")),
                "mean_finish_position": safe_float(selections["aggressive_upside"].get("mean_finish_position")),
                "mean_points": safe_float(selections["aggressive_upside"].get("mean_points")),
                "p_top10": safe_float(selections["aggressive_upside"].get("p_top10")),
                "p_top5": safe_float(selections["aggressive_upside"].get("p_top5")),
                "p_podium": safe_float(selections["aggressive_upside"].get("p_podium")),
                "p_win": safe_float(selections["aggressive_upside"].get("p_win")),
            },
            "best_points_strategy": {
                "strategy": selections["best_points"].get("strategy"),
                "plain_english_action": strategy_to_plain_english(selections["best_points"].get("strategy")),
                "mean_points": safe_float(selections["best_points"].get("mean_points")),
            },
            "fastest_time_strategy": {
                "strategy": selections["fastest_time"].get("strategy"),
                "plain_english_action": strategy_to_plain_english(selections["fastest_time"].get("strategy")),
                "mean_projected_race_time": safe_float(selections["fastest_time"].get("mean_projected_race_time")),
            },
            "best_position_strategy": {
                "strategy": selections["best_position"].get("strategy"),
                "plain_english_action": strategy_to_plain_english(selections["best_position"].get("strategy")),
                "mean_finish_position": safe_float(selections["best_position"].get("mean_finish_position")),
            },
            "baseline_strategy": {
                "baseline_name": selections["baseline_name"],
                "strategy": baseline.get("strategy"),
                "plain_english_action": strategy_to_plain_english(baseline.get("strategy")),
                "mean_finish_position": safe_float(baseline.get("mean_finish_position")),
                "mean_points": safe_float(baseline.get("mean_points")),
                "p_top10": safe_float(baseline.get("p_top10")),
            },
        },
        "outputs": output_paths,
    }

    return payload


# ============================================================
# Report writers
# ============================================================


def write_technical_report(
    path: Path,
    recommendation: Dict,
    ranked: pd.DataFrame,
    context: Dict,
    selected_risk: str,
    weights: Dict,
) -> None:
    rec = recommendation["final_recommendation"]
    alt = recommendation["alternative_recommendations"]

    lines = []

    lines.append("MODEL 6 TECHNICAL REPORT")
    lines.append("=" * 80)
    lines.append("")

    lines.append("1. PURPOSE")
    lines.append("-" * 80)
    lines.append(
        "Model 6 is the final risk-aware recommendation layer of the Formula 1 strategy pipeline. "
        "It runs or reads Model 5B scenario simulations and converts the simulated strategy outcomes "
        "into one final recommendation."
    )
    lines.append("")

    lines.append("2. RACE CONTEXT")
    lines.append("-" * 80)
    lines.append(f"Race ID: {context['race_id']}")
    lines.append(f"Driver ID: {context['driver_id']}")
    lines.append(f"Scenario: {context['scenario']}")
    lines.append(f"Scenario type: {context['scenario_type']}")
    lines.append(f"Current position: P{format_num(context['current_position'], 1)}")
    lines.append(f"Field size: {format_num(context['field_size'], 0)}")
    lines.append(f"Race role: {context['race_role']}")
    lines.append(f"Risk appetite: {selected_risk}")
    lines.append("")

    lines.append("3. FINAL RECOMMENDATION")
    lines.append("-" * 80)
    lines.append(f"Recommended strategy: {rec['strategy']}")
    lines.append(f"Plain-English action: {rec['plain_english_action']}")
    lines.append(f"Decision label: {rec['decision_label']}")
    lines.append(f"Confidence: {rec['confidence']}")
    lines.append(f"Confidence reason: {rec['confidence_reason']}")
    lines.append(f"Utility gap to second strategy: {format_num(rec['utility_gap_to_second'], 4)}")
    lines.append("")

    lines.append("4. PREDICTED OUTCOME")
    lines.append("-" * 80)
    lines.append(f"Mean finish position: P{format_num(rec['mean_finish_position'], 2)}")
    lines.append(f"Median finish position: P{format_num(rec['median_finish_position'], 2)}")
    lines.append(f"P90 downside finish: P{format_num(rec['p90_finish_position'], 2)}")
    lines.append(f"Mean points: {format_num(rec['mean_points'], 2)}")
    lines.append(f"Win probability: {format_pct(rec['p_win'])}")
    lines.append(f"Podium probability: {format_pct(rec['p_podium'])}")
    lines.append(f"Top 5 probability: {format_pct(rec['p_top5'])}")
    lines.append(f"Top 10 probability: {format_pct(rec['p_top10'])}")
    lines.append(f"Bad-result risk: {format_pct(rec['bad_result_risk'])}")
    lines.append(f"Downside position spread: {format_num(rec['downside_position_spread'], 2)}")
    lines.append("")

    lines.append("5. STRATEGY DETAILS")
    lines.append("-" * 80)
    lines.append(f"Recommended compound: {rec['strategy_compound']}")
    lines.append(f"Model 3 compound probability: {format_pct(rec['compound_probability'])}")
    lines.append(f"Pit cost used: {format_num(rec['pit_cost_used'], 2)} seconds")
    lines.append(f"Pit cost source: {rec['pit_cost_source']}")
    lines.append(f"Model 6 utility score: {format_num(rec['model6_utility'], 4)}")
    lines.append("")

    delta = rec["deltas_vs_baseline"]
    pos_gain = delta["mean_finish_position_gain_vs_baseline"]

    if pos_gain is None:
        pos_gain_text = "not available"
    else:
        pos_gain_text = f"{format_num(pos_gain, 2)} positions"

    lines.append("6. DELTA VERSUS BASELINE")
    lines.append("-" * 80)
    lines.append(f"Mean points gain: {format_num(delta['mean_points_gain_vs_baseline'], 2)}")
    lines.append(f"Mean finish position gain: {pos_gain_text}")
    lines.append(f"Top 10 probability gain: {format_pct(delta['p_top10_gain_vs_baseline'])}")
    lines.append(f"Top 5 probability gain: {format_pct(delta['p_top5_gain_vs_baseline'])}")
    lines.append(f"Podium probability gain: {format_pct(delta['p_podium_gain_vs_baseline'])}")
    lines.append("")

    lines.append("7. ALTERNATIVE RECOMMENDATIONS")
    lines.append("-" * 80)
    lines.append(f"Safe strategy: {alt['safe_strategy']['strategy']}")
    lines.append(f"Safe action: {alt['safe_strategy']['plain_english_action']}")
    lines.append("")
    lines.append(f"Aggressive strategy: {alt['aggressive_strategy']['strategy']}")
    lines.append(f"Aggressive action: {alt['aggressive_strategy']['plain_english_action']}")
    lines.append("")
    lines.append(f"Best-points strategy: {alt['best_points_strategy']['strategy']}")
    lines.append(f"Fastest-time strategy: {alt['fastest_time_strategy']['strategy']}")
    lines.append(f"Best-position strategy: {alt['best_position_strategy']['strategy']}")
    lines.append(f"Baseline strategy: {alt['baseline_strategy']['strategy']}")
    lines.append("")

    lines.append("8. WEIGHTING CONFIGURATION")
    lines.append("-" * 80)
    lines.append("| Component | Weight |")
    lines.append("|---|---:|")

    for k, v in weights.items():
        lines.append(f"| {k} | {format_num(v, 3)} |")

    lines.append("")

    lines.append("9. TOP RANKED STRATEGIES")
    lines.append("-" * 80)
    lines.append(make_top_table(ranked, n=10))
    lines.append("")

    lines.append("10. TECHNICAL INTERPRETATION")
    lines.append("-" * 80)
    lines.append(
        "Model 6 ranks strategies using a risk-aware utility score. The score combines expected points, "
        "expected finishing position, win/podium/top-five/top-ten probability, downside position spread, "
        "tail-time risk, bad-result risk, and Model 3 compound confidence."
    )
    lines.append("")
    lines.append(
        "For points-fighting cars, the model gives more weight to top-ten probability and expected points. "
        "For front-runners, it gives more weight to win probability, podium probability, and downside protection. "
        "For recovery-drive cars, it looks for upside while still penalising strategies with poor points probability."
    )

    with open(path, "w") as f:
        f.write("\n".join(lines))


def write_layman_report(path: Path, recommendation: Dict, context: Dict) -> None:
    rec = recommendation["final_recommendation"]
    alt = recommendation["alternative_recommendations"]

    lines = []

    lines.append("MODEL 6 SIMPLE STRATEGY REPORT")
    lines.append("=" * 80)
    lines.append("")

    lines.append("1. FINAL CALL")
    lines.append("-" * 80)
    lines.append(f"Recommended strategy: {rec['strategy']}")
    lines.append(f"Simple action: {rec['plain_english_action']}")
    lines.append(f"Decision type: {rec['decision_label']}")
    lines.append(f"Confidence: {rec['confidence']}")
    lines.append("")

    lines.append("2. WHAT THIS MEANS IN SIMPLE LANGUAGE")
    lines.append("-" * 80)
    lines.append(
        f"The car is currently around P{format_num(context['current_position'], 0)}. "
        f"The model expects this strategy to finish around P{format_num(rec['mean_finish_position'], 2)} "
        f"and score about {format_num(rec['mean_points'], 2)} points."
    )
    lines.append("")

    lines.append("3. CHANCES OF IMPORTANT OUTCOMES")
    lines.append("-" * 80)
    lines.append(f"Chance of winning: {format_pct(rec['p_win'])}")
    lines.append(f"Chance of podium: {format_pct(rec['p_podium'])}")
    lines.append(f"Chance of top 5: {format_pct(rec['p_top5'])}")
    lines.append(f"Chance of top 10 / points: {format_pct(rec['p_top10'])}")
    lines.append(f"Risk of missing the main target: {format_pct(rec['bad_result_risk'])}")
    lines.append("")

    lines.append("4. WHY THE MODEL CHOSE THIS")
    lines.append("-" * 80)

    if context["race_role"] == "points_fight":
        lines.append(
            "This is a points-fighting situation. The model gives more importance to "
            "scoring points safely than taking a risky gamble for a podium."
        )
    elif context["race_role"] == "front_runner":
        lines.append(
            "This is a front-running situation. The model balances race-winning upside "
            "with protecting a podium or strong finish."
        )
    else:
        lines.append(
            "This is a recovery-drive situation. The model looks for points upside, "
            "but avoids unnecessary risk."
        )

    lines.append("")
    lines.append(rec["confidence_reason"])
    lines.append("")

    lines.append("5. SAFE OPTION")
    lines.append("-" * 80)
    lines.append(f"Safe strategy: {alt['safe_strategy']['strategy']}")
    lines.append(f"Simple action: {alt['safe_strategy']['plain_english_action']}")
    lines.append(f"Expected finish: P{format_num(alt['safe_strategy']['mean_finish_position'], 2)}")
    lines.append(f"Expected points: {format_num(alt['safe_strategy']['mean_points'], 2)}")
    lines.append(f"Top 10 chance: {format_pct(alt['safe_strategy']['p_top10'])}")
    lines.append("")

    lines.append("6. AGGRESSIVE OPTION")
    lines.append("-" * 80)
    lines.append(f"Aggressive strategy: {alt['aggressive_strategy']['strategy']}")
    lines.append(f"Simple action: {alt['aggressive_strategy']['plain_english_action']}")
    lines.append(f"Expected finish: P{format_num(alt['aggressive_strategy']['mean_finish_position'], 2)}")
    lines.append(f"Expected points: {format_num(alt['aggressive_strategy']['mean_points'], 2)}")
    lines.append(f"Top 5 chance: {format_pct(alt['aggressive_strategy']['p_top5'])}")
    lines.append(f"Podium chance: {format_pct(alt['aggressive_strategy']['p_podium'])}")
    lines.append("")

    lines.append("7. BOTTOM LINE")
    lines.append("-" * 80)
    lines.append(
        f"Final recommendation: {rec['plain_english_action']} "
        "because it gives the best balance of result, points potential, and risk."
    )
    lines.append("")
    lines.append(
        "Model 6 is the final decision layer. It first runs or reads Model 5B, "
        "then turns the simulated strategies into one clear recommendation."
    )

    with open(path, "w") as f:
        f.write("\n".join(lines))


def write_combined_report(
    path: Path,
    recommendation: Dict,
    ranked: pd.DataFrame,
    context: Dict,
    selected_risk: str,
) -> None:
    rec = recommendation["final_recommendation"]
    alt = recommendation["alternative_recommendations"]

    lines = []

    lines.append("MODEL 6 FINAL COMBINED REPORT")
    lines.append("=" * 80)
    lines.append("")

    lines.append("1. FINAL RECOMMENDATION")
    lines.append("-" * 80)
    lines.append(f"Recommended strategy: {rec['strategy']}")
    lines.append(f"Plain-English action: {rec['plain_english_action']}")
    lines.append(f"Decision label: {rec['decision_label']}")
    lines.append(f"Confidence: {rec['confidence']}")
    lines.append(f"Confidence reason: {rec['confidence_reason']}")
    lines.append("")

    lines.append("2. SIMPLE EXPLANATION")
    lines.append("-" * 80)
    lines.append(
        f"The model expects this strategy to finish around P{format_num(rec['mean_finish_position'], 2)}, "
        f"score about {format_num(rec['mean_points'], 2)} points, and give a "
        f"{format_pct(rec['p_top10'])} chance of finishing in the points."
    )
    lines.append("")
    lines.append(
        f"Win chance: {format_pct(rec['p_win'])} | "
        f"Podium chance: {format_pct(rec['p_podium'])} | "
        f"Top 5 chance: {format_pct(rec['p_top5'])} | "
        f"Top 10 chance: {format_pct(rec['p_top10'])}"
    )
    lines.append("")

    lines.append("3. TECHNICAL SUMMARY")
    lines.append("-" * 80)
    lines.append(f"Race role: {context['race_role']}")
    lines.append(f"Scenario: {context['scenario']}")
    lines.append(f"Risk appetite: {selected_risk}")
    lines.append(f"Model 6 utility score: {format_num(rec['model6_utility'], 4)}")
    lines.append(f"Expected finish: P{format_num(rec['mean_finish_position'], 2)}")
    lines.append(f"P90 downside finish: P{format_num(rec['p90_finish_position'], 2)}")
    lines.append(f"Downside position spread: {format_num(rec['downside_position_spread'], 2)}")
    lines.append(f"Bad-result risk: {format_pct(rec['bad_result_risk'])}")
    lines.append(f"Expected points: {format_num(rec['mean_points'], 2)}")
    lines.append("")

    lines.append("4. STRATEGY DETAILS")
    lines.append("-" * 80)
    lines.append(f"Recommended compound: {rec['strategy_compound']}")
    lines.append(f"Model 3 compound probability: {format_pct(rec['compound_probability'])}")
    lines.append(f"Pit cost used: {format_num(rec['pit_cost_used'], 2)} seconds")
    lines.append(f"Pit cost source: {rec['pit_cost_source']}")
    lines.append("")

    lines.append("5. ALTERNATIVE STRATEGY VIEWS")
    lines.append("-" * 80)
    lines.append(f"Safe strategy: {alt['safe_strategy']['strategy']}")
    lines.append(f"Safe action: {alt['safe_strategy']['plain_english_action']}")
    lines.append("")
    lines.append(f"Aggressive strategy: {alt['aggressive_strategy']['strategy']}")
    lines.append(f"Aggressive action: {alt['aggressive_strategy']['plain_english_action']}")
    lines.append("")
    lines.append(f"Best-points strategy: {alt['best_points_strategy']['strategy']}")
    lines.append(f"Fastest-time strategy: {alt['fastest_time_strategy']['strategy']}")
    lines.append(f"Best-position strategy: {alt['best_position_strategy']['strategy']}")
    lines.append(f"Baseline strategy: {alt['baseline_strategy']['strategy']}")
    lines.append("")

    lines.append("6. TOP RANKED STRATEGIES")
    lines.append("-" * 80)
    lines.append(make_top_table(ranked, n=8))
    lines.append("")

    lines.append("7. FINAL INTERPRETATION")
    lines.append("-" * 80)
    lines.append(
        "Model 6 is the final layer of the strategy system. It runs or reads Model 5B "
        "scenario simulations and then chooses the strategy with the best risk-adjusted outcome."
    )
    lines.append("")
    lines.append(
        "The report is written in both technical and simple language so it can be used "
        "for research evaluation as well as a race-engineer style explanation."
    )

    with open(path, "w") as f:
        f.write("\n".join(lines))


# ============================================================
# Main
# ============================================================


def main() -> None:
    ap = argparse.ArgumentParser()

    # Full pipeline mode
    ap.add_argument("--model5b_script", default="run_model5b_scenario_simulator.py")
    ap.add_argument("--input_path", default=None)
    ap.add_argument("--model5a_strategy_table", default=None)
    ap.add_argument("--race_id", default=None)
    ap.add_argument("--driver_id", default=None)
    ap.add_argument("--decision_lap", type=int, default=None)
    ap.add_argument("--scenario", default="base")
    ap.add_argument("--strategies", default="auto")
    ap.add_argument("--n_sim", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=42)

    # Existing Model 5B mode
    ap.add_argument("--model5b_metrics", default=None)

    # Model 6 output/config
    ap.add_argument("--outputs_dir", default="outputs")
    ap.add_argument("--model6_run_dir", default=None)
    ap.add_argument("--risk_appetite", default="balanced", choices=["conservative", "balanced", "aggressive"])
    ap.add_argument("--keep_temp", type=int, default=0)

    # Optional Model 5B pass-through controls
    ap.add_argument("--model5b_ranking_mode", default=None, choices=["balanced", "points", "position"])
    ap.add_argument("--dry_on_wet_hard_loss_sec", type=float, default=None)
    ap.add_argument("--intermediate_on_dry_hard_loss_sec", type=float, default=None)
    ap.add_argument("--wet_on_dry_hard_loss_sec", type=float, default=None)
    ap.add_argument("--intermediate_warmup_laps", type=int, default=None)
    ap.add_argument("--wet_warmup_laps", type=int, default=None)
    ap.add_argument("--intermediate_warmup_loss_sec", type=float, default=None)
    ap.add_argument("--wet_warmup_loss_sec", type=float, default=None)
    ap.add_argument("--rain_crossover_delay_laps", type=int, default=None)
    ap.add_argument("--model3_low_confidence_penalty", type=float, default=None)
    ap.add_argument("--model3_zero_probability_penalty", type=float, default=None)
    ap.add_argument("--early_wet_pit_track_position_penalty", type=float, default=None)
    ap.add_argument("--rival_reacts_to_rain", type=int, default=None)

    args = ap.parse_args()

    tag = now_tag()

    validate_pipeline_args(args)

    model6_root = build_model6_root_dir(args, tag)
    model5b_folder = model6_root / "model5b_run"
    model6_result_folder = model6_root / "model6_result"

    ensure_dir(model6_root)
    ensure_dir(model6_result_folder)

    if args.model5b_metrics:
        model5b_metrics_used = copy_existing_model5b_to_model6_folder(
            model5b_metrics=args.model5b_metrics,
            model6_root=model6_root,
        )
    else:
        model5b_metrics_used = run_model5b_inside_model6(
            args=args,
            model6_root=model6_root,
        )

    df, model5b_payload = load_model5b_metrics(model5b_metrics_used)
    context = infer_race_context(df, model5b_payload)

    ranked, weights = score_strategies(
        df=df,
        context=context,
        risk_appetite=args.risk_appetite,
    )

    selections = select_named_recommendations(
        df=df,
        context=context,
        selected_risk=args.risk_appetite,
    )

    ranked = selections["ranked_selected"]

    ranked_csv = model6_result_folder / "model6_ranked_strategies.csv"
    recommendation_json = model6_result_folder / "model6_recommendation.json"
    technical_report = model6_result_folder / "model6_technical_report.md"
    layman_report = model6_result_folder / "model6_layman_report.md"
    combined_report = model6_result_folder / "model6_combined_report.md"

    output_paths = {
        "model6_run_directory": str(model6_root),
        "model5b_run_folder": str(model5b_folder),
        "model6_result_folder": str(model6_result_folder),
        "ranked_strategies_csv": str(ranked_csv),
        "recommendation_json": str(recommendation_json),
        "technical_report_md": str(technical_report),
        "layman_report_md": str(layman_report),
        "combined_report_md": str(combined_report),
    }

    recommendation = build_recommendation_payload(
        ranked=ranked,
        model5b_payload=model5b_payload,
        context=context,
        selected_risk=args.risk_appetite,
        weights=weights,
        selections=selections,
        output_paths=output_paths,
        model5b_metrics_used=model5b_metrics_used,
    )

    ranked.to_csv(ranked_csv, index=False)

    with open(recommendation_json, "w") as f:
        json.dump(clean_for_json(recommendation), f, indent=2)

    write_technical_report(
        path=technical_report,
        recommendation=recommendation,
        ranked=ranked,
        context=context,
        selected_risk=args.risk_appetite,
        weights=weights,
    )

    write_layman_report(
        path=layman_report,
        recommendation=recommendation,
        context=context,
    )

    write_combined_report(
        path=combined_report,
        recommendation=recommendation,
        ranked=ranked,
        context=context,
        selected_risk=args.risk_appetite,
    )

    rec = recommendation["final_recommendation"]
    alt = recommendation["alternative_recommendations"]

    print("")
    print("MODEL 6 FULL PIPELINE FINAL RECOMMENDATION")
    print("=" * 90)
    print("Model 6 run folder:", model6_root)
    print("Model 5B stored in:", model5b_folder)
    print("Model 6 result stored in:", model6_result_folder)
    print("")
    print("Race ID:", context["race_id"])
    print("Driver ID:", context["driver_id"])
    print("Scenario:", context["scenario"])
    print("Current position:", f"P{format_num(context['current_position'], 0)}")
    print("Race role:", context["race_role"])
    print("Risk appetite:", args.risk_appetite)
    print("")
    print("Recommended strategy:", rec["strategy"])
    print("Plain-English action:", rec["plain_english_action"])
    print("Decision label:", rec["decision_label"])
    print("Confidence:", rec["confidence"])
    print("")
    print("Expected finish:", f"P{format_num(rec['mean_finish_position'], 2)}")
    print("Expected points:", format_num(rec["mean_points"], 2))
    print("Win probability:", format_pct(rec["p_win"]))
    print("Podium probability:", format_pct(rec["p_podium"]))
    print("Top 5 probability:", format_pct(rec["p_top5"]))
    print("Top 10 probability:", format_pct(rec["p_top10"]))
    print("Bad-result risk:", format_pct(rec["bad_result_risk"]))
    print("")
    print("Safe option:", alt["safe_strategy"]["strategy"])
    print("Aggressive option:", alt["aggressive_strategy"]["strategy"])
    print("Best points option:", alt["best_points_strategy"]["strategy"])
    print("Fastest time option:", alt["fastest_time_strategy"]["strategy"])
    print("")
    print("Saved ranked strategies:", ranked_csv)
    print("Saved recommendation JSON:", recommendation_json)
    print("Saved technical report:", technical_report)
    print("Saved layman report:", layman_report)
    print("Saved combined report:", combined_report)
    print("=" * 90)


if __name__ == "__main__":
    main()