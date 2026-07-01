from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from typing import Dict, List

import numpy as np
import pandas as pd


KEYS = ["race_id", "driver_id", "lap_number"]


def now_tag() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def parse_actions(s: str) -> List[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


def safe_num(df: pd.DataFrame, col: str, default: float) -> pd.Series:
    if col not in df.columns:
        return pd.Series(default, index=df.index, dtype=float)
    return pd.to_numeric(df[col], errors="coerce").fillna(default)


def safe_num_nan(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(np.nan, index=df.index, dtype=float)
    return pd.to_numeric(df[col], errors="coerce")


def first_existing(df: pd.DataFrame, cols: List[str], default: float) -> pd.Series:
    out = pd.Series(np.nan, index=df.index, dtype=float)
    for c in cols:
        if c in df.columns:
            s = pd.to_numeric(df[c], errors="coerce")
            out = out.where(out.notna(), s)
    return out.fillna(default)


def horizon_from_action(action: str) -> int:
    if action == "pit_now":
        return 0
    if action.startswith("wait_"):
        return int(action.split("_")[1])
    raise ValueError(f"Unknown action: {action}")


def build_fallback_table(fallback_source_path: str | None) -> Dict:
    if fallback_source_path is None:
        return {
            "global_deg": 0.03,
            "global_pace": 0.25,
            "global_sigma": 1.0,
            "group_table": pd.DataFrame(),
        }

    ref = pd.read_csv(fallback_source_path, low_memory=False)

    for c in ["tyre_degradation_per_lap", "pace_loss_vs_expected", "lap_time_sigma"]:
        if c not in ref.columns:
            ref[c] = np.nan
        ref[c] = pd.to_numeric(ref[c], errors="coerce")

    ref["tyre_degradation_per_lap"] = ref["tyre_degradation_per_lap"].clip(lower=0.0)
    ref["pace_loss_vs_expected"] = ref["pace_loss_vs_expected"].clip(lower=0.0)
    ref["lap_time_sigma"] = ref["lap_time_sigma"].clip(lower=0.05, upper=10.0)

    global_deg = float(ref["tyre_degradation_per_lap"].median())
    global_pace = float(ref["pace_loss_vs_expected"].median())
    global_sigma = float(ref["lap_time_sigma"].median())

    if not np.isfinite(global_deg):
        global_deg = 0.03
    if not np.isfinite(global_pace):
        global_pace = 0.25
    if not np.isfinite(global_sigma):
        global_sigma = 1.0

    group_cols = [c for c in ["circuit_id", "tyre_compound", "race_phase"] if c in ref.columns]

    if group_cols:
        group_table = (
            ref.groupby(group_cols, dropna=False)
            .agg(
                fallback_deg_per_lap=("tyre_degradation_per_lap", "median"),
                fallback_pace_loss=("pace_loss_vs_expected", "median"),
                fallback_sigma=("lap_time_sigma", "median"),
            )
            .reset_index()
        )
    else:
        group_table = pd.DataFrame()

    return {
        "global_deg": global_deg,
        "global_pace": global_pace,
        "global_sigma": global_sigma,
        "group_table": group_table,
    }


def apply_fallbacks(out: pd.DataFrame, fallback: Dict) -> pd.DataFrame:
    group_table = fallback["group_table"]

    if len(group_table) > 0:
        group_cols = [
            c
            for c in ["circuit_id", "tyre_compound", "race_phase"]
            if c in out.columns and c in group_table.columns
        ]

        if group_cols:
            out = out.merge(group_table, on=group_cols, how="left")
        else:
            out["fallback_deg_per_lap"] = np.nan
            out["fallback_pace_loss"] = np.nan
            out["fallback_sigma"] = np.nan
    else:
        out["fallback_deg_per_lap"] = np.nan
        out["fallback_pace_loss"] = np.nan
        out["fallback_sigma"] = np.nan

    out["tyre_degradation_per_lap_num"] = out["tyre_degradation_per_lap_num"].where(
        out["tyre_degradation_per_lap_num"].notna(),
        out["fallback_deg_per_lap"],
    )

    out["pace_loss_vs_expected_num"] = out["pace_loss_vs_expected_num"].where(
        out["pace_loss_vs_expected_num"].notna(),
        out["fallback_pace_loss"],
    )

    out["lap_time_sigma_num"] = out["lap_time_sigma_num"].where(
        out["lap_time_sigma_num"].notna(),
        out["fallback_sigma"],
    )

    out["tyre_degradation_per_lap_num"] = (
        out["tyre_degradation_per_lap_num"]
        .fillna(fallback["global_deg"])
        .clip(lower=0.0)
    )

    out["pace_loss_vs_expected_num"] = (
        out["pace_loss_vs_expected_num"]
        .fillna(fallback["global_pace"])
        .clip(lower=0.0)
    )

    out["lap_time_sigma_num"] = (
        out["lap_time_sigma_num"]
        .fillna(fallback["global_sigma"])
        .clip(lower=0.05, upper=10.0)
    )

    return out


def build_features(df: pd.DataFrame, fallback: Dict) -> pd.DataFrame:
    out = df.copy()

    out["expected_lap_time_num"] = safe_num_nan(out, "expected_lap_time")
    out["expected_lap_time_fresh_num"] = safe_num_nan(out, "expected_lap_time_fresh")
    out["tyre_degradation_per_lap_num"] = safe_num_nan(out, "tyre_degradation_per_lap")
    out["pace_loss_vs_expected_num"] = safe_num_nan(out, "pace_loss_vs_expected")
    out["lap_time_sigma_num"] = safe_num_nan(out, "lap_time_sigma")

    out = apply_fallbacks(out, fallback)

    out["pit_lane_time_loss_num"] = safe_num(out, "pit_lane_time_loss", 22.0).clip(lower=0.0)
    out["pred_rejoin_penalty_sec_nextlap_num"] = safe_num(out, "pred_rejoin_penalty_sec_nextlap", np.nan)
    out["pred_rejoin_position_delta_nextlap_num"] = safe_num(out, "pred_rejoin_position_delta_nextlap", 0.0).clip(lower=0.0)
    out["p_rejoin_clean_air_nextlap_num"] = safe_num(out, "p_rejoin_clean_air_nextlap", 0.5).clip(0.0, 1.0)

    out["p_sc_next3_num"] = safe_num(out, "p_sc_next3", 0.0).clip(0.0, 1.0)
    out["p_sc_next5_num"] = safe_num(out, "p_sc_next5", 0.0).clip(0.0, 1.0)

    out["p_vsc_next3_num"] = safe_num(out, "p_vsc_next3", 0.0).clip(0.0, 1.0)
    out["p_vsc_next5_num"] = safe_num(out, "p_vsc_next5", 0.0).clip(0.0, 1.0)

    out["p_rf_next3_num"] = safe_num(out, "p_rf_next3", 0.0).clip(0.0, 1.0)
    out["p_rf_next5_num"] = safe_num(out, "p_rf_next5", 0.0).clip(0.0, 1.0)

    out["p_pit_next1_num"] = safe_num(out, "p_pit_next1", 0.0).clip(0.0, 1.0)
    out["p_pit_next3_num"] = safe_num(out, "p_pit_next3", 0.0).clip(0.0, 1.0)
    out["p_pit_next5_num"] = safe_num(out, "p_pit_next5", 0.0).clip(0.0, 1.0)

    out["traffic_window_risk_num"] = safe_num(out, "traffic_window_risk", 0.0).clip(lower=0.0)
    out["local_pack_density_num"] = safe_num(out, "local_pack_density", 0.0).clip(lower=0.0)
    out["surrounded_pressure_num"] = safe_num(out, "surrounded_pressure", 0.0).clip(lower=0.0)
    out["position_num"] = safe_num(out, "position", 10.0)
    out["laps_remaining_num"] = safe_num(out, "laps_remaining", 0.0)

    return out


def neutralisation_probability(df: pd.DataFrame, wait_laps: int) -> pd.Series:
    if wait_laps <= 0:
        return pd.Series(0.0, index=df.index)

    if wait_laps <= 1:
        p_sc = df["p_sc_next3_num"] / 3.0
        p_vsc = df["p_vsc_next3_num"] / 3.0
        p_rf = df["p_rf_next3_num"] / 3.0
    elif wait_laps <= 3:
        p_sc = df["p_sc_next3_num"]
        p_vsc = df["p_vsc_next3_num"]
        p_rf = df["p_rf_next3_num"]
    else:
        p_sc = df["p_sc_next5_num"]
        p_vsc = df["p_vsc_next5_num"]
        p_rf = df["p_rf_next5_num"]

    return (p_sc + p_vsc + p_rf).clip(0.0, 1.0)


def neutralisation_benefit(
    df: pd.DataFrame,
    wait_laps: int,
    discount_sc: float,
    discount_vsc: float,
    discount_rf: float,
) -> pd.Series:
    if wait_laps <= 0:
        return pd.Series(0.0, index=df.index)

    if wait_laps <= 1:
        p_sc = df["p_sc_next3_num"] / 3.0
        p_vsc = df["p_vsc_next3_num"] / 3.0
        p_rf = df["p_rf_next3_num"] / 3.0
    elif wait_laps <= 3:
        p_sc = df["p_sc_next3_num"]
        p_vsc = df["p_vsc_next3_num"]
        p_rf = df["p_rf_next3_num"]
    else:
        p_sc = df["p_sc_next5_num"]
        p_vsc = df["p_vsc_next5_num"]
        p_rf = df["p_rf_next5_num"]

    pit_base = df["pit_cost_base_sec"]

    benefit = (
        p_sc.clip(0.0, 1.0) * discount_sc * pit_base
        + p_vsc.clip(0.0, 1.0) * discount_vsc * pit_base
        + p_rf.clip(0.0, 1.0) * discount_rf * pit_base
    )

    return benefit.clip(lower=0.0)


def opponent_pressure_cost(df: pd.DataFrame, wait_laps: int) -> pd.Series:
    if wait_laps <= 0:
        return pd.Series(0.0, index=df.index)

    if wait_laps <= 1:
        p = df["p_pit_next1_num"]
    elif wait_laps <= 3:
        p = df["p_pit_next3_num"]
    else:
        p = df["p_pit_next5_num"]

    return (p * wait_laps * 0.75).clip(lower=0.0)


def traffic_cost(df: pd.DataFrame) -> pd.Series:
    raw = (
        0.35 * df["traffic_window_risk_num"]
        + 0.10 * df["local_pack_density_num"]
        + 0.20 * df["surrounded_pressure_num"]
    )
    return raw.clip(lower=0.0)


def position_cost(df: pd.DataFrame, position_weight: float) -> pd.Series:
    return (df["pred_rejoin_position_delta_nextlap_num"].clip(lower=0.0) * position_weight).clip(lower=0.0)


def pit_cost_base(df: pd.DataFrame, penalty_includes_pitloss: int) -> pd.Series:
    penalty = df["pred_rejoin_penalty_sec_nextlap_num"]
    pitlane = df["pit_lane_time_loss_num"]

    if penalty_includes_pitloss == 1:
        base = penalty
    else:
        base = penalty + pitlane

    base = base.where(base.notna(), pitlane)
    return base.clip(lower=0.0)


def simulate_action(
    df: pd.DataFrame,
    action: str,
    rng: np.random.Generator,
    n_sim: int,
    args,
) -> pd.DataFrame:
    wait_laps = horizon_from_action(action)

    pit_base = df["pit_cost_base_sec"].to_numpy(dtype=float)
    sigma = df["lap_time_sigma_num"].to_numpy(dtype=float)

    wait_deg = (df["tyre_degradation_per_lap_num"] * wait_laps).clip(lower=0.0)
    wait_pace = (df["pace_loss_vs_expected_num"] * wait_laps).clip(lower=0.0)

    neut_benefit = neutralisation_benefit(
        df,
        wait_laps=wait_laps,
        discount_sc=args.neutralisation_pit_discount_sc,
        discount_vsc=args.neutralisation_pit_discount_vsc,
        discount_rf=args.neutralisation_pit_discount_rf,
    )

    opp_cost = opponent_pressure_cost(df, wait_laps)
    traffic = traffic_cost(df)
    pos_cost = position_cost(df, args.position_weight)

    deterministic = (
        pit_base
        + wait_deg.to_numpy(dtype=float)
        + wait_pace.to_numpy(dtype=float)
        + opp_cost.to_numpy(dtype=float)
        + traffic.to_numpy(dtype=float)
        + pos_cost.to_numpy(dtype=float)
        - neut_benefit.to_numpy(dtype=float)
    )

    noise_scale = np.maximum(sigma * args.risk_weight, 0.05)
    sim_noise = rng.normal(loc=0.0, scale=noise_scale.reshape(-1, 1), size=(len(df), n_sim))
    sim_cost = deterministic.reshape(-1, 1) + sim_noise

    mean_cost = np.mean(sim_cost, axis=1)
    median_cost = np.median(sim_cost, axis=1)
    p10 = np.quantile(sim_cost, 0.10, axis=1)
    p90 = np.quantile(sim_cost, 0.90, axis=1)
    risk_spread = p90 - p10

    out = df[KEYS].copy()
    out["action"] = action
    out["wait_laps"] = wait_laps
    out["mean_cost_sec"] = mean_cost
    out["median_cost_sec"] = median_cost
    out["p10_cost_sec"] = p10
    out["p90_cost_sec"] = p90
    out["risk_spread_sec"] = risk_spread
    out["pit_cost_sec"] = pit_base
    out["wait_deg_cost_sec"] = wait_deg.to_numpy(dtype=float)
    out["wait_pace_cost_sec"] = wait_pace.to_numpy(dtype=float)
    out["neutralisation_benefit_sec"] = neut_benefit.to_numpy(dtype=float)
    out["opponent_pressure_cost_sec"] = opp_cost.to_numpy(dtype=float)
    out["traffic_risk_cost_sec"] = traffic.to_numpy(dtype=float)
    out["position_cost_sec"] = pos_cost.to_numpy(dtype=float)

    return out


def strategy_summary(strategy_table: pd.DataFrame) -> List[Dict]:
    rows = []
    for action, g in strategy_table.groupby("action", sort=False):
        rows.append(
            {
                "action": action,
                "n": int(len(g)),
                "mean_cost_sec": float(g["mean_cost_sec"].mean()),
                "median_cost_sec": float(g["median_cost_sec"].median()),
                "p10_cost_sec": float(g["p10_cost_sec"].median()),
                "p90_cost_sec": float(g["p90_cost_sec"].median()),
                "mean_risk_spread_sec": float(g["risk_spread_sec"].mean()),
                "mean_pit_cost_sec": float(g["pit_cost_sec"].mean()),
                "mean_wait_deg_cost_sec": float(g["wait_deg_cost_sec"].mean()),
                "mean_wait_pace_cost_sec": float(g["wait_pace_cost_sec"].mean()),
                "mean_neutralisation_benefit_sec": float(g["neutralisation_benefit_sec"].mean()),
                "mean_opponent_pressure_cost_sec": float(g["opponent_pressure_cost_sec"].mean()),
                "mean_traffic_risk_cost_sec": float(g["traffic_risk_cost_sec"].mean()),
                "mean_position_cost_sec": float(g["position_cost_sec"].mean()),
            }
        )
    return rows


def choose_best_actions(strategy_table: pd.DataFrame) -> pd.DataFrame:
    sort_cols = KEYS + ["mean_cost_sec", "risk_spread_sec"]
    tmp = strategy_table.sort_values(sort_cols).copy()
    best = tmp.groupby(KEYS, as_index=False).first()

    best = best.rename(
        columns={
            "action": "recommended_action",
            "mean_cost_sec": "recommended_mean_cost_sec",
            "median_cost_sec": "recommended_median_cost_sec",
            "p10_cost_sec": "recommended_p10_cost_sec",
            "p90_cost_sec": "recommended_p90_cost_sec",
            "risk_spread_sec": "recommended_risk_spread_sec",
        }
    )

    return best


def recommendation_summary(best_actions: pd.DataFrame) -> List[Dict]:
    vc = best_actions["recommended_action"].value_counts(dropna=False)
    total = int(vc.sum())

    return [
        {
            "recommended_action": str(action),
            "count": int(count),
            "share": float(count / total) if total else float("nan"),
        }
        for action, count in vc.items()
    ]


def historical_alignment(df: pd.DataFrame, best_actions: pd.DataFrame) -> Dict:
    base = df[KEYS].copy()
    base["historical_pit_now"] = 1

    m = base.merge(best_actions[KEYS + ["recommended_action"]], on=KEYS, how="left")
    pred_pit_now = (m["recommended_action"] == "pit_now").astype(int)
    actual = m["historical_pit_now"].astype(int)

    tp = int(((pred_pit_now == 1) & (actual == 1)).sum())
    fp = int(((pred_pit_now == 1) & (actual == 0)).sum())
    fn = int(((pred_pit_now == 0) & (actual == 1)).sum())
    tn = int(((pred_pit_now == 0) & (actual == 0)).sum())

    n = int(len(m))
    accuracy = float((tp + tn) / n) if n else float("nan")
    precision = float(tp / (tp + fp)) if (tp + fp) else float("nan")
    recall = float(tp / (tp + fn)) if (tp + fn) else float("nan")

    return {
        "n": n,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
    }


def write_report(path: str, metrics: Dict) -> None:
    lines = []
    lines.append("# Model 5A Local Strategy Simulator Report")
    lines.append("")
    lines.append(f"Timestamp: {metrics['timestamp']}")
    lines.append(f"Input path: `{metrics['input_path']}`")
    lines.append(f"Rows input: {metrics['rows_input']}")
    lines.append(f"Actions: {', '.join(metrics['actions'])}")
    lines.append(f"Simulations per row: {metrics['n_sim']}")
    lines.append("")
    lines.append("## Recommendation Summary")
    lines.append("")
    for row in metrics["recommendation_summary"]:
        lines.append(
            f"{row['recommended_action']}: {row['count']} rows, share {row['share']}"
        )
    lines.append("")
    lines.append("## Strategy Summary")
    lines.append("")
    for row in metrics["strategy_summary"]:
        lines.append(f"### {row['action']}")
        lines.append(f"Mean cost sec: {row['mean_cost_sec']}")
        lines.append(f"Median cost sec: {row['median_cost_sec']}")
        lines.append(f"Mean pit cost sec: {row['mean_pit_cost_sec']}")
        lines.append(f"Mean wait degradation cost sec: {row['mean_wait_deg_cost_sec']}")
        lines.append(f"Mean wait pace cost sec: {row['mean_wait_pace_cost_sec']}")
        lines.append(f"Mean neutralisation benefit sec: {row['mean_neutralisation_benefit_sec']}")
        lines.append("")
    lines.append("## Notes")
    lines.append("")
    for note in metrics["notes"]:
        lines.append(f"* {note}")

    with open(path, "w") as f:
        f.write("\n".join(lines))


def main() -> None:
    ap = argparse.ArgumentParser()

    ap.add_argument("--input_path", required=True)
    ap.add_argument("--fallback_source_path", default=None)
    ap.add_argument("--outputs_dir", default="outputs")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n_sim", type=int, default=3000)
    ap.add_argument("--actions", default="pit_now,wait_1,wait_3,wait_5")
    ap.add_argument("--output_prefix", default="model5a_local_strategy")
    ap.add_argument("--penalty_includes_pitloss", type=int, default=1)
    ap.add_argument("--neutralisation_pit_discount_sc", type=float, default=0.45)
    ap.add_argument("--neutralisation_pit_discount_vsc", type=float, default=0.25)
    ap.add_argument("--neutralisation_pit_discount_rf", type=float, default=0.65)
    ap.add_argument("--risk_weight", type=float, default=2.0)
    ap.add_argument("--position_weight", type=float, default=1.5)

    args = ap.parse_args()

    ensure_dir(args.outputs_dir)

    tag = now_tag()
    actions = parse_actions(args.actions)

    df_raw = pd.read_csv(args.input_path, low_memory=False)

    missing_keys = [c for c in KEYS if c not in df_raw.columns]
    if missing_keys:
        raise ValueError(f"Missing key columns: {missing_keys}")

    if "pred_rejoin_penalty_sec_nextlap" not in df_raw.columns:
        raise ValueError("Missing pred_rejoin_penalty_sec_nextlap. Model 5A needs Model 4C output.")

    df_raw = df_raw.copy()

    for c in KEYS:
        df_raw[c] = df_raw[c].astype(str)

    mask = pd.to_numeric(df_raw["pred_rejoin_penalty_sec_nextlap"], errors="coerce").notna()
    df_raw = df_raw.loc[mask].copy()

    fallback = build_fallback_table(args.fallback_source_path)
    df = build_features(df_raw, fallback)

    df["pit_cost_base_sec"] = pit_cost_base(df, args.penalty_includes_pitloss)

    rng = np.random.default_rng(args.seed)

    tables = []
    for action in actions:
        action_table = simulate_action(
            df=df,
            action=action,
            rng=rng,
            n_sim=args.n_sim,
            args=args,
        )
        tables.append(action_table)

    strategy_table = pd.concat(tables, ignore_index=True)
    best_actions = choose_best_actions(strategy_table)

    scored = df_raw.merge(best_actions, on=KEYS, how="left")

    strategy_path = os.path.join(args.outputs_dir, f"{args.output_prefix}_strategy_table_{tag}.csv")
    best_path = os.path.join(args.outputs_dir, f"{args.output_prefix}_best_actions_{tag}.csv")
    scored_path = os.path.join(args.outputs_dir, f"{args.output_prefix}_scored_{tag}.csv")
    metrics_path = os.path.join(args.outputs_dir, f"{args.output_prefix}_metrics_{tag}.json")
    config_path = os.path.join(args.outputs_dir, f"{args.output_prefix}_config_{tag}.json")
    report_path = os.path.join(args.outputs_dir, f"{args.output_prefix}_report_{tag}.md")

    strategy_table.to_csv(strategy_path, index=False)
    best_actions.to_csv(best_path, index=False)
    scored.to_csv(scored_path, index=False)

    metrics = {
        "timestamp": tag,
        "mode": "model5a_local_strategy_simulator",
        "input_path": args.input_path,
        "fallback_source_path": args.fallback_source_path,
        "rows_input": int(len(df_raw)),
        "actions": actions,
        "n_sim": int(args.n_sim),
        "settings": {
            "penalty_includes_pitloss": int(args.penalty_includes_pitloss),
            "neutralisation_pit_discount_sc": float(args.neutralisation_pit_discount_sc),
            "neutralisation_pit_discount_vsc": float(args.neutralisation_pit_discount_vsc),
            "neutralisation_pit_discount_rf": float(args.neutralisation_pit_discount_rf),
            "risk_weight": float(args.risk_weight),
            "position_weight": float(args.position_weight),
            "seed": int(args.seed),
            "fallback_global_deg": float(fallback["global_deg"]),
            "fallback_global_pace": float(fallback["global_pace"]),
            "fallback_global_sigma": float(fallback["global_sigma"]),
        },
        "outputs": {
            "strategy_table": strategy_path,
            "best_actions": best_path,
            "scored_output": scored_path,
            "metrics": metrics_path,
            "config": config_path,
            "report": report_path,
        },
        "strategy_summary": strategy_summary(strategy_table),
        "recommendation_summary": recommendation_summary(best_actions),
        "validation_metrics": {
            "historical_pit_now_alignment": historical_alignment(df_raw, best_actions)
        },
        "notes": [
            "Model 5A consumes frozen prediction columns from Models 1, 2, 3, and 4.",
            "Model 5A is run only on pit candidate rows where Model 4C has a rejoin penalty prediction.",
            "Because pit candidate rows are in laps, Model 1 degradation and pace fields are filled using train derived fallback medians from the full repaired file.",
            "It evaluates local pit timing actions rather than simulating the full race to the chequered flag.",
            "The output is designed to feed Model 5B and Model 6.",
        ],
    }

    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    with open(config_path, "w") as f:
        json.dump(
            {
                "mode": "model5a_local_strategy_simulator",
                "input_path": args.input_path,
                "fallback_source_path": args.fallback_source_path,
                "actions": actions,
                "n_sim": int(args.n_sim),
                "settings": metrics["settings"],
                "outputs": metrics["outputs"],
            },
            f,
            indent=2,
        )

    write_report(report_path, metrics)

    print("Saved strategy table:", strategy_path)
    print("Saved best actions:", best_path)
    print("Saved scored output:", scored_path)
    print("Saved metrics:", metrics_path)
    print("Saved config:", config_path)
    print("Saved report:", report_path)


if __name__ == "__main__":
    main()