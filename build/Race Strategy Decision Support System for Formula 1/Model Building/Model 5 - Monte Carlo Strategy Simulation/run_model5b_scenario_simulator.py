from __future__ import annotations

import argparse
import json
import os
import re
from datetime import datetime, timezone
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


POINTS_MAP = {
    1: 25,
    2: 18,
    3: 15,
    4: 12,
    5: 10,
    6: 8,
    7: 6,
    8: 4,
    9: 2,
    10: 1,
}


COMPOUND_PROFILE = {
    "SOFT": {
        "pace_offset": -0.18,
        "deg_mult": 1.45,
        "life": 18,
        "cliff_strength": 1.30,
        "attack_bonus": 0.18,
    },
    "MEDIUM": {
        "pace_offset": 0.00,
        "deg_mult": 1.00,
        "life": 30,
        "cliff_strength": 1.00,
        "attack_bonus": 0.08,
    },
    "HARD": {
        "pace_offset": 0.10,
        "deg_mult": 0.72,
        "life": 42,
        "cliff_strength": 0.70,
        "attack_bonus": 0.02,
    },
    "INTERMEDIATE": {
        "pace_offset": 0.35,
        "deg_mult": 1.05,
        "life": 32,
        "cliff_strength": 0.90,
        "attack_bonus": 0.06,
    },
    "WET": {
        "pace_offset": 0.70,
        "deg_mult": 1.20,
        "life": 30,
        "cliff_strength": 1.05,
        "attack_bonus": 0.03,
    },
}


DRY_COMPOUNDS = {"SOFT", "MEDIUM", "HARD"}
WET_COMPOUNDS = {"INTERMEDIATE", "WET"}
ALL_COMPOUNDS = ["SOFT", "MEDIUM", "HARD", "INTERMEDIATE", "WET"]


def now_tag() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def safe_name(x: str) -> str:
    x = str(x).strip()
    x = re.sub(r"[^A-Za-z0-9_]+", "_", x)
    x = re.sub(r"_+", "_", x)
    return x.strip("_")


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def safe_scalar(row: pd.Series, col: str, default: float) -> float:
    if col not in row.index:
        return float(default)

    value = pd.to_numeric(pd.Series([row[col]]), errors="coerce").iloc[0]

    if pd.isna(value):
        return float(default)

    return float(value)


def safe_num(df: pd.DataFrame, col: str, default: float) -> pd.Series:
    if col not in df.columns:
        return pd.Series(default, index=df.index, dtype=float)

    return pd.to_numeric(df[col], errors="coerce").fillna(default)


def clean_for_json(obj):
    if isinstance(obj, dict):
        return {str(k): clean_for_json(v) for k, v in obj.items()}

    if isinstance(obj, list):
        return [clean_for_json(v) for v in obj]

    if isinstance(obj, tuple):
        return [clean_for_json(v) for v in obj]

    if isinstance(obj, (np.integer,)):
        return int(obj)

    if isinstance(obj, (np.floating,)):
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


def parse_strategy_list(s: str, scenario: str) -> List[str]:
    if str(s).strip().lower() == "auto":
        return build_auto_strategies(scenario)

    return [x.strip() for x in s.split(",") if x.strip()]


def build_auto_strategies(scenario: str) -> List[str]:
    strategies = []

    for compound in ALL_COMPOUNDS:
        strategies.append(f"pit_now_to_{compound}")

    for wait in [1, 3, 5]:
        for compound in ALL_COMPOUNDS:
            strategies.append(f"wait_{wait}_then_pit_to_{compound}")

    strategies.append("stay_out")

    return strategies


def points_from_position(pos: np.ndarray) -> np.ndarray:
    rounded = np.rint(pos).astype(int)
    return np.array([POINTS_MAP.get(int(x), 0) for x in rounded], dtype=float)


def wait_laps_from_strategy(strategy: str) -> int:
    if strategy.startswith("pit_now"):
        return 0

    m = re.search(r"wait_(\d+)", strategy)

    if m:
        return int(m.group(1))

    if strategy == "stay_out":
        return 999

    raise ValueError(f"Unknown strategy: {strategy}")


def current_compound(row: pd.Series) -> str:
    for c in ["tyre_compound", "compound", "tyre_compound_current"]:
        if c in row.index and pd.notna(row[c]):
            value = str(row[c]).upper()
            if value in COMPOUND_PROFILE:
                return value

    return "MEDIUM"


def compound_probability(row: pd.Series, compound: str) -> float:
    compound = str(compound).upper()
    return safe_scalar(row, f"p_compound_{compound}", 0.0)


def model3_best_compound(row: pd.Series) -> Tuple[str, float]:
    probs = {c: compound_probability(row, c) for c in ALL_COMPOUNDS}
    compound = max(probs, key=probs.get)
    return compound, float(probs[compound])


def compound_from_strategy(strategy: str, row: pd.Series) -> Tuple[str, str, float]:
    if strategy == "stay_out":
        compound = current_compound(row)
        return compound, "current_tyre_stay_out", 1.0

    if "_to_" in strategy:
        compound = strategy.split("_to_")[-1].upper()

        if compound not in COMPOUND_PROFILE:
            compound = "HARD"

        prob = compound_probability(row, compound)
        return compound, "explicit_strategy", prob

    compound, prob = model3_best_compound(row)
    return compound, "model3_best_probability", prob


def parse_rain_scenario(scenario: str, decision_lap: int) -> Dict:
    if scenario in ["wet_track", "rain_now"]:
        return {
            "rain_active": True,
            "rain_start_lap": int(decision_lap),
        }

    m = re.search(r"rain_from_lap(\d+)", str(scenario))

    if m:
        return {
            "rain_active": True,
            "rain_start_lap": int(m.group(1)),
        }

    return {
        "rain_active": False,
        "rain_start_lap": None,
    }


def track_wet_state_for_lap(lap: int, scenario: str, decision_lap: int, args) -> str:
    rain = parse_rain_scenario(scenario, decision_lap)

    if not rain["rain_active"]:
        return "dry"

    rain_start = int(rain["rain_start_lap"])

    if lap < rain_start:
        return "dry"

    if lap < rain_start + int(args.rain_crossover_delay_laps):
        return "crossover"

    return "wet"


def count_track_states(start_lap: int, stint_laps: int, scenario: str, decision_lap: int, args) -> Dict[str, int]:
    counts = {
        "dry": 0,
        "crossover": 0,
        "wet": 0,
    }

    for i in range(int(max(stint_laps, 0))):
        lap = int(start_lap) + i
        state = track_wet_state_for_lap(lap, scenario, decision_lap, args)
        counts[state] += 1

    return counts


def split_dry_wet_laps(start_lap: int, stint_laps: int, scenario: str, decision_lap: int, args=None) -> Tuple[int, int]:
    if args is None:
        rain = parse_rain_scenario(scenario, decision_lap)

        if stint_laps <= 0:
            return 0, 0

        if not rain["rain_active"]:
            return int(stint_laps), 0

        dry_laps = 0
        wet_laps = 0

        for i in range(int(stint_laps)):
            lap = start_lap + i
            if lap >= int(rain["rain_start_lap"]):
                wet_laps += 1
            else:
                dry_laps += 1

        return dry_laps, wet_laps

    states = count_track_states(
        start_lap=start_lap,
        stint_laps=stint_laps,
        scenario=scenario,
        decision_lap=decision_lap,
        args=args,
    )

    return int(states["dry"]), int(states["wet"] + states["crossover"])


def build_fallbacks(full_df: pd.DataFrame, strategy_table: pd.DataFrame) -> Dict:
    df = full_df.copy()

    for c in ["tyre_degradation_per_lap", "pace_loss_vs_expected", "lap_time_sigma"]:
        if c not in df.columns:
            df[c] = np.nan
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df["tyre_degradation_per_lap"] = df["tyre_degradation_per_lap"].clip(lower=0.0)
    df["pace_loss_vs_expected"] = df["pace_loss_vs_expected"].clip(lower=0.0)
    df["lap_time_sigma"] = df["lap_time_sigma"].clip(lower=0.05, upper=10.0)

    global_deg = float(df["tyre_degradation_per_lap"].median())
    global_pace = float(df["pace_loss_vs_expected"].median())
    global_sigma = float(df["lap_time_sigma"].median())

    if not np.isfinite(global_deg):
        global_deg = 0.03

    if not np.isfinite(global_pace):
        global_pace = 0.25

    if not np.isfinite(global_sigma):
        global_sigma = 1.0

    pit_candidates = []

    if "pit_lane_time_loss" in df.columns:
        pit = pd.to_numeric(df["pit_lane_time_loss"], errors="coerce")
        pit = pit[(pit >= 12.0) & (pit <= 40.0)]

        if len(pit) > 0:
            pit_candidates.append(float(pit.median()))

    if "pit_cost_sec" in strategy_table.columns:
        pit = pd.to_numeric(strategy_table["pit_cost_sec"], errors="coerce")
        pit = pit[(pit >= 12.0) & (pit <= 40.0)]

        if len(pit) > 0:
            pit_candidates.append(float(pit.median()))

    if pit_candidates:
        global_pit_loss = float(np.median(pit_candidates))
    else:
        global_pit_loss = 22.0

    group_cols = [c for c in ["circuit_id", "tyre_compound", "race_phase"] if c in df.columns]

    if group_cols:
        model1_group = (
            df.groupby(group_cols, dropna=False)
            .agg(
                deg=("tyre_degradation_per_lap", "median"),
                pace=("pace_loss_vs_expected", "median"),
                sigma=("lap_time_sigma", "median"),
            )
            .reset_index()
        )
    else:
        model1_group = pd.DataFrame()

    if {"race_id", "circuit_id", "pit_lane_time_loss"}.issubset(df.columns):
        tmp = df.copy()
        tmp["pit_lane_time_loss_num"] = pd.to_numeric(tmp["pit_lane_time_loss"], errors="coerce")
        tmp = tmp[
            (tmp["pit_lane_time_loss_num"] >= 12.0)
            & (tmp["pit_lane_time_loss_num"] <= 40.0)
        ]

        pit_by_race = (
            tmp.groupby("race_id", dropna=False)["pit_lane_time_loss_num"]
            .median()
            .to_dict()
        )

        pit_by_circuit = (
            tmp.groupby("circuit_id", dropna=False)["pit_lane_time_loss_num"]
            .median()
            .to_dict()
        )
    else:
        pit_by_race = {}
        pit_by_circuit = {}

    return {
        "global_deg": global_deg,
        "global_pace": global_pace,
        "global_sigma": global_sigma,
        "global_pit_loss": global_pit_loss,
        "model1_group": model1_group,
        "pit_by_race": pit_by_race,
        "pit_by_circuit": pit_by_circuit,
    }


def fallback_model1_value(row: pd.Series, fallback: Dict, value_name: str, default_key: str) -> float:
    group = fallback["model1_group"]

    if len(group) == 0:
        return float(fallback[default_key])

    mask = pd.Series(True, index=group.index)

    for c in ["circuit_id", "tyre_compound", "race_phase"]:
        if c in group.columns and c in row.index:
            mask &= group[c].astype(str).eq(str(row[c]))

    sub = group.loc[mask]

    if len(sub) == 0:
        return float(fallback[default_key])

    value = pd.to_numeric(sub[value_name], errors="coerce").median()

    if not np.isfinite(value):
        return float(fallback[default_key])

    return float(value)


def get_model1_values(row: pd.Series, fallback: Dict) -> Dict:
    deg = safe_scalar(row, "tyre_degradation_per_lap", np.nan)
    pace = safe_scalar(row, "pace_loss_vs_expected", np.nan)
    sigma = safe_scalar(row, "lap_time_sigma", np.nan)

    if not np.isfinite(deg):
        deg = fallback_model1_value(row, fallback, "deg", "global_deg")

    if not np.isfinite(pace):
        pace = fallback_model1_value(row, fallback, "pace", "global_pace")

    if not np.isfinite(sigma):
        sigma = fallback_model1_value(row, fallback, "sigma", "global_sigma")

    return {
        "deg": max(float(deg), 0.0),
        "pace": max(float(pace), 0.0),
        "sigma": min(max(float(sigma), 0.05), 10.0),
    }


def realistic_pit_loss(row: pd.Series, fallback: Dict) -> Tuple[float, str]:
    raw = safe_scalar(row, "pit_lane_time_loss", np.nan)

    if np.isfinite(raw) and 12.0 <= raw <= 40.0:
        return float(raw), "row_pit_lane_time_loss"

    race_id = str(row["race_id"]) if "race_id" in row.index else None
    circuit_id = str(row["circuit_id"]) if "circuit_id" in row.index else None

    if race_id in fallback["pit_by_race"]:
        return float(fallback["pit_by_race"][race_id]), "race_median_pit_loss"

    if circuit_id in fallback["pit_by_circuit"]:
        return float(fallback["pit_by_circuit"][circuit_id]), "circuit_median_pit_loss"

    return float(fallback["global_pit_loss"]), "global_pit_loss"


def apply_scenario_probabilities(row: pd.Series, scenario: str) -> Dict:
    p_sc3 = safe_scalar(row, "p_sc_next3", 0.0)
    p_sc5 = safe_scalar(row, "p_sc_next5", p_sc3)
    p_sc10 = safe_scalar(row, "p_sc_next10", p_sc5)

    p_vsc3 = safe_scalar(row, "p_vsc_next3", 0.0)
    p_vsc5 = safe_scalar(row, "p_vsc_next5", p_vsc3)
    p_vsc10 = safe_scalar(row, "p_vsc_next10", p_vsc5)

    p_rf3 = safe_scalar(row, "p_rf_next3", 0.0)
    p_rf5 = safe_scalar(row, "p_rf_next5", p_rf3)
    p_rf10 = safe_scalar(row, "p_rf_next10", p_rf5)

    scenario = str(scenario)

    if scenario == "base":
        pass
    elif scenario == "force_sc_next3":
        p_sc3 = 1.0
        p_sc5 = 1.0
        p_sc10 = 1.0
    elif scenario == "force_sc_next5":
        p_sc5 = 1.0
        p_sc10 = 1.0
    elif scenario == "force_vsc_next3":
        p_vsc3 = 1.0
        p_vsc5 = 1.0
        p_vsc10 = 1.0
    elif scenario == "force_rf_next3":
        p_rf3 = 1.0
        p_rf5 = 1.0
        p_rf10 = 1.0
    elif scenario == "no_neutralisation":
        p_sc3 = p_sc5 = p_sc10 = 0.0
        p_vsc3 = p_vsc5 = p_vsc10 = 0.0
        p_rf3 = p_rf5 = p_rf10 = 0.0
    elif scenario in ["wet_track", "rain_now"] or scenario.startswith("rain_from_lap"):
        pass
    else:
        raise ValueError(
            "Unsupported scenario. Use base, force_sc_next3, force_sc_next5, "
            "force_vsc_next3, force_rf_next3, no_neutralisation, wet_track, "
            "rain_now, or rain_from_lapXX."
        )

    return {
        "p_sc3": float(np.clip(p_sc3, 0.0, 1.0)),
        "p_sc5": float(np.clip(p_sc5, 0.0, 1.0)),
        "p_sc10": float(np.clip(p_sc10, 0.0, 1.0)),
        "p_vsc3": float(np.clip(p_vsc3, 0.0, 1.0)),
        "p_vsc5": float(np.clip(p_vsc5, 0.0, 1.0)),
        "p_vsc10": float(np.clip(p_vsc10, 0.0, 1.0)),
        "p_rf3": float(np.clip(p_rf3, 0.0, 1.0)),
        "p_rf5": float(np.clip(p_rf5, 0.0, 1.0)),
        "p_rf10": float(np.clip(p_rf10, 0.0, 1.0)),
    }


def probs_for_wait(event_probs: Dict, wait_laps: int) -> Dict:
    if wait_laps <= 0:
        return {"sc": 0.0, "vsc": 0.0, "rf": 0.0}

    if wait_laps <= 1:
        return {
            "sc": event_probs["p_sc3"] / 3.0,
            "vsc": event_probs["p_vsc3"] / 3.0,
            "rf": event_probs["p_rf3"] / 3.0,
        }

    if wait_laps <= 3:
        return {
            "sc": event_probs["p_sc3"],
            "vsc": event_probs["p_vsc3"],
            "rf": event_probs["p_rf3"],
        }

    if wait_laps <= 5:
        return {
            "sc": event_probs["p_sc5"],
            "vsc": event_probs["p_vsc5"],
            "rf": event_probs["p_rf5"],
        }

    return {
        "sc": event_probs["p_sc10"],
        "vsc": event_probs["p_vsc10"],
        "rf": event_probs["p_rf10"],
    }


def model3_compound_confidence_cost(
    row: pd.Series,
    compound: str,
    scenario: str,
    start_lap: int,
    decision_lap: int,
    args,
) -> float:
    compound = str(compound).upper()
    prob = compound_probability(row, compound)

    wet_scenario_active = (
        scenario in ["wet_track", "rain_now"]
        or str(scenario).startswith("rain_from_lap")
    )

    track_state = track_wet_state_for_lap(
        lap=int(start_lap),
        scenario=scenario,
        decision_lap=decision_lap,
        args=args,
    )

    if prob <= 0.000001:
        if compound in WET_COMPOUNDS and wet_scenario_active:
            if track_state == "wet":
                return float(args.model3_zero_probability_penalty * 0.35)
            if track_state == "crossover":
                return float(args.model3_zero_probability_penalty * 0.55)
            return float(args.model3_zero_probability_penalty * 0.85)

        return float(args.model3_zero_probability_penalty)

    return float((1.0 - prob) * args.model3_low_confidence_penalty)


def tyre_warmup_and_crossover_cost(
    compound: str,
    start_lap: int,
    stint_laps: int,
    scenario: str,
    decision_lap: int,
    args,
) -> float:
    compound = str(compound).upper()

    if stint_laps <= 0:
        return 0.0

    states = count_track_states(
        start_lap=start_lap,
        stint_laps=stint_laps,
        scenario=scenario,
        decision_lap=decision_lap,
        args=args,
    )

    dry_laps = states["dry"]
    crossover_laps = states["crossover"]
    wet_laps = states["wet"]

    total = 0.0

    if compound == "INTERMEDIATE":
        warmup_laps = min(int(args.intermediate_warmup_laps), int(stint_laps))
        total += warmup_laps * float(args.intermediate_warmup_loss_sec)

        total += dry_laps * float(args.intermediate_on_dry_hard_loss_sec)
        total += crossover_laps * float(args.intermediate_on_dry_hard_loss_sec) * 0.35

    elif compound == "WET":
        warmup_laps = min(int(args.wet_warmup_laps), int(stint_laps))
        total += warmup_laps * float(args.wet_warmup_loss_sec)

        total += dry_laps * float(args.wet_on_dry_hard_loss_sec)
        total += crossover_laps * float(args.wet_on_dry_hard_loss_sec) * 0.50

    elif compound in DRY_COMPOUNDS:
        total += wet_laps * float(args.dry_on_wet_hard_loss_sec)
        total += crossover_laps * float(args.dry_on_wet_hard_loss_sec) * 0.35

    return float(total)


def compound_stint_cost(
    base_deg: float,
    base_pace: float,
    compound: str,
    start_age: float,
    stint_laps: int,
    old_tyre_multiplier: float,
    args,
    scenario: str,
    decision_lap: int,
    start_lap: int,
    row: pd.Series | None = None,
    apply_model3_confidence: bool = False,
) -> Dict:
    compound = str(compound).upper()

    if compound not in COMPOUND_PROFILE:
        compound = "HARD"

    if stint_laps <= 0:
        return {
            "total": 0.0,
            "pace": 0.0,
            "deg": 0.0,
            "cliff": 0.0,
            "warmup": 0.0,
            "model3_confidence": 0.0,
        }

    profile = COMPOUND_PROFILE[compound]

    adjusted_pace = base_pace + profile["pace_offset"]
    adjusted_deg = max(base_deg * profile["deg_mult"], 0.0)

    pace_cost = adjusted_pace * stint_laps
    deg_cost = adjusted_deg * stint_laps * old_tyre_multiplier

    age_end = start_age + stint_laps
    excess = max(age_end - profile["life"], 0.0)

    cliff_cost = (
        ((excess / max(profile["life"], 1.0)) ** 2)
        * stint_laps
        * profile["cliff_strength"]
        * args.tyre_cliff_weight
    )

    warmup_cost = tyre_warmup_and_crossover_cost(
        compound=compound,
        start_lap=start_lap,
        stint_laps=stint_laps,
        scenario=scenario,
        decision_lap=decision_lap,
        args=args,
    )

    model3_cost = 0.0

    if row is not None and apply_model3_confidence:
        model3_cost = model3_compound_confidence_cost(
            row=row,
            compound=compound,
            scenario=scenario,
            start_lap=start_lap,
            decision_lap=decision_lap,
            args=args,
        )

    total = pace_cost + deg_cost + cliff_cost + warmup_cost + model3_cost

    return {
        "total": float(total),
        "pace": float(pace_cost),
        "deg": float(deg_cost),
        "cliff": float(cliff_cost),
        "warmup": float(warmup_cost),
        "model3_confidence": float(model3_cost),
    }


def find_race_state(full_df: pd.DataFrame, race_id: str, decision_lap: int) -> pd.DataFrame:
    race = full_df[full_df["race_id"].astype(str) == str(race_id)].copy()

    if len(race) == 0:
        raise ValueError(f"No race found for race_id={race_id}")

    race["lap_number_num"] = pd.to_numeric(race["lap_number"], errors="coerce")

    exact = race[race["lap_number_num"] == int(decision_lap)].copy()

    if len(exact) > 0:
        return exact

    race["lap_gap"] = (race["lap_number_num"] - int(decision_lap)).abs()
    race = race.sort_values(["driver_id", "lap_gap"])

    state = race.groupby("driver_id", as_index=False).first()
    return state


def get_target_row(state: pd.DataFrame, driver_id: str) -> pd.Series:
    sub = state[state["driver_id"].astype(str) == str(driver_id)]

    if len(sub) == 0:
        available = sorted(state["driver_id"].astype(str).unique().tolist())[:20]
        raise ValueError(
            f"Driver {driver_id} not found in race state. Available examples: {available}"
        )

    return sub.iloc[0]


def infer_field_offsets(state: pd.DataFrame, args) -> pd.DataFrame:
    out = state.copy()

    out["position_num"] = safe_num(out, "position", 99.0)
    out = out.sort_values("position_num").reset_index(drop=True)

    gap_to_leader_cols = [
        "gap_to_leader",
        "time_gap_to_leader",
        "gap_to_race_leader",
        "cumulative_gap_to_leader",
        "race_time_gap_to_leader",
    ]

    used_col = None

    for col in gap_to_leader_cols:
        if col in out.columns:
            vals = pd.to_numeric(out[col], errors="coerce")
            valid = vals.notna() & (vals >= 0.0) & (vals <= 300.0)

            if valid.sum() >= max(3, int(0.30 * len(out))):
                out["initial_offset_sec"] = vals.fillna(np.nan)
                used_col = col
                break

    if used_col is None:
        gap_ahead_cols = [
            "gap_ahead",
            "time_gap_ahead",
            "interval_to_ahead",
            "gap_to_car_ahead",
        ]

        used_gap_ahead = None

        for col in gap_ahead_cols:
            if col in out.columns:
                vals = pd.to_numeric(out[col], errors="coerce")
                valid = vals.notna() & (vals >= 0.0) & (vals <= 60.0)

                if valid.sum() >= max(3, int(0.30 * len(out))):
                    used_gap_ahead = col
                    break

        if used_gap_ahead is not None:
            vals = pd.to_numeric(out[used_gap_ahead], errors="coerce")
            offsets = []
            running = 0.0

            for i in range(len(out)):
                if i == 0:
                    offsets.append(0.0)
                    continue

                gap = vals.iloc[i]

                if not np.isfinite(gap):
                    gap = args.default_position_gap_sec

                gap = float(np.clip(gap, 0.3, 15.0))
                running += gap
                offsets.append(running)

            out["initial_offset_sec"] = offsets
            used_col = used_gap_ahead
        else:
            out["initial_offset_sec"] = (
                out["position_num"].clip(lower=1.0).fillna(10.0) - 1.0
            ) * args.default_position_gap_sec
            used_col = "position_default_gap"

    out["initial_offset_sec"] = pd.to_numeric(out["initial_offset_sec"], errors="coerce")

    if out["initial_offset_sec"].isna().any():
        fallback_offsets = (
            out["position_num"].clip(lower=1.0).fillna(10.0) - 1.0
        ) * args.default_position_gap_sec
        out["initial_offset_sec"] = out["initial_offset_sec"].fillna(fallback_offsets)

    out["offset_source"] = used_col

    return out


def map_strategy_to_5a_action(strategy: str) -> str:
    if strategy.startswith("pit_now"):
        return "pit_now"

    if strategy.startswith("wait_1"):
        return "wait_1"

    if strategy.startswith("wait_3"):
        return "wait_3"

    if strategy.startswith("wait_5"):
        return "wait_5"

    return "wait_5"


def get_model5a_pit_cost(
    strategy_table: pd.DataFrame,
    race_id: str,
    driver_id: str,
    decision_lap: int,
    strategy: str,
    fallback_pit_loss: float,
) -> Dict:
    action = map_strategy_to_5a_action(strategy)

    st = strategy_table.copy()
    st["lap_number_num"] = pd.to_numeric(st["lap_number"], errors="coerce")

    same_driver = st[
        (st["race_id"].astype(str) == str(race_id))
        & (st["driver_id"].astype(str) == str(driver_id))
        & (st["action"].astype(str) == action)
    ].copy()

    if len(same_driver) > 0:
        exact = same_driver[same_driver["lap_number_num"] == int(decision_lap)]

        if len(exact) > 0:
            row = exact.iloc[0]
            pit_cost = safe_scalar(row, "pit_cost_sec", fallback_pit_loss)
            pit_cost = float(np.clip(pit_cost, 12.0, 40.0))

            return {
                "pit_cost": pit_cost,
                "source": "model5a_exact_driver_lap",
                "source_lap": int(decision_lap),
            }

        same_driver["lap_gap"] = (same_driver["lap_number_num"] - int(decision_lap)).abs()
        row = same_driver.sort_values("lap_gap").iloc[0]
        pit_cost = safe_scalar(row, "pit_cost_sec", fallback_pit_loss)
        pit_cost = float(np.clip(pit_cost, 12.0, 40.0))

        return {
            "pit_cost": pit_cost,
            "source": "model5a_nearest_driver_lap",
            "source_lap": int(row["lap_number_num"]),
        }

    same_race = st[
        (st["race_id"].astype(str) == str(race_id))
        & (st["action"].astype(str) == action)
    ].copy()

    if len(same_race) > 0 and "pit_cost_sec" in same_race.columns:
        pit = pd.to_numeric(same_race["pit_cost_sec"], errors="coerce")
        pit = pit[(pit >= 12.0) & (pit <= 40.0)]

        if len(pit) > 0:
            return {
                "pit_cost": float(pit.median()),
                "source": "model5a_same_race_action_median",
                "source_lap": None,
            }

    if "pit_cost_sec" in st.columns:
        pit = pd.to_numeric(st["pit_cost_sec"], errors="coerce")
        pit = pit[(pit >= 12.0) & (pit <= 40.0)]

        if len(pit) > 0:
            return {
                "pit_cost": float(pit.median()),
                "source": "model5a_global_action_median",
                "source_lap": None,
            }

    return {
        "pit_cost": float(fallback_pit_loss),
        "source": "fallback_realistic_pit_loss",
        "source_lap": None,
    }


def choose_rival_base_pit_wait(row: pd.Series, rng: np.random.Generator, n_sim: int) -> np.ndarray:
    p1 = safe_scalar(row, "p_pit_next1", 0.0)
    p3 = safe_scalar(row, "p_pit_next3", p1)
    p5 = safe_scalar(row, "p_pit_next5", p3)

    p1 = float(np.clip(p1, 0.0, 1.0))
    p3 = float(np.clip(p3, p1, 1.0))
    p5 = float(np.clip(p5, p3, 1.0))

    u = rng.random(n_sim)

    wait = np.full(n_sim, 999, dtype=int)
    wait[u < p5] = 5
    wait[u < p3] = 3
    wait[u < p1] = 1

    return wait


def choose_rival_pit_wait(
    row: pd.Series,
    rng: np.random.Generator,
    n_sim: int,
    scenario: str,
    actual_lap: int,
    laps_remaining: int,
    args,
) -> np.ndarray:
    base_wait = choose_rival_base_pit_wait(row, rng, n_sim)

    rain = parse_rain_scenario(scenario, actual_lap)

    if not rain["rain_active"] or int(args.rival_reacts_to_rain) != 1:
        return base_wait

    rain_start = int(rain["rain_start_lap"])

    if rain_start > actual_lap + laps_remaining:
        return base_wait

    if current_compound(row) not in DRY_COMPOUNDS:
        return base_wait

    delay_choices = np.array([0, 1, 2], dtype=int)
    delay_probs = np.array([0.55, 0.30, 0.15], dtype=float)

    delays = rng.choice(delay_choices, size=n_sim, p=delay_probs)

    rain_wait = np.maximum(rain_start - actual_lap + delays, 0)
    rain_wait = np.minimum(rain_wait, laps_remaining)

    return np.minimum(base_wait, rain_wait)


def choose_auto_compound_for_rival(
    row: pd.Series,
    scenario: str,
    actual_lap: int,
    wait_laps: int,
    laps_remaining: int,
    args,
) -> Tuple[str, str, float]:
    future_start = actual_lap + min(wait_laps, laps_remaining)
    future_laps = max(laps_remaining - min(wait_laps, laps_remaining), 0)

    states = count_track_states(
        start_lap=future_start,
        stint_laps=future_laps,
        scenario=scenario,
        decision_lap=actual_lap,
        args=args,
    )

    dry_laps = states["dry"]
    crossover_laps = states["crossover"]
    wet_laps = states["wet"]

    wet_like_laps = wet_laps + 0.5 * crossover_laps

    if wet_like_laps > dry_laps:
        p_inter = compound_probability(row, "INTERMEDIATE")
        p_wet = compound_probability(row, "WET")

        if p_wet > p_inter and p_wet > 0.05:
            return "WET", "scenario_wet_model3_probability", float(p_wet)

        return "INTERMEDIATE", "scenario_wet_intermediate_reaction", float(max(p_inter, 0.01))

    compound, prob = model3_best_compound(row)

    return compound, "model3_best_probability", prob


def expected_lap_time_for_row(row: pd.Series, args) -> float:
    for col in ["expected_lap_time", "expected_lap_time_fresh", "lap_time", "previous_lap_time"]:
        if col in row.index:
            value = safe_scalar(row, col, np.nan)

            if np.isfinite(value) and 50.0 <= value <= 160.0:
                return float(value)

    return float(args.default_lap_time)


def driver_attack_factor(row: pd.Series, compound: str, is_target: bool, args) -> float:
    clean_air = safe_scalar(row, "p_rejoin_clean_air_nextlap", 0.5)
    overtake_difficulty = safe_scalar(row, "overtaking_difficulty_index", 0.5)

    clean_air = float(np.clip(clean_air, 0.0, 1.0))
    overtake_difficulty = float(np.clip(overtake_difficulty, 0.0, 1.0))

    compound_bonus = COMPOUND_PROFILE.get(compound, COMPOUND_PROFILE["HARD"])["attack_bonus"]

    base = compound_bonus
    base += (1.0 - overtake_difficulty) * 0.08
    base += clean_air * 0.04

    if is_target:
        base += args.target_driver_attack_bias

    return float(max(base, 0.0))


def project_driver_time(
    row: pd.Series,
    strategy: str | None,
    is_target: bool,
    strategy_table: pd.DataFrame,
    fallback: Dict,
    scenario: str,
    race_id: str,
    decision_lap: int,
    n_sim: int,
    rng: np.random.Generator,
    args,
) -> Dict:
    actual_lap = int(pd.to_numeric(pd.Series([row["lap_number"]]), errors="coerce").iloc[0])

    model1 = get_model1_values(row, fallback)
    event_probs = apply_scenario_probabilities(row, scenario)

    laps_remaining = int(max(safe_scalar(row, "laps_remaining", args.default_laps_remaining), 0.0))
    tyre_age = max(safe_scalar(row, "tyre_age", 0.0), 0.0)

    base_lap_time = expected_lap_time_for_row(row, args)
    initial_offset = safe_scalar(row, "initial_offset_sec", 0.0)
    current_tyre = current_compound(row)
    current_position = safe_scalar(row, "position", np.nan)

    if is_target:
        wait_raw = wait_laps_from_strategy(strategy)

        if strategy == "stay_out":
            pit_after_wait = False
            wait_laps = np.full(n_sim, laps_remaining, dtype=int)
            strategy_compound, compound_source, compound_prob = compound_from_strategy(strategy, row)
        else:
            pit_after_wait = True
            wait_value = min(wait_raw, laps_remaining)
            wait_laps = np.full(n_sim, wait_value, dtype=int)
            strategy_compound, compound_source, compound_prob = compound_from_strategy(strategy, row)

        fallback_pit_loss, _ = realistic_pit_loss(row, fallback)

        m5a = get_model5a_pit_cost(
            strategy_table=strategy_table,
            race_id=race_id,
            driver_id=str(row["driver_id"]),
            decision_lap=decision_lap,
            strategy=strategy,
            fallback_pit_loss=fallback_pit_loss,
        )

        pit_loss = m5a["pit_cost"]
        pit_source = m5a["source"]
        pit_source_lap = m5a["source_lap"]
    else:
        wait_laps = choose_rival_pit_wait(
            row=row,
            rng=rng,
            n_sim=n_sim,
            scenario=scenario,
            actual_lap=actual_lap,
            laps_remaining=laps_remaining,
            args=args,
        )

        pit_after_wait = True

        median_wait = int(np.median(np.where(wait_laps >= 999, laps_remaining, wait_laps)))

        strategy_compound, compound_source, compound_prob = choose_auto_compound_for_rival(
            row=row,
            scenario=scenario,
            actual_lap=actual_lap,
            wait_laps=median_wait,
            laps_remaining=laps_remaining,
            args=args,
        )

        pit_loss, pit_source = realistic_pit_loss(row, fallback)
        pit_source_lap = None

    total = np.zeros(n_sim, dtype=float)
    total += initial_offset
    total += base_lap_time * laps_remaining

    stint_cost_values = np.zeros(n_sim, dtype=float)
    warmup_cost_values = np.zeros(n_sim, dtype=float)
    model3_cost_values = np.zeros(n_sim, dtype=float)
    neutralisation_flags = np.zeros(n_sim, dtype=int)
    pit_flags = np.zeros(n_sim, dtype=int)

    for i in range(n_sim):
        wait_i = int(wait_laps[i])

        if wait_i >= 999:
            wait_i = laps_remaining
            pit_i = False
        else:
            wait_i = min(wait_i, laps_remaining)
            pit_i = pit_after_wait

        active_probs = probs_for_wait(event_probs, wait_i)

        sc = rng.binomial(1, active_probs["sc"])
        vsc = rng.binomial(1, active_probs["vsc"])
        rf = rng.binomial(1, active_probs["rf"])

        neutralised = int(np.clip(sc + vsc + rf, 0, 1))
        neutralisation_flags[i] = neutralised

        wait_stint = compound_stint_cost(
            base_deg=model1["deg"],
            base_pace=model1["pace"],
            compound=current_tyre,
            start_age=tyre_age,
            stint_laps=wait_i,
            old_tyre_multiplier=args.old_tyre_wait_multiplier,
            args=args,
            scenario=scenario,
            decision_lap=actual_lap,
            start_lap=actual_lap,
            row=row,
            apply_model3_confidence=False,
        )

        future_laps = max(laps_remaining - wait_i, 0)

        if pit_i:
            pit_flags[i] = 1

            future_stint = compound_stint_cost(
                base_deg=model1["deg"],
                base_pace=model1["pace"],
                compound=strategy_compound,
                start_age=0.0,
                stint_laps=future_laps,
                old_tyre_multiplier=1.0,
                args=args,
                scenario=scenario,
                decision_lap=actual_lap,
                start_lap=actual_lap + wait_i,
                row=row,
                apply_model3_confidence=True,
            )

            discount = (
                sc * args.sc_pit_discount
                + vsc * args.vsc_pit_discount
                + rf * args.rf_pit_discount
            )
            discount = min(discount, 0.85)

            effective_pit_loss = pit_loss * (1.0 - discount)

            rain = parse_rain_scenario(scenario, actual_lap)

            if rain["rain_active"] and strategy_compound in WET_COMPOUNDS:
                rain_start_lap = int(rain["rain_start_lap"])

                if actual_lap + wait_i < rain_start_lap:
                    laps_too_early = rain_start_lap - (actual_lap + wait_i)
                    effective_pit_loss += laps_too_early * float(args.early_wet_pit_track_position_penalty)

            p_clean_air = safe_scalar(row, "p_rejoin_clean_air_nextlap", 0.5)
            clean_air = rng.binomial(1, float(np.clip(p_clean_air, 0.0, 1.0)))
            clean_air_bonus = clean_air * args.clean_air_bonus_sec
        else:
            future_stint = {
                "total": 0.0,
                "pace": 0.0,
                "deg": 0.0,
                "cliff": 0.0,
                "warmup": 0.0,
                "model3_confidence": 0.0,
            }
            effective_pit_loss = 0.0
            clean_air_bonus = 0.0

        stint_cost = wait_stint["total"] + future_stint["total"]

        stint_cost_values[i] = stint_cost
        warmup_cost_values[i] = wait_stint["warmup"] + future_stint["warmup"]
        model3_cost_values[i] = wait_stint["model3_confidence"] + future_stint["model3_confidence"]

        total[i] += effective_pit_loss
        total[i] += stint_cost
        total[i] -= clean_air_bonus
        total[i] -= neutralised * args.neutralisation_race_time_bonus_sec

        total[i] += rng.normal(
            loc=0.0,
            scale=max(model1["sigma"] * args.race_noise_weight, 0.05),
        )

    attack_factor = driver_attack_factor(
        row=row,
        compound=strategy_compound,
        is_target=is_target,
        args=args,
    )

    return {
        "driver_id": str(row["driver_id"]),
        "total_time": total,
        "expected_lap_time": float(base_lap_time),
        "initial_offset": float(initial_offset),
        "current_position": float(current_position) if np.isfinite(current_position) else np.nan,
        "laps_remaining": int(laps_remaining),
        "current_compound": current_tyre,
        "strategy_compound": strategy_compound,
        "compound_probability": float(compound_prob),
        "compound_source": compound_source,
        "pit_cost_used": float(pit_loss),
        "pit_cost_source": pit_source,
        "pit_cost_source_lap": pit_source_lap,
        "model1_deg": float(model1["deg"]),
        "model1_pace": float(model1["pace"]),
        "model1_sigma": float(model1["sigma"]),
        "attack_factor": float(attack_factor),
        "mean_stint_cost": float(np.mean(stint_cost_values)),
        "mean_warmup_crossover_cost": float(np.mean(warmup_cost_values)),
        "mean_model3_confidence_cost": float(np.mean(model3_cost_values)),
        "mean_neutralisation": float(np.mean(neutralisation_flags)),
        "mean_pit_flag": float(np.mean(pit_flags)),
    }


def overtake_probability(
    target_time: np.ndarray,
    rival_time: np.ndarray,
    target_projection: Dict,
    rival_projection: Dict,
    args,
) -> np.ndarray:
    deficit = target_time - rival_time

    target_pace = target_projection["expected_lap_time"]
    rival_pace = rival_projection["expected_lap_time"]

    raw_pace_adv = rival_pace - target_pace

    compound_attack = target_projection["attack_factor"] - rival_projection["attack_factor"]

    laps_remaining = max(target_projection["laps_remaining"], 1)

    closing_potential = (
        raw_pace_adv * laps_remaining * args.pace_advantage_overtake_weight
        + compound_attack * laps_remaining
    )

    target_pos = target_projection["current_position"]
    rival_pos = rival_projection["current_position"]

    target_offset = target_projection["initial_offset"]
    rival_offset = rival_projection["initial_offset"]
    initial_gap = max(target_offset - rival_offset, 0.0)

    leader_attack_context = (
        np.isfinite(target_pos)
        and np.isfinite(rival_pos)
        and target_pos <= 2.5
        and rival_pos < target_pos
        and initial_gap <= args.leader_attack_gap_sec
    )

    overtake_window = args.overtake_window_sec + max(closing_potential, 0.0)

    if leader_attack_context:
        overtake_window += args.leader_attack_bonus_sec
        dynamic_max_gap = max(args.max_overtake_gap_sec, initial_gap + args.leader_attack_extra_gap_sec)
    else:
        dynamic_max_gap = args.max_overtake_gap_sec

    x = (overtake_window - deficit) / max(args.overtake_temperature, 0.1)
    prob = 1.0 / (1.0 + np.exp(-x))

    prob = np.where(deficit <= 0.0, 0.0, prob)
    prob = np.where(deficit > dynamic_max_gap, 0.0, prob)

    if leader_attack_context:
        prob = np.where(
            (deficit > 0.0) & (deficit <= dynamic_max_gap),
            np.maximum(prob, args.leader_attack_floor_probability),
            prob,
        )

    return np.clip(prob, 0.0, 1.0)


def simulate_strategy_field(
    state: pd.DataFrame,
    target_row: pd.Series,
    strategy: str,
    strategy_table: pd.DataFrame,
    fallback: Dict,
    scenario: str,
    race_id: str,
    decision_lap: int,
    n_sim: int,
    rng: np.random.Generator,
    args,
) -> Dict:
    target_driver = str(target_row["driver_id"])

    target_projection = project_driver_time(
        row=target_row,
        strategy=strategy,
        is_target=True,
        strategy_table=strategy_table,
        fallback=fallback,
        scenario=scenario,
        race_id=race_id,
        decision_lap=decision_lap,
        n_sim=n_sim,
        rng=rng,
        args=args,
    )

    target_time = target_projection["total_time"]

    finish_position = np.ones(n_sim, dtype=float)

    expected_successful_overtakes = np.zeros(n_sim, dtype=float)
    raw_rivals_ahead_before_overtake_layer = np.zeros(n_sim, dtype=float)

    leader_pass_prob_values = []
    rival_projection_count = 0

    for _, rival_row in state.iterrows():
        if str(rival_row["driver_id"]) == target_driver:
            continue

        rival_projection = project_driver_time(
            row=rival_row,
            strategy=None,
            is_target=False,
            strategy_table=strategy_table,
            fallback=fallback,
            scenario=scenario,
            race_id=race_id,
            decision_lap=decision_lap,
            n_sim=n_sim,
            rng=rng,
            args=args,
        )

        rival_time = rival_projection["total_time"]

        raw_rival_ahead = rival_time < target_time

        pass_prob = overtake_probability(
            target_time=target_time,
            rival_time=rival_time,
            target_projection=target_projection,
            rival_projection=rival_projection,
            args=args,
        )

        pass_prob_for_ahead_rival = np.where(raw_rival_ahead, pass_prob, 0.0)
        pass_draw = rng.binomial(1, pass_prob_for_ahead_rival, size=n_sim)

        rival_remains_ahead = np.where(raw_rival_ahead & (pass_draw == 0), 1.0, 0.0)

        finish_position += rival_remains_ahead

        expected_successful_overtakes += pass_prob_for_ahead_rival
        raw_rivals_ahead_before_overtake_layer += raw_rival_ahead.astype(float)

        if (
            np.isfinite(rival_projection["current_position"])
            and np.isfinite(target_projection["current_position"])
            and rival_projection["current_position"] < target_projection["current_position"]
        ):
            leader_pass_prob_values.append(float(np.mean(pass_prob_for_ahead_rival)))

        rival_projection_count += 1

    finish_position = np.clip(finish_position, 1.0, float(len(state)))
    points = points_from_position(finish_position)

    cvar_threshold = np.quantile(target_time, 0.90)
    cvar90 = np.mean(target_time[target_time >= cvar_threshold])

    strategy_compound, compound_source, compound_prob = compound_from_strategy(strategy, target_row)

    if leader_pass_prob_values:
        mean_leader_attack_pass_probability = float(np.mean(leader_pass_prob_values))
    else:
        mean_leader_attack_pass_probability = 0.0

    return {
        "strategy": strategy,
        "scenario": scenario,
        "current_position": float(safe_scalar(target_row, "position", np.nan)),
        "current_compound": current_compound(target_row),
        "strategy_compound": strategy_compound,
        "compound_source": compound_source,
        "compound_probability": float(compound_prob),
        "n_sim": int(n_sim),

        "mean_projected_race_time": float(np.mean(target_time)),
        "median_projected_race_time": float(np.median(target_time)),
        "p10_projected_race_time": float(np.quantile(target_time, 0.10)),
        "p90_projected_race_time": float(np.quantile(target_time, 0.90)),
        "cvar90_projected_race_time": float(cvar90),

        "mean_finish_position": float(np.mean(finish_position)),
        "median_finish_position": float(np.median(finish_position)),
        "p10_finish_position": float(np.quantile(finish_position, 0.10)),
        "p90_finish_position": float(np.quantile(finish_position, 0.90)),

        "mean_points": float(np.mean(points)),
        "p_points": float(np.mean(points > 0)),
        "p_win": float(np.mean(finish_position == 1)),
        "p_podium": float(np.mean(finish_position <= 3)),
        "p_top5": float(np.mean(finish_position <= 5)),
        "p_top10": float(np.mean(finish_position <= 10)),

        "mean_expected_overtakes": float(np.mean(expected_successful_overtakes)),
        "mean_raw_rivals_ahead_before_overtake_layer": float(np.mean(raw_rivals_ahead_before_overtake_layer)),
        "mean_leader_attack_pass_probability": mean_leader_attack_pass_probability,
        "rival_projection_count": int(rival_projection_count),

        "pit_cost_used": float(target_projection["pit_cost_used"]),
        "pit_cost_source": target_projection["pit_cost_source"],
        "pit_cost_source_lap": target_projection["pit_cost_source_lap"],

        "target_initial_offset_sec": float(target_projection["initial_offset"]),
        "target_expected_lap_time": float(target_projection["expected_lap_time"]),
        "target_attack_factor": float(target_projection["attack_factor"]),
        "target_mean_stint_cost": float(target_projection["mean_stint_cost"]),
        "target_mean_warmup_crossover_cost": float(target_projection["mean_warmup_crossover_cost"]),
        "target_mean_model3_confidence_cost": float(target_projection["mean_model3_confidence_cost"]),
        "target_mean_neutralisation": float(target_projection["mean_neutralisation"]),
        "target_mean_pit_flag": float(target_projection["mean_pit_flag"]),

        "model1_deg_per_lap_used": float(target_projection["model1_deg"]),
        "model1_pace_loss_used": float(target_projection["model1_pace"]),
        "model1_sigma_used": float(target_projection["model1_sigma"]),

        "field_size": int(len(state)),
    }


def choose_final(results: pd.DataFrame, risk_lambda: float, ranking_mode: str) -> pd.DataFrame:
    out = results.copy()

    out["risk_adjusted_time"] = (
        out["mean_projected_race_time"]
        + risk_lambda * out["cvar90_projected_race_time"]
    )

    out["risk_adjusted_position"] = (
        out["mean_finish_position"]
        + risk_lambda * (out["p90_finish_position"] - out["mean_finish_position"])
    )

    out["strategy_score"] = (
        out["risk_adjusted_position"]
        - out["mean_points"] * 0.05
        - out["p_win"] * 0.40
        - out["p_podium"] * 0.15
        - out["mean_expected_overtakes"] * 0.03
    )

    if ranking_mode == "points":
        out = out.sort_values(
            ["mean_points", "p_win", "risk_adjusted_position"],
            ascending=[False, False, True],
        )
    elif ranking_mode == "position":
        out = out.sort_values(
            ["risk_adjusted_position", "mean_points", "p_win"],
            ascending=[True, False, False],
        )
    else:
        out = out.sort_values(
            ["strategy_score", "mean_finish_position", "mean_points"],
            ascending=[True, True, False],
        )

    return out.reset_index(drop=True)


def write_report(path: str, payload: Dict, ranked: pd.DataFrame) -> None:
    lines = []

    lines.append("# Model 5B Final Unified Dry/Wet Scenario Strategy Simulator Report")
    lines.append("")
    lines.append(f"Timestamp: {payload['timestamp']}")
    lines.append(f"Race ID: {payload['query']['race_id']}")
    lines.append(f"Driver ID: {payload['query']['driver_id']}")
    lines.append(f"Requested decision lap: {payload['query']['requested_decision_lap']}")
    lines.append(f"Actual lap used: {payload['query']['actual_lap_used']}")
    lines.append(f"Scenario: {payload['query']['scenario']}")
    lines.append(f"Field size: {payload['query']['field_size']}")
    lines.append(f"Field offset source: {payload['query']['field_offset_source']}")
    lines.append("")

    best = ranked.iloc[0]

    lines.append("## Final best strategy")
    lines.append("")
    lines.append(f"Best strategy: {best['strategy']}")
    lines.append(f"Strategy compound: {best['strategy_compound']}")
    lines.append(f"Mean finish position: {best['mean_finish_position']}")
    lines.append(f"Mean points: {best['mean_points']}")
    lines.append(f"Win probability: {best['p_win']}")
    lines.append(f"Podium probability: {best['p_podium']}")
    lines.append(f"Top 5 probability: {best['p_top5']}")
    lines.append(f"Expected overtakes: {best['mean_expected_overtakes']}")
    lines.append(f"Raw rivals ahead before overtake layer: {best['mean_raw_rivals_ahead_before_overtake_layer']}")
    lines.append(f"Leader attack pass probability: {best['mean_leader_attack_pass_probability']}")
    lines.append(f"Pit cost used: {best['pit_cost_used']}")
    lines.append(f"Pit cost source: {best['pit_cost_source']}")
    lines.append(f"Target stint cost: {best['target_mean_stint_cost']}")
    lines.append(f"Target warmup/crossover cost: {best['target_mean_warmup_crossover_cost']}")
    lines.append(f"Target Model 3 confidence cost: {best['target_mean_model3_confidence_cost']}")
    lines.append("")

    lines.append("## Ranked strategies")
    lines.append("")

    for _, r in ranked.iterrows():
        lines.append(f"### {r['strategy']}")
        lines.append(f"Compound: {r['strategy_compound']}")
        lines.append(f"Mean finish position: {r['mean_finish_position']}")
        lines.append(f"Mean points: {r['mean_points']}")
        lines.append(f"Win probability: {r['p_win']}")
        lines.append(f"Podium probability: {r['p_podium']}")
        lines.append(f"Top 5 probability: {r['p_top5']}")
        lines.append(f"Expected overtakes: {r['mean_expected_overtakes']}")
        lines.append(f"Target stint cost: {r['target_mean_stint_cost']}")
        lines.append(f"Warmup/crossover cost: {r['target_mean_warmup_crossover_cost']}")
        lines.append(f"Model 3 confidence cost: {r['target_mean_model3_confidence_cost']}")
        lines.append(f"Pit cost source: {r['pit_cost_source']}")
        lines.append("")

    lines.append("## Model usage")
    lines.append("")
    lines.append("This final unified Model 5B simulator handles dry, wet, rain, Safety Car, VSC, RF, and no-neutralisation scenarios inside one script.")
    lines.append("It evaluates dry compounds, Intermediate, Wet, pit-now, wait-and-pit, and stay-out strategies.")
    lines.append("The base score comes from Models 1, 2, 3, 4, and 5A. The racing-realism layer prevents physically unrealistic tyre decisions such as using Intermediates too early on a dry track without warm-up or track-position loss.")
    lines.append("")

    lines.append("## Output files")
    lines.append("")
    lines.append(f"Results CSV: `{payload['outputs']['results_csv']}`")
    lines.append(f"Metrics JSON: `{payload['outputs']['metrics_json']}`")
    lines.append(f"Config JSON: `{payload['outputs']['config_json']}`")
    lines.append(f"Report: `{payload['outputs']['report_md']}`")

    with open(path, "w") as f:
        f.write("\n".join(lines))


def main() -> None:
    ap = argparse.ArgumentParser()

    ap.add_argument("--input_path", required=True)
    ap.add_argument("--model5a_strategy_table", required=True)
    ap.add_argument("--outputs_dir", default="outputs")
    ap.add_argument("--race_id", required=True)
    ap.add_argument("--driver_id", required=True)
    ap.add_argument("--decision_lap", type=int, required=True)
    ap.add_argument("--scenario", default="base")

    ap.add_argument(
        "--strategies",
        default="auto",
        help="Use 'auto' for final dry/wet candidate generation, or pass comma-separated custom strategies.",
    )

    ap.add_argument("--n_sim", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--ranking_mode", default="balanced", choices=["balanced", "points", "position"])

    ap.add_argument("--default_lap_time", type=float, default=90.0)
    ap.add_argument("--default_laps_remaining", type=int, default=25)
    ap.add_argument("--default_position_gap_sec", type=float, default=2.0)

    ap.add_argument("--old_tyre_wait_multiplier", type=float, default=1.35)
    ap.add_argument("--tyre_cliff_weight", type=float, default=4.5)

    ap.add_argument("--dry_on_wet_hard_loss_sec", type=float, default=6.0)
    ap.add_argument("--intermediate_on_dry_hard_loss_sec", type=float, default=4.5)
    ap.add_argument("--wet_on_dry_hard_loss_sec", type=float, default=7.5)

    ap.add_argument("--intermediate_warmup_laps", type=int, default=2)
    ap.add_argument("--wet_warmup_laps", type=int, default=3)
    ap.add_argument("--intermediate_warmup_loss_sec", type=float, default=1.2)
    ap.add_argument("--wet_warmup_loss_sec", type=float, default=1.8)

    ap.add_argument("--rain_crossover_delay_laps", type=int, default=1)
    ap.add_argument("--model3_low_confidence_penalty", type=float, default=8.0)
    ap.add_argument("--model3_zero_probability_penalty", type=float, default=14.0)
    ap.add_argument("--early_wet_pit_track_position_penalty", type=float, default=4.0)
    ap.add_argument("--rival_reacts_to_rain", type=int, default=1)

    ap.add_argument("--sc_pit_discount", type=float, default=0.45)
    ap.add_argument("--vsc_pit_discount", type=float, default=0.25)
    ap.add_argument("--rf_pit_discount", type=float, default=0.65)
    ap.add_argument("--neutralisation_race_time_bonus_sec", type=float, default=0.40)

    ap.add_argument("--clean_air_bonus_sec", type=float, default=0.25)
    ap.add_argument("--race_noise_weight", type=float, default=1.0)
    ap.add_argument("--risk_lambda", type=float, default=0.15)

    ap.add_argument("--target_driver_attack_bias", type=float, default=0.08)
    ap.add_argument("--pace_advantage_overtake_weight", type=float, default=0.55)
    ap.add_argument("--overtake_window_sec", type=float, default=1.20)
    ap.add_argument("--overtake_temperature", type=float, default=1.10)
    ap.add_argument("--max_overtake_gap_sec", type=float, default=8.0)

    ap.add_argument("--leader_attack_gap_sec", type=float, default=5.0)
    ap.add_argument("--leader_attack_bonus_sec", type=float, default=1.75)
    ap.add_argument("--leader_attack_extra_gap_sec", type=float, default=3.0)
    ap.add_argument("--leader_attack_floor_probability", type=float, default=0.12)

    args = ap.parse_args()

    tag = now_tag()
    rng = np.random.default_rng(args.seed)

    race_safe = safe_name(args.race_id)
    driver_safe = safe_name(args.driver_id)
    scenario_safe = safe_name(args.scenario)

    run_name = f"{race_safe}_{driver_safe}_L{int(args.decision_lap)}_{scenario_safe}_{tag}"
    run_dir = os.path.join(args.outputs_dir, "model5b_runs", run_name)
    ensure_dir(run_dir)

    full_df = pd.read_csv(args.input_path, low_memory=False)
    strategy_table = pd.read_csv(args.model5a_strategy_table, low_memory=False)

    required_full = ["race_id", "driver_id", "lap_number"]
    required_5a = ["race_id", "driver_id", "lap_number", "action"]

    for c in required_full:
        if c not in full_df.columns:
            raise ValueError(f"Missing column in input_path: {c}")

    for c in required_5a:
        if c not in strategy_table.columns:
            raise ValueError(f"Missing column in model5a_strategy_table: {c}")

    full_df["race_id"] = full_df["race_id"].astype(str)
    full_df["driver_id"] = full_df["driver_id"].astype(str)

    strategy_table["race_id"] = strategy_table["race_id"].astype(str)
    strategy_table["driver_id"] = strategy_table["driver_id"].astype(str)

    state = find_race_state(
        full_df=full_df,
        race_id=args.race_id,
        decision_lap=args.decision_lap,
    )

    state = infer_field_offsets(state, args)

    target_row = get_target_row(state, args.driver_id)
    actual_lap_used = int(pd.to_numeric(pd.Series([target_row["lap_number"]]), errors="coerce").iloc[0])

    fallback = build_fallbacks(full_df, strategy_table)

    strategies = parse_strategy_list(args.strategies, args.scenario)

    rows = []

    for strategy in strategies:
        result = simulate_strategy_field(
            state=state,
            target_row=target_row,
            strategy=strategy,
            strategy_table=strategy_table,
            fallback=fallback,
            scenario=args.scenario,
            race_id=args.race_id,
            decision_lap=actual_lap_used,
            n_sim=args.n_sim,
            rng=rng,
            args=args,
        )
        rows.append(result)

    results = pd.DataFrame(rows)

    ranked = choose_final(
        results=results,
        risk_lambda=args.risk_lambda,
        ranking_mode=args.ranking_mode,
    )

    out_csv = os.path.join(run_dir, "results.csv")
    out_json = os.path.join(run_dir, "metrics.json")
    out_report = os.path.join(run_dir, "report.md")
    out_config = os.path.join(run_dir, "config.json")

    ranked.to_csv(out_csv, index=False)

    field_offset_source = str(state["offset_source"].iloc[0]) if "offset_source" in state.columns else "unknown"

    config_payload = {
        "timestamp": tag,
        "run_directory": run_dir,
        "input_path": args.input_path,
        "model5a_strategy_table": args.model5a_strategy_table,
        "race_id": args.race_id,
        "driver_id": args.driver_id,
        "requested_decision_lap": int(args.decision_lap),
        "actual_lap_used": int(actual_lap_used),
        "scenario": args.scenario,
        "field_size": int(len(state)),
        "field_offset_source": field_offset_source,
        "strategies": strategies,
        "n_sim": int(args.n_sim),
        "seed": int(args.seed),
        "ranking_mode": args.ranking_mode,
        "settings": vars(args),
    }

    payload = {
        "timestamp": tag,
        "mode": "model5b_final_unified_dry_wet_ml_grounded_scenario_simulator",
        "query": {
            "race_id": args.race_id,
            "driver_id": args.driver_id,
            "requested_decision_lap": int(args.decision_lap),
            "actual_lap_used": int(actual_lap_used),
            "scenario": args.scenario,
            "field_size": int(len(state)),
            "field_offset_source": field_offset_source,
            "strategies": strategies,
            "n_sim": int(args.n_sim),
        },
        "inputs": {
            "input_path": args.input_path,
            "model5a_strategy_table": args.model5a_strategy_table,
        },
        "settings": vars(args),
        "fallbacks": {
            "global_deg": float(fallback["global_deg"]),
            "global_pace": float(fallback["global_pace"]),
            "global_sigma": float(fallback["global_sigma"]),
            "global_pit_loss": float(fallback["global_pit_loss"]),
        },
        "best_strategy": ranked.iloc[0].to_dict(),
        "results": ranked.to_dict(orient="records"),
        "outputs": {
            "run_directory": run_dir,
            "results_csv": out_csv,
            "metrics_json": out_json,
            "config_json": out_config,
            "report_md": out_report,
        },
        "notes": [
            "This is the final ML-grounded unified Model 5B scenario simulator.",
            "The base score comes from Model 1 pace/degradation/uncertainty, Model 2 neutralisation, Model 3 pit and compound probabilities, Model 4 clean-air/overtaking context, and Model 5A pit cost.",
            "The realism layer prevents impossible wet tyre behaviour by applying warm-up, crossover, early wet-tyre track-position loss, and Model 3 confidence costs.",
            "Rivals react to rain using the same wet-track logic rather than staying unrealistically on dry tyres.",
            "This output is designed to feed Model 6 risk-aware recommendation.",
        ],
    }

    with open(out_json, "w") as f:
        json.dump(clean_for_json(payload), f, indent=2)

    with open(out_config, "w") as f:
        json.dump(clean_for_json(config_payload), f, indent=2)

    write_report(out_report, payload, ranked)

    print("Run directory:", run_dir)
    print("Saved results:", out_csv)
    print("Saved metrics:", out_json)
    print("Saved config:", out_config)
    print("Saved report:", out_report)
    print("")
    print("Best strategy:", ranked.iloc[0]["strategy"])
    print("Strategy compound:", ranked.iloc[0]["strategy_compound"])
    print("Expected finish:", ranked.iloc[0]["mean_finish_position"])
    print("Expected points:", ranked.iloc[0]["mean_points"])
    print("Win probability:", ranked.iloc[0]["p_win"])
    print("Podium probability:", ranked.iloc[0]["p_podium"])
    print("Top 5 probability:", ranked.iloc[0]["p_top5"])
    print("Expected overtakes:", ranked.iloc[0]["mean_expected_overtakes"])
    print("Raw rivals ahead before overtake layer:", ranked.iloc[0]["mean_raw_rivals_ahead_before_overtake_layer"])
    print("Leader attack pass probability:", ranked.iloc[0]["mean_leader_attack_pass_probability"])
    print("Target stint cost:", ranked.iloc[0]["target_mean_stint_cost"])
    print("Target warmup/crossover cost:", ranked.iloc[0]["target_mean_warmup_crossover_cost"])
    print("Target Model 3 confidence cost:", ranked.iloc[0]["target_mean_model3_confidence_cost"])
    print("Pit cost source:", ranked.iloc[0]["pit_cost_source"])
    print("Pit cost used:", ranked.iloc[0]["pit_cost_used"])
    print("Field offset source:", field_offset_source)


if __name__ == "__main__":
    main()