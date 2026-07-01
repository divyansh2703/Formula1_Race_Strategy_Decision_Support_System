from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Literal

import pandas as pd
import fastf1


SessionType = Literal["R", "S", "Q", "SQ"]


@dataclass
class ExtractConfig:
    cache_dir: str = "fastf1_cache"
    session_type: SessionType = "R"
    drop_inaccurate_laps: bool = True
    drop_deleted_laps: bool = True
    keep_pit_laps_even_if_inaccurate: bool = True
    keep_lap1_even_if_inaccurate: bool = True
    fill_missing_lap_end_time: bool = True


def _race_id(year: int, round_number: int, session_type: str) -> str:
    return f"{year}_{int(round_number):02d}_{session_type}"


def _map_track_status_flags(track_status_value: str) -> tuple[bool, bool]:
    s = "" if pd.isna(track_status_value) else str(track_status_value)
    is_sc = "4" in s
    is_vsc = ("6" in s) or ("7" in s)
    return is_sc, is_vsc


def _detect_pit_laps(laps: pd.DataFrame) -> pd.Series:
    pit_mask = pd.Series(False, index=laps.index)
    for col in ["PitInTime", "PitOutTime", "PitTime"]:
        if col in laps.columns:
            pit_mask = pit_mask | laps[col].notna()
    return pit_mask


def _build_lap_end_time(laps: pd.DataFrame, *, fill_missing: bool) -> pd.Series:
    # Primary: FastF1 session time at end of lap
    if "Time" in laps.columns:
        lap_end_time = laps["Time"].copy()
    else:
        lap_end_time = pd.Series(pd.NaT, index=laps.index)

    if not fill_missing:
        return lap_end_time

    # Fallback 1: LapStartTime + LapTime
    if "LapStartTime" in laps.columns and "LapTime" in laps.columns:
        fallback_end = laps["LapStartTime"] + laps["LapTime"]
        lap_end_time = lap_end_time.fillna(fallback_end)

    # Fallback 2: per driver cumulative lap time (fixes Lap 1 missing Time/LapStartTime)
    if "Driver" in laps.columns and "LapTime" in laps.columns:
        lap_time_safe = laps["LapTime"].copy()
        lap_time_safe = lap_time_safe.fillna(pd.Timedelta(0))
        cum_end = lap_time_safe.groupby(laps["Driver"]).cumsum()
        lap_end_time = lap_end_time.fillna(cum_end)

    return lap_end_time


def extract_lap_by_lap(
    year: int,
    round_number: int,
    config: ExtractConfig = ExtractConfig(),
) -> pd.DataFrame:
    os.makedirs(config.cache_dir, exist_ok=True)
    fastf1.Cache.enable_cache(config.cache_dir)

    session = fastf1.get_session(year, round_number, config.session_type)
    session.load()

    laps = session.laps.copy()

    pit_mask = _detect_pit_laps(laps)

    # Cleaning but keep pit laps and ALWAYS keep Lap 1
    if config.drop_inaccurate_laps and "IsAccurate" in laps.columns:
        if config.keep_pit_laps_even_if_inaccurate:
            lap1_mask = (laps["LapNumber"] == 1) if ("LapNumber" in laps.columns and config.keep_lap1_even_if_inaccurate) else False
            laps = laps[(laps["IsAccurate"] == True) | (pit_mask) | (lap1_mask)].copy()
        else:
            laps = laps[laps["IsAccurate"] == True].copy()

    # Deleted laps filter (also keep pit laps and lap 1)
    if config.drop_deleted_laps and "Deleted" in laps.columns:
        pit_mask = _detect_pit_laps(laps)
        lap1_mask = (laps["LapNumber"] == 1) if ("LapNumber" in laps.columns and config.keep_lap1_even_if_inaccurate) else False
        laps = laps[(laps["Deleted"] == False) | (pit_mask) | (lap1_mask)].copy()

    # Update pit mask after filtering
    pit_mask = _detect_pit_laps(laps)

    # Map driver_id and team_id from session.results (Driver is TLA like VER)
    results = session.results.copy()
    driver_map = results.set_index("Abbreviation")["DriverId"].to_dict()
    team_map = results.set_index("Abbreviation")["TeamId"].to_dict()

    laps["driver_id"] = laps["Driver"].map(driver_map)
    laps["team_id"] = laps["Driver"].map(team_map)

    lap_end_time = _build_lap_end_time(laps, fill_missing=config.fill_missing_lap_end_time)

    def col_or_na(col_name: str, dtype=None):
        if col_name in laps.columns:
            return laps[col_name]
        return pd.Series(pd.NA, index=laps.index, dtype=dtype)

    df = pd.DataFrame(
        {
            "race_id": _race_id(year, round_number, config.session_type),
            "season": year,
            "lap_number": col_or_na("LapNumber", "Int64").astype("Int64"),
            "driver_id": laps["driver_id"],
            "team_id": laps["team_id"],
            "position": col_or_na("Position", "Int64").astype("Int64"),
            "lap_time": col_or_na("LapTime"),
            "stint_number": col_or_na("Stint", "Int64").astype("Int64"),
            "tyre_compound": col_or_na("Compound", "string").astype("string"),
            "tyre_age": col_or_na("TyreLife", "Int64").astype("Int64"),
            "is_out_lap": (laps["PitOutTime"].notna() if "PitOutTime" in laps.columns else False),
            "is_in_lap": (laps["PitInTime"].notna() if "PitInTime" in laps.columns else False),
            "is_pit_lap": pit_mask,
            "track_status": col_or_na("TrackStatus", "string").astype("string"),
            "sector_1_time": col_or_na("Sector1Time"),
            "sector_2_time": col_or_na("Sector2Time"),
            "sector_3_time": col_or_na("Sector3Time"),
            "speed_trap": col_or_na("SpeedST"),
            "_lap_end_time": lap_end_time,
        }
    )

    # Keep laps only if we can compute some end time
    df = df[df["_lap_end_time"].notna()].copy()

    # Leader lap end time per lap
    df["_leader_end_time"] = df.groupby("lap_number")["_lap_end_time"].transform("min")
    df["lap_time_delta_to_leader"] = df["_lap_end_time"] - df["_leader_end_time"]

    # Use position if present, otherwise infer order from lap end time
    df["_pos_for_gap"] = df["position"]
    missing_pos = df["_pos_for_gap"].isna()
    if missing_pos.any():
        df.loc[missing_pos, "_pos_for_gap"] = (
            df.loc[missing_pos]
            .groupby("lap_number")["_lap_end_time"]
            .rank(method="first")
            .astype("Int64")
        )

    # Gap ahead/behind by lap based on order
    df = df.sort_values(["lap_number", "_pos_for_gap", "_lap_end_time"]).copy()
    df["_ahead_end_time"] = df.groupby("lap_number")["_lap_end_time"].shift(1)
    df["_behind_end_time"] = df.groupby("lap_number")["_lap_end_time"].shift(-1)

    df["gap_ahead"] = df["_lap_end_time"] - df["_ahead_end_time"]
    df["gap_behind"] = df["_behind_end_time"] - df["_lap_end_time"]

    # Safety car flags derived from TrackStatus
    sc_flags = df["track_status"].apply(_map_track_status_flags)
    df["is_under_safety_car"] = sc_flags.apply(lambda x: x[0])
    df["is_under_virtual_safety_car"] = sc_flags.apply(lambda x: x[1])

    final_cols = [
        "race_id",
        "season",
        "lap_number",
        "driver_id",
        "team_id",
        "position",
        "lap_time",
        "lap_time_delta_to_leader",
        "gap_ahead",
        "gap_behind",
        "stint_number",
        "tyre_compound",
        "tyre_age",
        "is_pit_lap",
        "is_out_lap",
        "is_in_lap",
        "track_status",
        "sector_1_time",
        "sector_2_time",
        "sector_3_time",
        "speed_trap",
        "is_under_safety_car",
        "is_under_virtual_safety_car",
    ]

    return df[final_cols].copy()


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--round", type=int, required=True)
    parser.add_argument("--session", type=str, default="R", choices=["R", "S", "Q", "SQ"])
    parser.add_argument("--cache-dir", type=str, default="fastf1_cache")
    parser.add_argument("--pitstops-file", type=str, default="", help="Accepted for compatibility, not used here")
    parser.add_argument("--out", type=str, default="")
    args = parser.parse_args()

    cfg = ExtractConfig(cache_dir=args.cache_dir, session_type=args.session)
    df = extract_lap_by_lap(args.year, args.round, cfg)

    print(df.head(10))
    print(f"Rows: {len(df):,}")
    print(
        f"Lap 1 rows: {(df['lap_number'] == 1).sum()}  "
        f"Distinct laps: {df['lap_number'].nunique()}  "
        f"Max lap: {int(df['lap_number'].max())}"
    )

    if args.out:
        if args.out.endswith(".parquet"):
            df.to_parquet(args.out, index=False)
        elif args.out.endswith(".csv"):
            df.to_csv(args.out, index=False)
        else:
            raise ValueError("Output must end with .parquet or .csv")
        print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
