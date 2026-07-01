# extract_weather_track_conditions.py

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Literal, Optional

import pandas as pd
import fastf1


SessionType = Literal["R", "S"]


@dataclass
class ExtractConfig:
    cache_dir: str = "fastf1_cache"
    session_type: SessionType = "R"


def _race_id(year: int, round_number: int, session_type: str) -> str:
    return f"{year}_{int(round_number):02d}_{session_type}"


def _build_lap_end_times_per_lap(laps: pd.DataFrame) -> pd.DataFrame:
    """
    Builds one timestamp per lap (leader lap end time) to join with weather samples.
    """
    laps = laps.copy()
    laps = laps[laps["LapNumber"].notna()].copy()
    laps["lap_number"] = pd.to_numeric(laps["LapNumber"], errors="coerce").astype("Int64")

    lap_end = None
    if "Time" in laps.columns:
        lap_end = laps["Time"].copy()
    else:
        lap_end = pd.Series(pd.NaT, index=laps.index)

    if "LapStartTime" in laps.columns and "LapTime" in laps.columns:
        lap_end = lap_end.fillna(laps["LapStartTime"] + laps["LapTime"])

    laps["_lap_end_time"] = lap_end
    laps = laps[laps["_lap_end_time"].notna()].copy()

    # leader is minimum lap end time on that lap
    per_lap = (
        laps.groupby("lap_number", as_index=False)["_lap_end_time"]
        .min()
        .sort_values("lap_number")
        .reset_index(drop=True)
    )
    return per_lap


def extract_weather_track_conditions(
    year: int,
    round_number: int,
    config: ExtractConfig = ExtractConfig(),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns two datasets:
    1) lap level weather when available
    2) race level fallback summary
    """
    os.makedirs(config.cache_dir, exist_ok=True)
    fastf1.Cache.enable_cache(config.cache_dir)

    session = fastf1.get_session(year, round_number, config.session_type)
    session.load()

    race_id = _race_id(year, round_number, config.session_type)

    # Weather samples from FastF1
    try:
        w = session.weather_data.copy()
    except Exception:
        w = pd.DataFrame()

    laps = session.laps.copy()
    per_lap_time = _build_lap_end_times_per_lap(laps)

    if w is None or w.empty or "Time" not in w.columns:
        # No lap level weather, produce empty lap table and race level summary
        lap_df = pd.DataFrame(
            columns=[
                "race_id",
                "season",
                "lap_number",
                "track_temperature",
                "air_temperature",
                "rain_flag",
            ]
        )

        race_level = pd.DataFrame(
            {
                "race_id": [race_id],
                "season": [year],
                "race_level_weather_condition": [pd.NA],
            }
        )
        return lap_df, race_level

    w = w.sort_values("Time").reset_index(drop=True)

    # Normalize rainfall into a boolean
    rain_flag = None
    if "Rainfall" in w.columns:
        # sometimes Rainfall is boolean, sometimes numeric
        if pd.api.types.is_bool_dtype(w["Rainfall"]):
            rain_flag = w["Rainfall"].fillna(False).astype(bool)
        else:
            rain_flag = pd.to_numeric(w["Rainfall"], errors="coerce").fillna(0) > 0
    else:
        rain_flag = pd.Series(False, index=w.index)

    w["_rain_flag"] = rain_flag

    # Keep only columns we need if present
    if "TrackTemp" in w.columns:
        w["_track_temp"] = pd.to_numeric(w["TrackTemp"], errors="coerce")
    else:
        w["_track_temp"] = pd.NA

    if "AirTemp" in w.columns:
        w["_air_temp"] = pd.to_numeric(w["AirTemp"], errors="coerce")
    else:
        w["_air_temp"] = pd.NA

    w_small = w[["Time", "_track_temp", "_air_temp", "_rain_flag"]].copy()

    # As of join: for each lap end time take latest weather sample at or before that time
    per_lap_time = per_lap_time.sort_values("_lap_end_time").reset_index(drop=True)

    merged = pd.merge_asof(
        per_lap_time,
        w_small,
        left_on="_lap_end_time",
        right_on="Time",
        direction="backward",
        allow_exact_matches=True,
    )

    lap_df = pd.DataFrame(
        {
            "race_id": race_id,
            "season": year,
            "lap_number": merged["lap_number"].astype("Int64"),
            "track_temperature": merged["_track_temp"],
            "air_temperature": merged["_air_temp"],
            "rain_flag": merged["_rain_flag"].fillna(False).astype(bool),
        }
    )

    # Race level summary fallback
    any_rain = bool(w_small["_rain_flag"].fillna(False).any())
    mean_track = pd.to_numeric(w_small["_track_temp"], errors="coerce").mean()
    mean_air = pd.to_numeric(w_small["_air_temp"], errors="coerce").mean()

    if any_rain:
        cond = "rain"
    else:
        cond = "dry"

    race_level = pd.DataFrame(
        {
            "race_id": [race_id],
            "season": [year],
            "race_level_weather_condition": [cond],
            "avg_track_temperature": [mean_track],
            "avg_air_temperature": [mean_air],
            "any_rain_flag": [any_rain],
        }
    )

    return lap_df, race_level


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--round", type=int, required=True)
    parser.add_argument("--session", type=str, default="R", choices=["R", "S"])
    parser.add_argument("--cache-dir", type=str, default="fastf1_cache")
    parser.add_argument("--out-laps", type=str, default="")
    parser.add_argument("--out-race", type=str, default="")
    args = parser.parse_args()

    cfg = ExtractConfig(cache_dir=args.cache_dir, session_type=args.session)
    lap_df, race_df = extract_weather_track_conditions(args.year, args.round, cfg)

    print(lap_df.head(20))
    print(f"Lap rows: {len(lap_df):,}")
    print(race_df)

    if args.out_laps:
        if args.out_laps.endswith(".csv"):
            lap_df.to_csv(args.out_laps, index=False)
        elif args.out_laps.endswith(".parquet"):
            lap_df.to_parquet(args.out_laps, index=False)
        else:
            raise ValueError("Output must end with .csv or .parquet")

    if args.out_race:
        if args.out_race.endswith(".csv"):
            race_df.to_csv(args.out_race, index=False)
        elif args.out_race.endswith(".parquet"):
            race_df.to_parquet(args.out_race, index=False)
        else:
            raise ValueError("Output must end with .csv or .parquet")


if __name__ == "__main__":
    main()
