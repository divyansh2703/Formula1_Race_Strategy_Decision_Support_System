# extract_pitstops.py

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Literal

import pandas as pd
import fastf1


SessionType = Literal["R", "S"]


@dataclass
class ExtractConfig:
    cache_dir: str = "fastf1_cache"
    session_type: SessionType = "R"
    drop_deleted_laps: bool = True
    drop_inaccurate_laps: bool = True


def _race_id(year: int, round_number: int, session_type: str) -> str:
    return f"{year}_{int(round_number):02d}_{session_type}"


def extract_pit_stops(year: int, round_number: int, config: ExtractConfig = ExtractConfig()) -> pd.DataFrame:
    os.makedirs(config.cache_dir, exist_ok=True)
    fastf1.Cache.enable_cache(config.cache_dir)

    session = fastf1.get_session(year, round_number, config.session_type)
    session.load()

    laps = session.laps.copy()

    # basic cleaning for general lap data
    laps_clean = laps
    if config.drop_inaccurate_laps and "IsAccurate" in laps_clean.columns:
        laps_clean = laps_clean[laps_clean["IsAccurate"] == True].copy()

    if config.drop_deleted_laps and "Deleted" in laps_clean.columns:
        laps_clean = laps_clean[laps_clean["Deleted"] == False].copy()

    # If cleaning removes all pit timestamps, fall back to raw laps for pit stops.
    if (
        ("PitInTime" in laps_clean.columns and laps_clean["PitInTime"].notna().sum() == 0)
        or ("PitOutTime" in laps_clean.columns and laps_clean["PitOutTime"].notna().sum() == 0)
    ):
        laps_for_pit = laps
    else:
        laps_for_pit = laps_clean

    # Driver and team mapping
    results = session.results.copy()
    driver_map = results.set_index("Abbreviation")["DriverId"].to_dict()
    team_map = results.set_index("Abbreviation")["TeamId"].to_dict()

    laps["driver_id"] = laps["Driver"].map(driver_map)
    laps["team_id"] = laps["Driver"].map(team_map)

    # PitInTime is on the lap entering the pits; PitOutTime is on the out lap.
    # For races, the pit out lap is typically the next lap number.
    laps_sorted = laps_for_pit.sort_values(["Driver", "LapNumber"]).copy()

    pit_in = laps_sorted[laps_sorted["PitInTime"].notna()].copy()
    pit_out = laps_sorted[laps_sorted["PitOutTime"].notna()].copy()

    pit_out = pit_out.rename(
        columns={
            "LapNumber": "pit_out_lap",
            "PitOutTime": "PitOutTime",
            "Compound": "Compound_out",
            "TyreLife": "TyreLife_out",
        }
    )[["Driver", "pit_out_lap", "PitOutTime", "Compound_out", "TyreLife_out"]]

    pit_in = pit_in.rename(columns={"LapNumber": "pit_in_lap"})
    pit_in = pit_in[["Driver", "pit_in_lap", "PitInTime", "driver_id", "team_id", "Compound", "TyreLife"]]
    pit_in["pit_out_lap"] = pit_in["pit_in_lap"] + 1

    # Pair each pit-in lap with its next lap for pit-out.
    pit = pit_in.merge(pit_out, on=["Driver", "pit_out_lap"], how="left")
    pit = pit[pit["PitOutTime"].notna()].copy()
    pit["pit_stop_number"] = pit.groupby("Driver").cumcount() + 1

    if "PitTime" in laps.columns:
        pit_lane_time = pit["PitOutTime"] - pit["PitInTime"] if pit["PitOutTime"].notna().any() else pd.NA
    else:
        pit_lane_time = pit["PitOutTime"] - pit["PitInTime"]

    df = pd.DataFrame(
        {
            "race_id": _race_id(year, round_number, config.session_type),
            "season": year,
            "lap_number": pit["pit_in_lap"].astype("Int64"),
            "driver_id": pit["driver_id"],
            "team_id": pit["team_id"],
            "pit_stop_number": pit["pit_stop_number"].astype("Int64"),
            "pit_entry_time": pit["PitInTime"],
            "pit_exit_time": pit["PitOutTime"],
            "pit_lane_time": pit_lane_time,

            # FastF1 does not always provide stationary time directly.
            # If you need stationary_time precisely, you must derive it from detailed telemetry or FIA timing feeds.
            "stationary_time": pd.NA,

            "tyre_compound_before": pit["Compound"],
            "tyre_compound_after": pit["Compound_out"],
            "tyre_age_before": pit["TyreLife"].astype("Int64"),
        }
    )

    return df


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--round", type=int, required=True)
    parser.add_argument("--session", type=str, default="R", choices=["R", "S"])
    parser.add_argument("--cache-dir", type=str, default="fastf1_cache")
    parser.add_argument("--out", type=str, default="")
    args = parser.parse_args()

    cfg = ExtractConfig(cache_dir=args.cache_dir, session_type=args.session)
    df = extract_pit_stops(args.year, args.round, cfg)

    print(df.head(20))
    print(f"Rows: {len(df):,}")

    if args.out:
        if args.out.endswith(".csv"):
            df.to_csv(args.out, index=False)
        elif args.out.endswith(".parquet"):
            df.to_parquet(args.out, index=False)
        else:
            raise ValueError("Output must end with .csv or .parquet")
        print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
