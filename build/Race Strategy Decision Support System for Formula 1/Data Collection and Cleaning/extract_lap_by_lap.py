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


def _race_id(year: int, round_number: int, session_type: str) -> str:
    return f"{year}_{int(round_number):02d}_{session_type}"


def _map_track_status_flags(track_status_value: str) -> tuple[bool, bool]:
    s = "" if pd.isna(track_status_value) else str(track_status_value)
    is_sc = "4" in s
    is_vsc = ("6" in s) or ("7" in s)
    return is_sc, is_vsc


def extract_lap_by_lap(year: int, round_number: int, config: ExtractConfig = ExtractConfig()) -> pd.DataFrame:
    os.makedirs(config.cache_dir, exist_ok=True)
    fastf1.Cache.enable_cache(config.cache_dir)

    session = fastf1.get_session(year, round_number, config.session_type)
    session.load()

    laps = session.laps.copy()

    # make a safe integer lap number for filtering
    laps["_lap_int"] = pd.to_numeric(laps.get("LapNumber"), errors="coerce").astype("Int64")

    # Cleaning but NEVER drop lap 1
    if config.drop_inaccurate_laps and "IsAccurate" in laps.columns:
        laps = laps[(laps["IsAccurate"] == True) | (laps["_lap_int"] == 1)].copy()

    if config.drop_deleted_laps and "Deleted" in laps.columns:
        laps = laps[(laps["Deleted"] == False) | (laps["_lap_int"] == 1)].copy()

    results = session.results.copy()
    driver_map = results.set_index("Abbreviation")["DriverId"].to_dict() if "DriverId" in results.columns else {}
    team_map = results.set_index("Abbreviation")["TeamId"].to_dict() if "TeamId" in results.columns else {}

    laps["driver_id"] = laps["Driver"].astype(str).map(driver_map)
    laps["team_id"] = laps["Driver"].astype(str).map(team_map)

    # robust lap end time
    lap_end_time = laps["Time"] if "Time" in laps.columns else pd.Series(pd.NaT, index=laps.index)
    if "LapStartTime" in laps.columns and "LapTime" in laps.columns:
        lap_end_time = lap_end_time.fillna(laps["LapStartTime"] + laps["LapTime"])

    df = pd.DataFrame(
        {
            "race_id": _race_id(year, round_number, config.session_type),
            "season": year,
            "lap_number": laps["_lap_int"],
            "driver_id": laps["driver_id"],
            "team_id": laps["team_id"],
            "position": laps["Position"].astype("Int64") if "Position" in laps.columns else pd.NA,
            "lap_time": laps["LapTime"] if "LapTime" in laps.columns else pd.NA,
            "stint_number": laps["Stint"].astype("Int64") if "Stint" in laps.columns else pd.NA,
            "tyre_compound": laps["Compound"] if "Compound" in laps.columns else pd.NA,
            "tyre_age": laps["TyreLife"].astype("Int64") if "TyreLife" in laps.columns else pd.NA,
            "is_out_lap": laps["PitOutTime"].notna() if "PitOutTime" in laps.columns else False,
            "is_in_lap": laps["PitInTime"].notna() if "PitInTime" in laps.columns else False,
            "is_pit_lap": (laps["PitOutTime"].notna() if "PitOutTime" in laps.columns else False)
            | (laps["PitInTime"].notna() if "PitInTime" in laps.columns else False),
            "track_status": laps["TrackStatus"].astype("string") if "TrackStatus" in laps.columns else pd.NA,
            "sector_1_time": laps["Sector1Time"] if "Sector1Time" in laps.columns else pd.NA,
            "sector_2_time": laps["Sector2Time"] if "Sector2Time" in laps.columns else pd.NA,
            "sector_3_time": laps["Sector3Time"] if "Sector3Time" in laps.columns else pd.NA,
            "speed_trap": laps["SpeedST"] if "SpeedST" in laps.columns else pd.NA,
            "_lap_end_time": lap_end_time,
        }
    )

    # do not drop rows based on _lap_end_time
    df["lap_time_delta_to_leader"] = pd.NA
    df["gap_ahead"] = pd.NA
    df["gap_behind"] = pd.NA

    ok = df["lap_number"].notna() & df["position"].notna() & df["_lap_end_time"].notna()
    tmp = df.loc[ok, ["lap_number", "position", "_lap_end_time"]].copy()

    tmp["_leader_end_time"] = tmp.groupby("lap_number")["_lap_end_time"].transform("min")
    tmp["lap_time_delta_to_leader"] = tmp["_lap_end_time"] - tmp["_leader_end_time"]

    tmp = tmp.sort_values(["lap_number", "position", "_lap_end_time"]).copy()
    tmp["_ahead_end_time"] = tmp.groupby("lap_number")["_lap_end_time"].shift(1)
    tmp["_behind_end_time"] = tmp.groupby("lap_number")["_lap_end_time"].shift(-1)
    tmp["gap_ahead"] = tmp["_lap_end_time"] - tmp["_ahead_end_time"]
    tmp["gap_behind"] = tmp["_behind_end_time"] - tmp["_lap_end_time"]

    df.loc[ok, "lap_time_delta_to_leader"] = tmp["lap_time_delta_to_leader"].values
    df.loc[ok, "gap_ahead"] = tmp["gap_ahead"].values
    df.loc[ok, "gap_behind"] = tmp["gap_behind"].values

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

    df = df[final_cols].copy()
    df = df.sort_values(["lap_number", "position", "driver_id"], na_position="last").reset_index(drop=True)
    return df


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--round", type=int, required=True)
    parser.add_argument("--session", type=str, default="R", choices=["R", "S", "Q", "SQ"])
    parser.add_argument("--cache-dir", type=str, default="fastf1_cache")
    parser.add_argument("--out", type=str, default="")
    args = parser.parse_args()

    cfg = ExtractConfig(cache_dir=args.cache_dir, session_type=args.session)
    df = extract_lap_by_lap(args.year, args.round, cfg)

    print(df[df["lap_number"] == 1].head(10))
    print(f"Rows: {len(df):,}")

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
