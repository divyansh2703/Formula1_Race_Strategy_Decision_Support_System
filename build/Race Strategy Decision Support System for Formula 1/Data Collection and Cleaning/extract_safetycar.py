# extract_race_events.py
# Outputs ONLY: safety_car and virtual_safety_car intervals (no red_flag)

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
    drop_deleted_laps: bool = True
    drop_inaccurate_laps: bool = True


def _race_id(year: int, round_number: int, session_type: str) -> str:
    return f"{year}_{int(round_number):02d}_{session_type}"


def _parse_track_status_codes(v: object) -> set[str]:
    if pd.isna(v):
        return set()
    s = str(v)
    return {c for c in s if c.isdigit()}


def _event_type_from_codes(codes: set[str]) -> Optional[str]:
    # FastF1 TrackStatus codes used here
    # 4 = Safety Car
    # 6 = VSC deployed
    # 7 = VSC ending (still treat as VSC related)
    if "4" in codes:
        return "safety_car"
    if "6" in codes or "7" in codes:
        return "virtual_safety_car"
    return None


def _union_codes(series: pd.Series) -> set[str]:
    out: set[str] = set()
    for v in series.dropna():
        out |= _parse_track_status_codes(v)
    return out


def _extract_intervals_from_laps(laps: pd.DataFrame) -> pd.DataFrame:
    """
    TrackStatus exists per driver per lap row, so aggregate to ONE signal per lap
    (union across drivers) and convert to continuous intervals.
    """
    if laps.empty or "LapNumber" not in laps.columns or "TrackStatus" not in laps.columns:
        return pd.DataFrame(columns=["lap_start", "lap_end", "event_type", "cause"])

    per_lap = (
        laps[laps["LapNumber"].notna()][["LapNumber", "TrackStatus"]]
        .assign(lap=lambda d: d["LapNumber"].astype(int))
        .groupby("lap", as_index=False)["TrackStatus"]
        .agg(_union_codes)
        .rename(columns={"TrackStatus": "codes"})
        .sort_values("lap")
        .reset_index(drop=True)
    )

    per_lap["event_type"] = per_lap["codes"].apply(_event_type_from_codes)

    intervals = []
    current_type = None
    start_lap = None

    for _, r in per_lap.iterrows():
        lap = int(r["lap"])
        et = r["event_type"]

        if current_type is None:
            if et is not None:
                current_type = et
                start_lap = lap
            continue

        if et == current_type:
            continue

        intervals.append(
            {"lap_start": start_lap, "lap_end": lap - 1, "event_type": current_type, "cause": pd.NA}
        )
        current_type = None
        start_lap = None

        if et is not None:
            current_type = et
            start_lap = lap

    if current_type is not None and start_lap is not None:
        last_lap = int(per_lap["lap"].max())
        intervals.append({"lap_start": start_lap, "lap_end": last_lap, "event_type": current_type, "cause": pd.NA})

    if not intervals:
        return pd.DataFrame(columns=["lap_start", "lap_end", "event_type", "cause"])
    return pd.DataFrame(intervals)


def extract_race_events(year: int, round_number: int, config: ExtractConfig = ExtractConfig()) -> pd.DataFrame:
    os.makedirs(config.cache_dir, exist_ok=True)
    fastf1.Cache.enable_cache(config.cache_dir)

    session = fastf1.get_session(year, round_number, config.session_type)
    session.load()

    laps = session.laps.copy()

    if config.drop_inaccurate_laps and "IsAccurate" in laps.columns:
        laps = laps[laps["IsAccurate"] == True].copy()

    if config.drop_deleted_laps and "Deleted" in laps.columns:
        laps = laps[laps["Deleted"] == False].copy()

    all_events = _extract_intervals_from_laps(laps)

    all_events.insert(0, "season", year)
    all_events.insert(0, "race_id", _race_id(year, round_number, config.session_type))

    all_events = all_events[["race_id", "season", "lap_start", "lap_end", "event_type", "cause"]].copy()

    all_events["lap_start_sort"] = pd.to_numeric(all_events["lap_start"], errors="coerce")
    all_events = all_events.sort_values(["lap_start_sort", "event_type"]).drop(columns=["lap_start_sort"])

    return all_events


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
    df = extract_race_events(args.year, args.round, cfg)

    print(df.head(80))
    print(f"Rows: {len(df):,}")

    if args.out:
        if args.out.endswith(".csv"):
            df.to_csv(args.out, index=False)
        elif args.out.endswith(".parquet"):
            df.to_parquet(args.out, index=False)
        else:
            raise ValueError("Output must end with .csv or .parquet")


if __name__ == "__main__":
    main()
