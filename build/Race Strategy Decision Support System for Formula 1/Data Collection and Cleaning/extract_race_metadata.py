# extract_race_metadata.py

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Optional

import pandas as pd
import fastf1


@dataclass
class ExtractConfig:
    cache_dir: str = "fastf1_cache"
    session_type: str = "R"
    drop_deleted_laps: bool = True
    drop_inaccurate_laps: bool = True


def _race_id(year: int, round_number: int, session_type: str) -> str:
    return f"{year}_{int(round_number):02d}_{session_type}"


def _slug(s: str) -> str:
    s = "" if s is None else str(s)
    s = s.lower().strip()
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^a-z0-9_]", "", s)
    return s


def _has_safety_car(track_status_value: object) -> bool:
    s = "" if pd.isna(track_status_value) else str(track_status_value)
    return "4" in s


def _estimate_track_length_km(session: fastf1.core.Session) -> Optional[float]:
    try:
        fastest = session.laps.pick_fastest()
        if fastest is None:
            return None
        tel = fastest.get_telemetry()
        if tel is None or "Distance" not in tel.columns or tel["Distance"].dropna().empty:
            return None
        length_m = float(tel["Distance"].max())
        if length_m <= 0:
            return None
        return length_m / 1000.0
    except Exception:
        return None


def _estimate_pit_lane_time_loss(laps: pd.DataFrame) -> pd.Timedelta | pd.NaT:
    needed = {"Driver", "LapNumber", "LapTime", "PitInTime", "PitOutTime", "TrackStatus"}
    if not needed.issubset(set(laps.columns)):
        return pd.NaT

    clean = laps[
        laps["PitInTime"].isna()
        & laps["PitOutTime"].isna()
        & laps["LapTime"].notna()
        & (~laps["TrackStatus"].apply(_has_safety_car))
    ].copy()

    if clean.empty:
        return pd.NaT

    clean_median = clean.groupby("Driver")["LapTime"].median()

    laps_idx = laps.set_index(["Driver", "LapNumber"], drop=False)

    pit_in = laps[laps["PitInTime"].notna()].copy()
    if pit_in.empty:
        return pd.NaT

    losses = []
    for _, row in pit_in.iterrows():
        drv = row["Driver"]
        lapn = int(row["LapNumber"])
        base = clean_median.get(drv, pd.NaT)
        if pd.isna(base):
            continue

        pit_out_row = laps_idx.loc[(drv, lapn + 1)] if (drv, lapn + 1) in laps_idx.index else None
        if pit_out_row is None or pd.isna(pit_out_row["PitOutTime"]):
            continue

        pit_time = pit_out_row["PitOutTime"] - row["PitInTime"]
        if pd.isna(pit_time):
            pit_time = pd.Timedelta(0)

        inlap_time = row["LapTime"] if pd.notna(row["LapTime"]) else pd.NaT

        outlap_time = (
            pit_out_row["LapTime"] if pit_out_row is not None and pd.notna(pit_out_row["LapTime"]) else pd.NaT
        )

        inloss = inlap_time - base if pd.notna(inlap_time) else pd.Timedelta(0)
        outloss = outlap_time - base if pd.notna(outlap_time) else pd.Timedelta(0)

        total_loss = pit_time + inloss + outloss
        if pd.notna(total_loss) and total_loss > pd.Timedelta(0):
            losses.append(total_loss)

    if not losses:
        return pd.NaT

    return pd.Series(losses).median()


def _compute_sc_probability_baseline(
    year_from: int,
    year_to: int,
    round_number: int,
    circuit_key: str,
    config: ExtractConfig,
) -> Optional[float]:
    if year_to < year_from:
        return None

    values = []
    for y in range(year_from, year_to + 1):
        try:
            s = fastf1.get_session(y, round_number, config.session_type)
            s.load()
            ev = s.event
            key = _slug(ev.get("Location", "") or ev.get("EventName", ""))
            if key != circuit_key:
                continue

            ls = s.laps
            if ls is None or ls.empty or "TrackStatus" not in ls.columns:
                continue

            any_sc = bool(ls["TrackStatus"].apply(_has_safety_car).any())
            values.append(1.0 if any_sc else 0.0)
        except Exception:
            continue

    if not values:
        # Fallback: compute baseline using the same round only.
        for y in range(year_from, year_to + 1):
            try:
                s = fastf1.get_session(y, round_number, config.session_type)
                s.load()
                ls = s.laps
                if ls is None or ls.empty or "TrackStatus" not in ls.columns:
                    continue
                any_sc = bool(ls["TrackStatus"].apply(_has_safety_car).any())
                values.append(1.0 if any_sc else 0.0)
            except Exception:
                continue

    if not values:
        return 0.0
    return float(sum(values) / len(values))


def _estimate_overtaking_difficulty(laps: pd.DataFrame) -> Optional[float]:
    needed = {"Driver", "LapNumber", "Position"}
    if not needed.issubset(set(laps.columns)):
        return None

    use = laps[laps["Position"].notna()].copy()
    if use.empty:
        return None

    use = use.sort_values(["Driver", "LapNumber"])
    use["pos_change"] = use.groupby("Driver")["Position"].diff().abs()
    change_rate = (use["pos_change"] > 0).mean()
    if pd.isna(change_rate):
        return None

    # Higher position change rate implies easier overtaking.
    difficulty = 1.0 - float(change_rate)
    if difficulty < 0.0:
        difficulty = 0.0
    if difficulty > 1.0:
        difficulty = 1.0
    return difficulty


def extract_race_metadata(
    year: int,
    round_number: int,
    baseline_year_from: Optional[int] = None,
    baseline_year_to: Optional[int] = None,
    config: ExtractConfig = ExtractConfig(),
) -> pd.DataFrame:
    os.makedirs(config.cache_dir, exist_ok=True)
    fastf1.Cache.enable_cache(config.cache_dir)

    session = fastf1.get_session(year, round_number, config.session_type)
    session.load()

    laps = session.laps.copy()

    laps_clean = laps
    if config.drop_inaccurate_laps and "IsAccurate" in laps_clean.columns:
        laps_clean = laps_clean[laps_clean["IsAccurate"] == True].copy()

    if config.drop_deleted_laps and "Deleted" in laps_clean.columns:
        laps_clean = laps_clean[laps_clean["Deleted"] == False].copy()

    # Pit timestamps can be removed by accuracy filters; use raw laps if that happens.
    if (
        ("PitInTime" in laps_clean.columns and laps_clean["PitInTime"].notna().sum() == 0)
        or ("PitOutTime" in laps_clean.columns and laps_clean["PitOutTime"].notna().sum() == 0)
    ):
        laps_for_pit = laps
    else:
        laps_for_pit = laps_clean

    ev = session.event

    circuit_name = ev.get("Location", "") or ev.get("EventName", "")
    country = ev.get("Country", "")
    circuit_id = _slug(circuit_name)

    race_date = ev.get("EventDate", None)
    if race_date is not None:
        race_date = pd.to_datetime(race_date).date()

    total_laps = (
        int(laps_clean["LapNumber"].max())
        if not laps_clean.empty and laps_clean["LapNumber"].notna().any()
        else None
    )

    pit_lane_time_loss = _estimate_pit_lane_time_loss(laps_for_pit)

    track_length = _estimate_track_length_km(session)

    safety_prob = None
    if baseline_year_from is not None and baseline_year_to is not None:
        safety_prob = _compute_sc_probability_baseline(
            baseline_year_from,
            baseline_year_to,
            round_number,
            circuit_id,
            config,
        )

    overtaking_index = _estimate_overtaking_difficulty(laps_clean)

    df = pd.DataFrame(
        [
            {
                "race_id": _race_id(year, round_number, config.session_type),
                "season": year,
                "round": int(round_number),
                "circuit_id": circuit_id,
                "circuit_name": circuit_name,
                "country": country,
                "race_date": race_date,
                "total_laps": total_laps,
                "pit_lane_time_loss": pit_lane_time_loss,
                "safety_car_probability_baseline": safety_prob,
                "track_length": track_length,
                "overtaking_difficulty_index": overtaking_index,
            }
        ]
    )

    return df


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--round", type=int, required=True)
    parser.add_argument("--cache-dir", type=str, default="fastf1_cache")
    parser.add_argument("--out", type=str, default="")
    parser.add_argument("--baseline-from", type=int, default=None)
    parser.add_argument("--baseline-to", type=int, default=None)
    args = parser.parse_args()

    cfg = ExtractConfig(cache_dir=args.cache_dir, session_type="R")
    df = extract_race_metadata(
        args.year,
        args.round,
        baseline_year_from=args.baseline_from,
        baseline_year_to=args.baseline_to,
        config=cfg,
    )

    print(df)

    if args.out:
        if args.out.endswith(".csv"):
            df.to_csv(args.out, index=False)
        elif args.out.endswith(".parquet"):
            df.to_parquet(args.out, index=False)
        else:
            raise ValueError("Output must end with .csv or .parquet")


if __name__ == "__main__":
    main()
