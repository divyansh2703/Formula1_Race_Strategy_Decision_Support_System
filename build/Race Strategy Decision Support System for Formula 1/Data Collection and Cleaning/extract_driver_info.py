# extract_driver_info.py

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Literal, Optional

import pandas as pd
import fastf1


SessionType = Literal["R", "S", "Q", "SQ"]


@dataclass
class ExtractConfig:
    cache_dir: str = "fastf1_cache"
    session_type: SessionType = "R"


def _race_id(year: int, round_number: int, session_type: str) -> str:
    return f"{year}_{int(round_number):02d}_{session_type}"


def _norm_team_id(team_name: Optional[str]) -> str:
    if team_name is None or (isinstance(team_name, float) and pd.isna(team_name)):
        return ""
    s = str(team_name).strip().lower()
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^a-z0-9_]+", "", s)
    return s


def extract_driver_information(
    year: int,
    round_number: int,
    config: ExtractConfig = ExtractConfig(),
) -> pd.DataFrame:
    os.makedirs(config.cache_dir, exist_ok=True)
    fastf1.Cache.enable_cache(config.cache_dir)

    session = fastf1.get_session(year, round_number, config.session_type)
    session.load()

    res = session.results.copy()

    # Driver id and team id
    if "DriverId" in res.columns:
        res["driver_id"] = res["DriverId"].astype(str)
    elif "Abbreviation" in res.columns:
        res["driver_id"] = res["Abbreviation"].astype(str).str.lower()
    else:
        res["driver_id"] = pd.NA

    if "TeamId" in res.columns:
        res["team_id"] = res["TeamId"].astype(str)
    elif "TeamName" in res.columns:
        res["team_id"] = res["TeamName"].astype(str).apply(_norm_team_id)
    else:
        res["team_id"] = pd.NA

    # Names
    if "FullName" in res.columns:
        res["driver_name"] = res["FullName"].astype(str)
    elif "GivenName" in res.columns and "FamilyName" in res.columns:
        res["driver_name"] = (res["GivenName"].astype(str) + " " + res["FamilyName"].astype(str)).str.strip()
    elif "Abbreviation" in res.columns:
        res["driver_name"] = res["Abbreviation"].astype(str)
    else:
        res["driver_name"] = pd.NA

    if "TeamName" in res.columns:
        res["team_name"] = res["TeamName"].astype(str)
    elif "ConstructorName" in res.columns:
        res["team_name"] = res["ConstructorName"].astype(str)
    else:
        res["team_name"] = pd.NA

    # Driver number
    if "DriverNumber" in res.columns:
        res["driver_number"] = pd.to_numeric(res["DriverNumber"], errors="coerce").astype("Int64")
    else:
        res["driver_number"] = pd.NA

    out = pd.DataFrame(
        {
            "driver_id": res["driver_id"],
            "driver_name": res["driver_name"],
            "team_id": res["team_id"],
            "team_name": res["team_name"],
            "season": year,
            "driver_number": res["driver_number"],
            "experience_years": pd.NA,  # optional, keep empty unless you compute it separately
        }
    )

    out = out.drop_duplicates(subset=["season", "driver_id"]).reset_index(drop=True)
    return out


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
    df = extract_driver_information(args.year, args.round, cfg)

    print(df.head(30))
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
