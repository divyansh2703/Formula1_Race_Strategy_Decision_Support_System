from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Literal, Optional

import pandas as pd
import fastf1


SessionType = Literal["R", "S", "Q", "SQ"]
EventType = Literal["safety_car", "virtual_safety_car", "red_flag"]


@dataclass
class ExtractConfig:
    cache_dir: str = "fastf1_cache"
    session_type: SessionType = "R"
    include_trackstatus_fallback: bool = True


def _race_id(year: int, round_number: int, session_type: str) -> str:
    return f"{year}_{int(round_number):02d}_{session_type}"


def _is_code_present(track_status_value: object, code: str) -> bool:
    if pd.isna(track_status_value):
        return False
    return code in str(track_status_value)


def _collapse_bool_segments(lap_numbers: pd.Series, mask: pd.Series) -> list[tuple[int, int]]:
    df = pd.DataFrame({"lap": lap_numbers.astype("Int64"), "m": mask.astype(bool)}).dropna(subset=["lap"])
    df = df.sort_values("lap")
    laps = df["lap"].astype(int).to_list()
    ms = df["m"].to_list()

    segments: list[tuple[int, int]] = []
    start: Optional[int] = None

    for i, (lap, on) in enumerate(zip(laps, ms)):
        if on and start is None:
            start = lap

        if start is not None:
            is_last = i == (len(laps) - 1)
            next_lap = None if is_last else laps[i + 1]
            next_on = None if is_last else ms[i + 1]

            if is_last or (not next_on) or (next_lap != lap + 1):
                segments.append((start, lap))
                start = None

    return segments


def _normalize_text_series(s: pd.Series) -> pd.Series:
    # Works for a full Series safely
    return (
        s.astype("string")
        .fillna("")
        .str.strip()
        .str.lower()
    )


def _events_from_race_control_messages(rcm: pd.DataFrame) -> pd.DataFrame:
    if rcm is None or len(rcm) == 0:
        return pd.DataFrame(columns=["lap_start", "lap_end", "event_type", "cause"])

    cols = {c.lower(): c for c in rcm.columns}

    def pick(*names: str) -> Optional[str]:
        for n in names:
            if n in cols:
                return cols[n]
        return None

    lap_col = pick("lap", "lapnumber", "lap_number")
    msg_col = pick("message", "msg", "text")
    cat_col = pick("category", "flag", "type")
    status_col = pick("status")

    if lap_col is None:
        return pd.DataFrame(columns=["lap_start", "lap_end", "event_type", "cause"])

    df = rcm.copy()
    df["lap"] = pd.to_numeric(df[lap_col], errors="coerce").astype("Int64")
    df = df.dropna(subset=["lap"]).copy()
    df["lap"] = df["lap"].astype(int)

    df["message_text"] = _normalize_text_series(df[msg_col]) if msg_col else ""
    df["category_text"] = _normalize_text_series(df[cat_col]) if cat_col else ""
    df["status_text"] = _normalize_text_series(df[status_col]) if status_col else ""

    def classify_row(message_text: str, category_text: str, status_text: str) -> Optional[EventType]:
        t = f"{message_text} {category_text} {status_text}".strip()

        if "red flag" in t or "redflag" in t:
            return "red_flag"

        # VSC first so it does not get misread as "safety car"
        if "virtual safety car" in t or "vsc" in t:
            return "virtual_safety_car"

        if "safety car" in t or "sc deployed" in t:
            return "safety_car"

        return None

    df["event_type"] = [
        classify_row(m, c, s) for m, c, s in zip(df["message_text"], df["category_text"], df["status_text"])
    ]

    df = df[df["event_type"].notna()].copy()

    def is_start(t: str) -> bool:
        return any(
            k in t
            for k in [
                "deployed",
                "deployment",
                "started",
                "begin",
                "in this lap",
                "red flag",
                "safety car deployed",
                "virtual safety car deployed",
            ]
        )

    def is_end(t: str) -> bool:
        return any(
            k in t
            for k in [
                "ending",
                "ended",
                "withdrawn",
                "rescinded",
                "no longer",
                "green flag",
                "track clear",
            ]
        )

    df["is_start"] = df["message_text"].apply(is_start)
    df["is_end"] = df["message_text"].apply(is_end)

    out_rows = []

    for et in ["safety_car", "virtual_safety_car", "red_flag"]:
        sub = df[df["event_type"] == et].sort_values("lap").copy()
        if sub.empty:
            continue

        active_start: Optional[int] = None
        active_cause: str = ""

        for _, r in sub.iterrows():
            lap = int(r["lap"])
            msg = str(r["message_text"])

            if r["is_start"] and active_start is None:
                active_start = lap
                active_cause = msg
                continue

            if r["is_end"] and active_start is not None:
                out_rows.append(
                    {"lap_start": active_start, "lap_end": lap, "event_type": et, "cause": active_cause}
                )
                active_start = None
                active_cause = ""
                continue

            if active_start is None and (not r["is_start"]) and (not r["is_end"]):
                out_rows.append({"lap_start": lap, "lap_end": lap, "event_type": et, "cause": msg})

        if active_start is not None:
            last_lap = int(sub["lap"].max())
            out_rows.append(
                {"lap_start": active_start, "lap_end": last_lap, "event_type": et, "cause": active_cause}
            )

    out = pd.DataFrame(out_rows)
    if out.empty:
        return out

    out["cause"] = out["cause"].fillna("").astype(str)
    return out


def _events_from_trackstatus(laps: pd.DataFrame) -> pd.DataFrame:
    if laps is None or len(laps) == 0 or "LapNumber" not in laps.columns:
        return pd.DataFrame(columns=["lap_start", "lap_end", "event_type", "cause"])

    lap_numbers = pd.to_numeric(laps["LapNumber"], errors="coerce").astype("Int64")
    ts = laps["TrackStatus"] if "TrackStatus" in laps.columns else pd.Series(pd.NA, index=laps.index)

    sc_mask = ts.apply(lambda x: _is_code_present(x, "4"))
    vsc_mask = ts.apply(lambda x: _is_code_present(x, "6") or _is_code_present(x, "7"))

    sc_segments = _collapse_bool_segments(lap_numbers, sc_mask)
    vsc_segments = _collapse_bool_segments(lap_numbers, vsc_mask)

    rows = []
    for a, b in sc_segments:
        rows.append({"lap_start": a, "lap_end": b, "event_type": "safety_car", "cause": ""})
    for a, b in vsc_segments:
        rows.append({"lap_start": a, "lap_end": b, "event_type": "virtual_safety_car", "cause": ""})

    return pd.DataFrame(rows)


def _dedupe_segments(events: pd.DataFrame) -> pd.DataFrame:
    if events.empty:
        return events

    events = events.copy()
    events["lap_start"] = events["lap_start"].astype(int)
    events["lap_end"] = events["lap_end"].astype(int)

    events = events.sort_values(["event_type", "lap_start", "lap_end"]).drop_duplicates(
        subset=["event_type", "lap_start", "lap_end"], keep="first"
    )
    return events


def extract_race_events(year: int, round_number: int, config: ExtractConfig = ExtractConfig()) -> pd.DataFrame:
    os.makedirs(config.cache_dir, exist_ok=True)
    fastf1.Cache.enable_cache(config.cache_dir)

    session = fastf1.get_session(year, round_number, config.session_type)
    session.load()

    race_id = _race_id(year, round_number, config.session_type)

    rcm = getattr(session, "race_control_messages", None)
    events_rcm = _events_from_race_control_messages(rcm)

    events_ts = pd.DataFrame()
    if config.include_trackstatus_fallback:
        events_ts = _events_from_trackstatus(session.laps.copy())

    events = pd.concat([events_rcm, events_ts], ignore_index=True)
    events = _dedupe_segments(events)

    if events.empty:
        return pd.DataFrame(columns=["race_id", "season", "lap_start", "lap_end", "event_type", "cause"])

    out = events.copy()
    out.insert(0, "race_id", race_id)
    out.insert(1, "season", year)

    out["event_type"] = out["event_type"].astype(str).str.lower()
    out = out[["race_id", "season", "lap_start", "lap_end", "event_type", "cause"]]
    out["cause"] = out["cause"].replace("", pd.NA)

    return out


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--round", type=int, required=True)
    parser.add_argument("--session", type=str, default="R", choices=["R", "S", "Q", "SQ"])
    parser.add_argument("--cache-dir", type=str, default="fastf1_cache")
    parser.add_argument("--out", type=str, required=True)
    args = parser.parse_args()

    cfg = ExtractConfig(cache_dir=args.cache_dir, session_type=args.session)
    df = extract_race_events(args.year, args.round, cfg)

    print(df)
    print(f"Rows: {len(df):,}")

    if args.out.endswith(".csv"):
        df.to_csv(args.out, index=False)
    elif args.out.endswith(".parquet"):
        df.to_parquet(args.out, index=False)
    else:
        raise ValueError("Output must end with .csv or .parquet")

    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
