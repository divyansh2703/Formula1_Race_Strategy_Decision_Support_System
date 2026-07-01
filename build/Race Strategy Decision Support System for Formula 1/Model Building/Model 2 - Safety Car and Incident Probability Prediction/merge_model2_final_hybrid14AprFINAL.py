from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from typing import Dict, List

import numpy as np
import pandas as pd


def now_ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def safe_numeric(s: pd.Series, fill: float = 0.0) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").fillna(fill)


def write_report_md(report_path: str, title: str, payload: Dict) -> None:
    lines: List[str] = []
    lines.append(f"# {title}")
    lines.append("")
    lines.append(f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
    lines.append("")
    for key, value in payload.items():
        lines.append(f"## {key}")
        lines.append("")
        if isinstance(value, (dict, list)):
            lines.append("```json")
            lines.append(json.dumps(value, indent=2))
            lines.append("```")
        else:
            lines.append(str(value))
        lines.append("")
    with open(report_path, "w") as f:
        f.write("\n".join(lines))


def score_prob_split(y_true: np.ndarray, p_hat: np.ndarray) -> Dict:
    from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score

    y_true = np.asarray(y_true, dtype=float)
    p_hat = np.asarray(p_hat, dtype=float)

    mask = np.isfinite(y_true) & np.isfinite(p_hat)
    y_true = y_true[mask].astype(int)
    p_hat = p_hat[mask].astype(float)

    out = {
        "n": int(len(y_true)),
        "event_rate": float(np.mean(y_true)) if len(y_true) else float("nan"),
        "roc_auc": float("nan"),
        "pr_auc": float("nan"),
        "brier": float("nan"),
        "logloss": float("nan"),
    }

    if len(y_true) == 0:
        return out

    p_hat = np.clip(p_hat, 1e-6, 1.0 - 1e-6)

    if len(np.unique(y_true)) >= 2:
        out["roc_auc"] = float(roc_auc_score(y_true, p_hat))
        out["pr_auc"] = float(average_precision_score(y_true, p_hat))
        out["brier"] = float(brier_score_loss(y_true, p_hat))
        out["logloss"] = float(log_loss(y_true, p_hat))

    return out


def reliability_table(y_true: np.ndarray, p_hat: np.ndarray, n_bins: int = 10) -> List[Dict]:
    y_true = np.asarray(y_true, dtype=float)
    p_hat = np.asarray(p_hat, dtype=float)

    mask = np.isfinite(y_true) & np.isfinite(p_hat)
    y_true = y_true[mask].astype(int)
    p_hat = p_hat[mask].astype(float)

    df = pd.DataFrame({"y": y_true, "p": p_hat})
    if len(df) == 0:
        return []

    try:
        df["bin"] = pd.qcut(df["p"], q=n_bins, duplicates="drop")
    except Exception:
        df["bin"] = pd.cut(df["p"], bins=n_bins, include_lowest=True)

    out = (
        df.groupby("bin", observed=True)
        .agg(
            n=("y", "size"),
            event_rate=("y", "mean"),
            mean_p=("p", "mean"),
            min_p=("p", "min"),
            max_p=("p", "max"),
        )
        .reset_index()
    )
    out["bin"] = out["bin"].astype(str)
    return out.to_dict(orient="records")


def add_event_start_flags(master: pd.DataFrame, event_col: str, out_col: str) -> pd.DataFrame:
    df = master.sort_values(["race_id", "lap_number"]).copy()
    cur = safe_numeric(df[event_col], fill=0).astype(int)
    prev = df.groupby("race_id", sort=False)[event_col].shift(1).fillna(0).astype(int)
    df[out_col] = ((cur == 1) & (prev == 0)).astype(int)
    return df


def build_horizon_truth(master: pd.DataFrame, start_col: str, horizons: List[int]) -> pd.DataFrame:
    df = master[["race_id", "lap_number", "total_laps", start_col]].copy()
    df["total_laps"] = safe_numeric(df["total_laps"], fill=np.nan)
    out = df[["race_id", "lap_number"]].drop_duplicates().copy()

    future_start = df[["race_id", "lap_number", start_col]].copy()

    for h in horizons:
        parts = []
        for k in range(1, h + 1):
            tmp = df[["race_id", "lap_number", "total_laps"]].copy()
            tmp["future_lap_number"] = tmp["lap_number"] + k
            tmp = tmp.loc[tmp["future_lap_number"] <= tmp["total_laps"]].copy()
            tmp = tmp.merge(
                future_start.rename(columns={"lap_number": "future_lap_number", start_col: "y_future"}),
                on=["race_id", "future_lap_number"],
                how="left",
            )
            tmp["y_future"] = tmp["y_future"].fillna(0).astype(int)
            parts.append(tmp[["race_id", "lap_number", "y_future"]])

        if parts:
            allp = pd.concat(parts, axis=0, ignore_index=True)
            y = (
                allp.groupby(["race_id", "lap_number"], as_index=False)["y_future"]
                .max()
                .rename(columns={"y_future": f"y_next{h}"})
            )
            out = out.merge(y, on=["race_id", "lap_number"], how="left")
            out[f"y_next{h}"] = out[f"y_next{h}"].fillna(0).astype(int)

    return out


def fill_prob_cols_by_race(df: pd.DataFrame, prob_cols: List[str]) -> pd.DataFrame:
    out = df.copy()
    for c in prob_cols:
        out[c] = pd.to_numeric(out[c], errors="coerce")
        out[c] = out.groupby("race_id")[c].ffill()
        out[c] = out.groupby("race_id")[c].bfill()
        med = float(out[c].median()) if out[c].notna().any() else 0.0
        if not np.isfinite(med):
            med = 0.0
        out[c] = out[c].fillna(med).clip(0.0, 1.0)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_path", required=True, help="Base lap-level master csv")
    ap.add_argument("--outputs_dir", required=True)
    ap.add_argument("--models_dir", required=True)

    ap.add_argument("--sc_supervised_csv", required=True, help="model2 hazard race-lap predictions csv from supervised rebuild")
    ap.add_argument("--vsc_supervised_csv", required=True, help="model2 hazard race-lap predictions csv from supervised rebuild")
    ap.add_argument("--vsc_prior_csv", required=True, help="VSC prior race-lap predictions csv")
    ap.add_argument("--rf_prior_source_csv", required=True, help="Either supervised rebuild race-lap predictions csv containing p_rf_* or dedicated rf prior csv")

    ap.add_argument("--horizons", default="3,5,10")
    ap.add_argument("--output_name", default="lap_level_with_model2_hybrid_final.csv")

    ap.add_argument("--sc_event_col", default="is_under_safety_car_event")
    ap.add_argument("--vsc_event_col", default="is_under_vsc_event")
    ap.add_argument("--rf_event_col", default="is_under_red_flag_event")

    args = ap.parse_args()

    ensure_dir(args.outputs_dir)
    ensure_dir(args.models_dir)
    ts = now_ts()
    horizons = [int(x.strip()) for x in args.horizons.split(",") if x.strip()]

    master = pd.read_csv(args.input_path, low_memory=False)
    sc_df = pd.read_csv(args.sc_supervised_csv, low_memory=False)
    vsc_sup_df = pd.read_csv(args.vsc_supervised_csv, low_memory=False)
    vsc_prior_df = pd.read_csv(args.vsc_prior_csv, low_memory=False)
    rf_df = pd.read_csv(args.rf_prior_source_csv, low_memory=False)

    key_cols = ["race_id", "lap_number"]

    hybrid = master[key_cols].drop_duplicates().copy()

    # SC from supervised branch for all requested horizons
    for h in horizons:
        c = f"p_sc_next{h}"
        if c not in sc_df.columns:
            raise ValueError(f"Missing {c} in sc_supervised_csv")
        hybrid = hybrid.merge(sc_df[key_cols + [c]].drop_duplicates(key_cols), on=key_cols, how="left")

    # VSC hybrid
    for h in horizons:
        c = f"p_vsc_next{h}"
        if h in [3, 5]:
            if c not in vsc_sup_df.columns:
                raise ValueError(f"Missing {c} in vsc_supervised_csv")
            use_df = vsc_sup_df
            source = "supervised"
        elif h == 10:
            if c not in vsc_prior_df.columns:
                raise ValueError(f"Missing {c} in vsc_prior_csv")
            use_df = vsc_prior_df
            source = "prior"
        else:
            if c in vsc_sup_df.columns:
                use_df = vsc_sup_df
                source = "supervised"
            elif c in vsc_prior_df.columns:
                use_df = vsc_prior_df
                source = "prior"
            else:
                raise ValueError(f"Missing {c} in both vsc_supervised_csv and vsc_prior_csv")

        tmp = use_df[key_cols + [c]].drop_duplicates(key_cols).copy()
        hybrid = hybrid.merge(tmp, on=key_cols, how="left")
        hybrid[f"{c}_source"] = source

    # RF from prior fallback source
    for h in horizons:
        c = f"p_rf_next{h}"
        if c not in rf_df.columns:
            raise ValueError(f"Missing {c} in rf_prior_source_csv")
        hybrid = hybrid.merge(rf_df[key_cols + [c]].drop_duplicates(key_cols), on=key_cols, how="left")

    # Fill hybrid probabilities before evaluation
    hybrid_prob_cols = [c for c in hybrid.columns if c.startswith("p_") and not c.endswith("_source")]
    hybrid = fill_prob_cols_by_race(hybrid, hybrid_prob_cols)

    # Merge hybrid predictions back to master
    merged = master.merge(hybrid, on=key_cols, how="left")

    prob_cols = [c for c in hybrid.columns if c.startswith("p_") and not c.endswith("_source")]
    for c in prob_cols:
        merged[f"{c}_was_nan"] = merged[c].isna().astype(int)
    merged = fill_prob_cols_by_race(merged, prob_cols)

    if prob_cols:
        merged["model2_hybrid_prob_imputed_any"] = merged[[f"{c}_was_nan" for c in prob_cols]].max(axis=1).astype(int)

    # Add truth labels for evaluation
    eval_master = master.copy()

    if args.sc_event_col not in eval_master.columns:
        raise ValueError(f"Missing {args.sc_event_col} in input_path")
    if args.vsc_event_col not in eval_master.columns:
        raise ValueError(f"Missing {args.vsc_event_col} in input_path")
    if args.rf_event_col not in eval_master.columns:
        raise ValueError(f"Missing {args.rf_event_col} in input_path")
    if "total_laps" not in eval_master.columns:
        raise ValueError("Missing total_laps in input_path")

    eval_master = add_event_start_flags(eval_master, args.sc_event_col, "sc_start_this_lap")
    eval_master = add_event_start_flags(eval_master, args.vsc_event_col, "vsc_start_this_lap")
    eval_master = add_event_start_flags(eval_master, args.rf_event_col, "rf_start_this_lap")

    sc_truth = build_horizon_truth(eval_master, "sc_start_this_lap", horizons)
    vsc_truth = build_horizon_truth(eval_master, "vsc_start_this_lap", horizons)
    rf_truth = build_horizon_truth(eval_master, "rf_start_this_lap", horizons)

    eval_df = hybrid.merge(sc_truth, on=key_cols, how="left", suffixes=("", "_sc"))
    for h in horizons:
        eval_df = eval_df.merge(
            vsc_truth[key_cols + [f"y_next{h}"]].rename(columns={f"y_next{h}": f"y_vsc_next{h}"}),
            on=key_cols,
            how="left",
        )
        eval_df = eval_df.merge(
            rf_truth[key_cols + [f"y_next{h}"]].rename(columns={f"y_next{h}": f"y_rf_next{h}"}),
            on=key_cols,
            how="left",
        )
        if f"y_next{h}" in eval_df.columns:
            eval_df = eval_df.rename(columns={f"y_next{h}": f"y_sc_next{h}"})

    # Evaluate overall hybrid probabilities
    metrics: Dict[str, Dict] = {
        "sc": {},
        "vsc": {},
        "rf": {},
    }

    for h in horizons:
        sc_p = eval_df[f"p_sc_next{h}"].to_numpy(dtype=float)
        sc_y = eval_df[f"y_sc_next{h}"].fillna(0).to_numpy(dtype=int)
        metrics["sc"][f"next{h}"] = {
            "overall": score_prob_split(sc_y, sc_p),
            "reliability": reliability_table(sc_y, sc_p, n_bins=10),
        }

        vsc_p = eval_df[f"p_vsc_next{h}"].to_numpy(dtype=float)
        vsc_y = eval_df[f"y_vsc_next{h}"].fillna(0).to_numpy(dtype=int)
        metrics["vsc"][f"next{h}"] = {
            "overall": score_prob_split(vsc_y, vsc_p),
            "reliability": reliability_table(vsc_y, vsc_p, n_bins=10),
            "source": "supervised" if h in [3, 5] else "prior",
        }

        rf_p = eval_df[f"p_rf_next{h}"].to_numpy(dtype=float)
        rf_y = eval_df[f"y_rf_next{h}"].fillna(0).to_numpy(dtype=int)
        metrics["rf"][f"next{h}"] = {
            "overall": score_prob_split(rf_y, rf_p),
            "reliability": reliability_table(rf_y, rf_p, n_bins=10),
            "source": "prior_poisson_fallback",
        }

    summary_rows = []
    for ev in ["sc", "vsc", "rf"]:
        for h in horizons:
            hk = f"next{h}"
            row = {
                "event": ev,
                "horizon": h,
                "source": metrics[ev][hk].get("source", "supervised"),
                "n": metrics[ev][hk]["overall"]["n"],
                "event_rate": metrics[ev][hk]["overall"]["event_rate"],
                "roc_auc": metrics[ev][hk]["overall"]["roc_auc"],
                "pr_auc": metrics[ev][hk]["overall"]["pr_auc"],
                "brier": metrics[ev][hk]["overall"]["brier"],
                "logloss": metrics[ev][hk]["overall"]["logloss"],
            }
            summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)

    # Save outputs
    merged_path = os.path.join(args.outputs_dir, args.output_name)
    merged.to_csv(merged_path, index=False)

    racelap_pred_path = os.path.join(args.outputs_dir, f"model2_hybrid_racelap_predictions_{ts}.csv")
    hybrid.to_csv(racelap_pred_path, index=False)

    metrics_path = os.path.join(args.outputs_dir, f"model2_hybrid_final_metrics_{ts}.json")
    with open(metrics_path, "w") as f:
        json.dump(
            {
                "mode": "model2_hybrid_final",
                "selection_logic": {
                    "sc": "supervised for all horizons",
                    "vsc_next3": "supervised",
                    "vsc_next5": "supervised",
                    "vsc_next10": "prior",
                    "rf": "prior_poisson_fallback for all horizons",
                },
                "inputs": {
                    "input_path": args.input_path,
                    "sc_supervised_csv": args.sc_supervised_csv,
                    "vsc_supervised_csv": args.vsc_supervised_csv,
                    "vsc_prior_csv": args.vsc_prior_csv,
                    "rf_prior_source_csv": args.rf_prior_source_csv,
                },
                "metrics": metrics,
            },
            f,
            indent=2,
        )

    summary_csv_path = os.path.join(args.outputs_dir, f"model2_hybrid_final_summary_{ts}.csv")
    summary_df.to_csv(summary_csv_path, index=False)

    freeze_manifest = {
        "freeze_name": "model2_hybrid_final",
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
        "status": "CANDIDATE_FROZEN",
        "selection_logic": {
            "sc": "supervised for all horizons",
            "vsc": {
                "next3": "supervised",
                "next5": "supervised",
                "next10": "prior",
            },
            "rf": "prior_poisson_fallback for all horizons",
        },
        "artifacts": {
            "merged_output": merged_path,
            "racelap_predictions": racelap_pred_path,
            "metrics_json": metrics_path,
            "summary_csv": summary_csv_path,
        },
    }
    freeze_manifest_path = os.path.join(args.models_dir, f"model2_hybrid_freeze_manifest_{ts}.json")
    with open(freeze_manifest_path, "w") as f:
        json.dump(freeze_manifest, f, indent=2)

    report_payload = {
        "run_meta": {
            "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
            "script": "merge_model2_final_hybrid.py",
            "horizons": horizons,
        },
        "selection_logic": freeze_manifest["selection_logic"],
        "inputs": {
            "input_path": args.input_path,
            "sc_supervised_csv": args.sc_supervised_csv,
            "vsc_supervised_csv": args.vsc_supervised_csv,
            "vsc_prior_csv": args.vsc_prior_csv,
            "rf_prior_source_csv": args.rf_prior_source_csv,
        },
        "artifacts": freeze_manifest["artifacts"],
        "summary_rows": summary_rows,
        "notes": [
            "SC is taken from the supervised hazard rebuild for all horizons.",
            "VSC uses supervised predictions for next3 and next5, and prior model for next10.",
            "RF uses the prior plus Poisson fallback for all horizons.",
            "This script creates the final hybrid Model 2 output to be used downstream.",
        ],
    }
    report_path = os.path.join(args.outputs_dir, f"model2_hybrid_final_report_{ts}.md")
    write_report_md(report_path, "Model 2 Hybrid Final Report", report_payload)

    print("Saved merged hybrid output:", merged_path)
    print("Saved race-lap hybrid predictions:", racelap_pred_path)
    print("Saved metrics json:", metrics_path)
    print("Saved summary csv:", summary_csv_path)
    print("Saved freeze manifest:", freeze_manifest_path)
    print("Saved report:", report_path)


if __name__ == "__main__":
    main()