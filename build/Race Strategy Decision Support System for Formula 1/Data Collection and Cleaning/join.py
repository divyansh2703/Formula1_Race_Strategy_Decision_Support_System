import pandas as pd

LAP_PATH = "lapbylap_2019_2025_R_all.csv"
PIT_PATH = "pitstops_2019_2025_R_all.csv"
OUT_PATH = "lapbylap_with_pitstops_2019_2025_R_all.csv"

# Load
laps = pd.read_csv(LAP_PATH)
pits = pd.read_csv(PIT_PATH)

# Standardize join keys
for df in (laps, pits):
    df["race_id"] = df["race_id"].astype(str)
    df["driver_id"] = df["driver_id"].astype(str)
    df["lap_number"] = pd.to_numeric(df["lap_number"], errors="coerce").astype("Int64")
    df["season"] = pd.to_numeric(df["season"], errors="coerce").astype("Int64")

# Keep only required pit columns and make them unique per driver-lap
pit_cols = [
    "race_id",
    "season",
    "driver_id",
    "lap_number",
    "pit_stop_number",
    "pit_entry_time",
    "pit_exit_time",
    "pit_lane_time",
    "stationary_time",
    "tyre_compound_before",
    "tyre_compound_after",
    "tyre_age_before",
]
pit_cols = [c for c in pit_cols if c in pits.columns]
pits = pits[pit_cols].copy()

# If there are multiple pit events on same driver-lap, keep the first by pit_stop_number
if "pit_stop_number" in pits.columns:
    pits["pit_stop_number"] = pd.to_numeric(pits["pit_stop_number"], errors="coerce").astype("Int64")
    pits = pits.sort_values(["race_id", "driver_id", "lap_number", "pit_stop_number"])
else:
    pits = pits.sort_values(["race_id", "driver_id", "lap_number"])

pits = pits.drop_duplicates(subset=["race_id", "driver_id", "lap_number"], keep="first").copy()

# Rename pit columns to avoid collisions
rename_map = {c: f"pit_{c}" for c in pits.columns if c not in ["race_id", "season", "driver_id", "lap_number"]}
pits = pits.rename(columns=rename_map)

# Left join so you keep every lap row
merged = laps.merge(
    pits,
    on=["race_id", "season", "driver_id", "lap_number"],
    how="left",
)

# Create clean pit flags in the lap table
# in-lap = lap where pit happened (has pit_lane_time or pit_entry/exit)
pit_signal_cols = [c for c in ["pit_pit_lane_time", "pit_pit_entry_time", "pit_pit_exit_time"] if c in merged.columns]
if pit_signal_cols:
    merged["is_in_lap"] = merged[pit_signal_cols].notna().any(axis=1)
else:
    merged["is_in_lap"] = False

# out-lap = next lap after pit lap for same race+driver
merged = merged.sort_values(["race_id", "driver_id", "lap_number"]).copy()
merged["is_out_lap"] = merged.groupby(["race_id", "driver_id"])["is_in_lap"].shift(1).fillna(False).astype(bool)

merged["is_pit_lap"] = (merged["is_in_lap"] | merged["is_out_lap"]).astype(bool)

# Save
merged.to_csv(OUT_PATH, index=False)
print("Wrote:", OUT_PATH)
print("Pit laps in merged:", int(merged["is_in_lap"].sum()))
