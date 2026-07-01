import pandas as pd
from pathlib import Path
import re

folder = Path(".")

def key(p):
    # ONLY matches: lapbylap_2019_01_new_R.csv
    m = re.search(r"lapbylap_(\d{4})_(\d{2})_new_R\.csv$", p.name)
    return (int(m.group(1)), int(m.group(2))) if m else (9999, 9999)

files = sorted(folder.glob("lapbylap_*_new_R.csv"), key=key)

if not files:
    raise FileNotFoundError("No lapbylap *_new_R.csv files found in current folder")

dfs = []
for f in files:
    df = pd.read_csv(f)
    df["source_file"] = f.name
    dfs.append(df)

combined = pd.concat(dfs, ignore_index=True)

# Optional: stable sort and dedupe
sort_cols = [c for c in ["season", "race_id", "driver_id", "lap_number"] if c in combined.columns]
if sort_cols:
    combined = combined.sort_values(sort_cols).reset_index(drop=True)

if all(c in combined.columns for c in ["race_id", "driver_id", "lap_number"]):
    combined = combined.drop_duplicates(
        subset=["race_id", "driver_id", "lap_number"], keep="first"
    )

combined.to_csv("lapbylap_2019_2025_R_all_new.csv", index=False)

print(f"Merged {len(files)} files")
print(f"Total rows: {len(combined)}")
print("Output: lapbylap_2019_2025_R_all_new.csv")
