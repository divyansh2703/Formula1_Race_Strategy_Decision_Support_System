import pandas as pd
from pathlib import Path
import re

folder = Path(".")

def key(p):
    m = re.search(r"race_sc_vsc_(\d{4})_(\d{2})_R\.csv", p.name)
    return (int(m.group(1)), int(m.group(2))) if m else (9999, 9999)

files = sorted(folder.glob("race_sc_vsc_*_R.csv"), key=key)

if not files:
    raise FileNotFoundError("No safety car CSV files found")

dfs = []
for f in files:
    df = pd.read_csv(f)
    df["source_file"] = f.name
    dfs.append(df)

combined = pd.concat(dfs, ignore_index=True)

combined.to_csv("race_sc_vsc_2019_2025_R_all.csv", index=False)

print(f"Merged {len(files)} files")
print(f"Total rows: {len(combined)}")
