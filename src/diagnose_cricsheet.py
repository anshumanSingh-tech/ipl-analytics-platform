# src/diagnose_cricsheet.py
from pathlib import Path
import pandas as pd

cricsheet_dir = Path("E:/ipl-analytics/data/raw/cricsheet_ipl")

match_files   = sorted([f for f in cricsheet_dir.glob("*.csv")
                        if "_info" not in f.stem])

print(f"Match files : {len(match_files)}")

print("\nScanning all match files for seasons (30 seconds)...")
season_counts = {}
for fp in match_files:
    try:
        row = pd.read_csv(fp, usecols=["season"], nrows=1)
        s   = str(row["season"].iloc[0]).strip()
        season_counts[s] = season_counts.get(s, 0) + 1
    except Exception:
        continue

print(f"\nAll unique season labels in Cricsheet:")
for s in sorted(season_counts):
    print(f"  '{s}' : {season_counts[s]} matches")