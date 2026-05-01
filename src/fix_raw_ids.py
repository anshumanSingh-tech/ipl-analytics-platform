import pandas as pd
from pathlib import Path

RAW = Path("E:/ipl-analytics/data/raw")

matches    = pd.read_csv(RAW / "matches.csv")
deliveries = pd.read_csv(RAW / "deliveries.csv")

print("=== RAW matches.csv ===")
print(f"  shape   : {matches.shape}")
print(f"  columns : {matches.columns.tolist()}")

print("\n=== RAW deliveries.csv ===")
print(f"  shape   : {deliveries.shape}")
print(f"  columns : {deliveries.columns.tolist()}")

match_id_col    = "match_id" if "match_id" in matches.columns    else "id"
delivery_id_col = "match_id" if "match_id" in deliveries.columns else "id"

print(f"\nMatches ID col    : '{match_id_col}'")
print(f"Deliveries ID col : '{delivery_id_col}'")

print(f"\nMatches   {match_id_col} nulls : {matches[match_id_col].isna().sum()}")
print(f"Deliveries {delivery_id_col} nulls : {deliveries[delivery_id_col].isna().sum()}")

null_mask = matches[match_id_col].isna()
print(f"\nRows with null match ID : {null_mask.sum()}")

if null_mask.sum() > 0:
    last_valid_id = int(matches[match_id_col].dropna().max())
    new_ids = range(last_valid_id + 1,
                    last_valid_id + 1 + null_mask.sum())
    matches.loc[null_mask, match_id_col] = list(new_ids)
    matches[match_id_col] = matches[match_id_col].astype(int)
    print(f"Assigned new IDs : {last_valid_id + 1} → {last_valid_id + null_mask.sum()}")
else:
    print("No null match IDs — matches.csv is fine.")

if match_id_col == "id":
    matches = matches.rename(columns={"id": "match_id"})
    print("Renamed 'id' → 'match_id' in matches.csv")
    match_id_col = "match_id"

if delivery_id_col == "id":
    deliveries = deliveries.rename(columns={"id": "match_id"})
    print("Renamed 'id' → 'match_id' in deliveries.csv")
    delivery_id_col = "match_id"

valid_ids = set(matches["match_id"].unique())
delivery_ids = set(deliveries["match_id"].unique())
orphans = delivery_ids - valid_ids

print(f"\nOrphaned delivery match_ids : {len(orphans)}")

if orphans:
    print(f"Sample orphan IDs : {list(orphans)[:10]}")

    new_id_range = set(range(last_valid_id + 1,
                              last_valid_id + 1 + null_mask.sum())) \
                   if null_mask.sum() > 0 else set()

    recoverable     = orphans & new_id_range
    unrecoverable   = orphans - new_id_range

    print(f"  Recoverable (within new ID range) : {len(recoverable)}")
    print(f"  Unrecoverable (unknown origin)    : {len(unrecoverable)}")

    if unrecoverable:
        print(f"  Dropping {len(unrecoverable)} truly orphaned match_ids from deliveries")
        deliveries = deliveries[
            ~deliveries["match_id"].isin(unrecoverable)
        ].copy()

orphans_after = set(deliveries["match_id"].unique()) - set(matches["match_id"].unique())
print(f"\nOrphaned deliveries after fix : {len(orphans_after)}")

matches.to_csv(RAW / "matches.csv",    index=False)
deliveries.to_csv(RAW / "deliveries.csv", index=False)

print("\n=== Saved ===")
print(f"  matches.csv    : {matches.shape}")
print(f"  deliveries.csv : {deliveries.shape}")
print(f"  Seasons        : {sorted(matches['season'].unique())}")

m2 = pd.read_csv(RAW / "matches.csv")
d2 = pd.read_csv(RAW / "deliveries.csv")

print("\n=== Post-save verification ===")
print(f"  matches ID col      : {'match_id' if 'match_id' in m2.columns else 'id'}")
print(f"  deliveries ID col   : {'match_id' if 'match_id' in d2.columns else 'id'}")
print(f"  match_id nulls      : {m2['match_id'].isna().sum() if 'match_id' in m2.columns else d2['id'].isna().sum()}")
final_orphans = set(d2["match_id" if "match_id" in d2.columns else "id"].unique()) - \
                set(m2["match_id" if "match_id" in m2.columns else "id"].unique())
print(f"  Final orphan count  : {len(final_orphans)}")

if len(final_orphans) == 0:
    print("\nAll clean. Re-run notebook 02 now.")
else:
    print(f"\nWARNING — {len(final_orphans)} orphans remain. Share output for further diagnosis.")