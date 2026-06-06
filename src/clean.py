import pandas as pd
import numpy as np
from pathlib import Path

RAW = Path("E:/ipl-analytics/data/raw")
PROCESSED = Path("E:/ipl-analytics/data/processed")
PROCESSED.mkdir(parents=True, exist_ok=True)

TEAM_NAME_MAP = {
    "Rising Pune Supergiant" : "Rising Pune Supergiants",
    "King XI Punjab" : "Punjab Kings",
    "Pune Warriors" : "Pune Warriors India",
    "Deccan Chargrers" : "Sunrisers Hyderabad",
    "Delhi Daredevils" : "Delhi Capitals",
    "Royal Challengers Bengaluru" : "Royal Challengers Bangalore",
    
}

VENUE_KEYWORDS = {
    "Wankhede": "Wankhede Stadium",
    "Chinnaswamy": "M Chinnaswamy Stadium",
    "Chidambaram": "MA Chidambaram Stadium",
    "Chepauk": "MA Chidambaram Stadium",
    "Eden Gardens": "Eden Gardens",
    "Arun Jaitley": "Feroz Shah Kotla",
    "Feroz Shah Kotla": "Feroz Shah Kotla",
    "Rajasekhara": "Vizag Stadium",
    "Yadavindra": "Mullanpur Stadium",
    "Ekana": "Ekana Stadium",
    "Rajiv Gandhi": "Uppal Stadium",
    "IS Bindra": "PCA Stadium",
    "Mohali": "PCA Stadium",
    "Narendra Modi": "Motera Stadium",
    "Dy Patil": "DY Patil Stadium",
    "Brabourne": "Brabourne Stadium",
}

IPL_VALID_SEASONS = set(range(2008, 2025))

import datetime
IPL_VALID_SEASONS = set(range(2008, datetime.date.today().year + 1))

SLASH_SEASON_MAP = {
    "2007/08" : 2008,
    "2009/10" : 2010,
    "2020/21" : 2020,
}

def clean_matches(matches: pd.DataFrame) -> pd.DataFrame:
    df = matches.copy()

    if "id" in df.columns:
        df = df.rename(columns={"id": "match_id"})

    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    def fix_season(val):
        val = str(val).strip()
        
        if val in SLASH_SEASON_MAP:
            return SLASH_SEASON_MAP[val]

        if "/" in val:
            parts  = val.split("/")
            suffix = parts[1].strip()
            prefix = parts[0].strip()[:2]
            return int(prefix + suffix)
        
        val_clean = val.replace("IPL", "").replace("Season", "").strip()
        year = int(float(val_clean))

        if year == 2007:
            return 2008
        return year

    df["season"] = df["season"].apply(fix_season).astype("int32")

    detected  = set(df["season"].unique())
    unexpected = detected - IPL_VALID_SEASONS
    missing    = IPL_VALID_SEASONS - detected

    if unexpected:
        print(f"WARNING — unexpected seasons found    : {sorted(unexpected)}")
        print(f"  Dropping {len(df[df['season'].isin(unexpected)])} rows.")
        df = df[df["season"].isin(IPL_VALID_SEASONS)].copy()
    else:
        print("OK — all seasons are valid IPL years.")

    if missing:
        print(f"INFO — seasons not in dataset : {sorted(missing)}")
        print(f"  Expected if your CSV doesn't cover those years yet.")
    else:
        print("OK — all expected IPL seasons are present.")

    print(f"\nSeason distribution : {sorted(df['season'].unique())}")
    print(f"Season range        : {df['season'].min()} – {df['season'].max()}")
    print(f"Total seasons       : {df['season'].nunique()}")
    print(f"Total matches       : {len(df)}")

    df["venue"] = df["venue"].astype(str).str.strip()
    for keyword, clean_name in VENUE_KEYWORDS.items():
        mask = df["venue"].str.contains(keyword, case=False, na=False)
        df.loc[mask, "venue"] = clean_name

    for col in ["result_margin", "dl_applied", "target_runs"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

    for col in ["team1", "team2", "winner", "toss_winner"]:
        if col in df.columns:
            df[col] = df[col].replace(TEAM_NAME_MAP)

    df["player_of_match"] = df["player_of_match"].fillna("N/A")

    def result_type(row):
        r = str(row.get("result", "")).lower().strip()
        if r == "runs":     return "runs"
        if r == "wickets":  return "wickets"
        return "no_result"

    df["result_type"] = df.apply(result_type, axis=1)

    df["is_no_result"] = df["winner"].isna().astype(int)
    df["winner"]       = df["winner"].fillna("No Result")

    df["day_of_week"] = df["date"].dt.day_name()
    df["month"]       = df["date"].dt.month

    df = df.sort_values(["season", "date"]).reset_index(drop=True)

    print(f"\nclean_matches done: {matches.shape} → {df.shape}")
    remaining_nulls = df.isnull().sum()
    remaining_nulls = remaining_nulls[remaining_nulls > 0]
    if len(remaining_nulls):
        print(f"Remaining nulls:\n{remaining_nulls.to_string()}")
    else:
        print("No nulls remaining.")

    return df

def clean_deliveries(deliveries: pd.DataFrame) -> pd.DataFrame:
    df = deliveries.copy()
    
    df = df.rename(columns={
        "batter": "batsman",
    })
    
    for col in ["batting_team", "bowling_team"]:
        if col in df.columns:
            df[col] = df[col].replace(TEAM_NAME_MAP)
    
    num_cols = ["inning", "over", "ball", "batsman_runs", "extra_runs", "total_runs"]
    for col in num_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)
            
    df["is_wide"] = (df["extras_type"] == "wides").astype(int)
    df["is_noball"] = (df["extras_type"] == "noballs").astype(int)

    for col in ["batting_team", "bowling_team"]:
        df[col] = df[col].replace(TEAM_NAME_MAP)
            
    df["player_dismissed"] = df["player_dismissed"].fillna("not_out")
    df["dismissal_kind"] = df["dismissal_kind"].fillna("not_out")
    
    df["is_wicket"] = df["is_wicket"].fillna(0).astype(int)
    
    df["is_four"] = (df["batsman_runs"] == 4).astype(int)
    df["is_six"] = (df["batsman_runs"] == 6).astype(int)
    
    df["is_dot_ball"] = (
        (df["batsman_runs"] == 0) & 
        (df["is_wide"] == 0) & 
        (df["is_noball"] == 0)
    ).astype(int)
    
    def over_phase(over):
        if over <= 6: return "powerplay"
        if over <= 15: return "middle"
        return "death"
    
    df["over_phase"] = df["over"].apply(over_phase)
    
    df["is_legal_delivery"] = ((df["is_wide"] == 0) & (df["is_noball"] == 0)).astype(int)
    
    df = df.sort_values(["match_id", "inning", "over", "ball"]).reset_index(drop=True)
    print(f"\nclean_deliveries: {deliveries.shape} → {df.shape}")
    print(f"  Nulls remaining:\n{df.isnull().sum()[df.isnull().sum()>0].to_string()}")
    return df