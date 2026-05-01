# src/update_dataset.py
import requests, zipfile, io, datetime, json
import pandas as pd
import numpy as np
from pathlib import Path

RAW = Path("E:/ipl-analytics/data/raw")

SLASH_SEASON_MAP = {
    "2007/08" : 2008,
    "2009/10" : 2010,
    "2020/21" : 2020,
}

def normalize_season(val) -> int:
    val = str(val).strip()
    if val in SLASH_SEASON_MAP:
        return SLASH_SEASON_MAP[val]
    if "/" in val:
        parts  = val.split("/")
        prefix = parts[0].strip()[:2]
        suffix = parts[1].strip()
        return int(prefix + suffix)
    return int(float(val.replace("IPL","").replace("Season","").strip()))


def safe_int(val, default: int = 0) -> int:
    try:
        if val is None:
            return default
        if isinstance(val, float) and np.isnan(val):
            return default
        return int(float(val))
    except (ValueError, TypeError):
        return default


def download_cricsheet_json() -> Path:
    url = "https://cricsheet.org/downloads/ipl_json.zip"
    print("Fetching Cricsheet IPL JSON data...")
    r = requests.get(url, timeout=180)
    r.raise_for_status()
    out = RAW / "cricsheet_ipl_json"
    out.mkdir(parents=True, exist_ok=True)
    zipfile.ZipFile(io.BytesIO(r.content)).extractall(out)
    json_files = list(out.glob("*.json"))
    print(f"  {len(json_files)} JSON match files extracted.")
    return out


def get_existing_seasons(matches_path: Path) -> set:
    df = pd.read_csv(matches_path, usecols=["season"])
    return set(df["season"].apply(normalize_season).unique())


def parse_json_match(filepath: Path, match_id: int) -> tuple:
    try:
        with open(filepath, encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return None, []

    info = data.get("info", {})

    # ── Season ────────────────────────────────────────────────────────
    try:
        season = normalize_season(info.get("season", ""))
    except Exception:
        return None, []

    # ── Teams ─────────────────────────────────────────────────────────
    teams = info.get("teams", [])
    if len(teams) < 2:
        return None, []
    team1 = teams[0]
    team2 = teams[1]

    # ── Date ──────────────────────────────────────────────────────────
    dates       = info.get("dates", [])
    raw_date    = dates[0] if dates else np.nan
    parsed_date = pd.to_datetime(raw_date, errors="coerce")
    day_of_week = parsed_date.day_name() if pd.notna(parsed_date) else np.nan
    month       = int(parsed_date.month)  if pd.notna(parsed_date) else np.nan

    # ── Toss — directly from info["toss"] dict ────────────────────────
    toss          = info.get("toss", {})
    toss_winner   = toss.get("winner",   np.nan)
    toss_decision = toss.get("decision", np.nan)

    # ── Outcome ───────────────────────────────────────────────────────
    outcome      = info.get("outcome", {})
    winner       = outcome.get("winner", np.nan)
    is_no_result = 0

    if pd.isna(winner) or winner == "":
        if outcome.get("result") == "no result" or "result" in outcome:
            winner       = "No Result"
            is_no_result = 1
        else:
            winner = "No Result"

    by            = outcome.get("by", {})
    result_margin = 0
    result_type   = "normal"
    if "runs" in by:
        result_margin = safe_int(by["runs"])
        result_type   = "runs"
    elif "wickets" in by:
        result_margin = safe_int(by["wickets"])
        result_type   = "wickets"

    # ── Player of match ───────────────────────────────────────────────
    pom_list        = info.get("player_of_match", [])
    player_of_match = pom_list[0] if pom_list else np.nan

    # ── Venue / city ──────────────────────────────────────────────────
    venue = info.get("venue", np.nan)
    city  = info.get("city",  np.nan)

    match_row = {
        "match_id"        : match_id,
        "season"          : season,
        "date"            : raw_date,
        "team1"           : team1,
        "team2"           : team2,
        "toss_winner"     : toss_winner,
        "toss_decision"   : toss_decision,
        "winner"          : winner,
        "player_of_match" : player_of_match,
        "venue"           : venue,
        "result"          : result_type,
        "result_margin"   : result_margin,
        "method"          : info.get("method", np.nan),
        "city"            : city,
        "is_no_result"    : is_no_result,
        "result_type"     : result_type,
        "day_of_week"     : day_of_week,
        "month"           : month,
    }

    # ── Deliveries ────────────────────────────────────────────────────
    delivery_rows = []

    for inning_idx, inning in enumerate(data.get("innings", []), start=1):
        batting_team = inning.get("team", np.nan)
        bowling_team = team2 if batting_team == team1 else team1

        for over_data in inning.get("overs", []):
            over_num = safe_int(over_data.get("over", 0)) + 1

            if over_num <= 6:    phase = "powerplay"
            elif over_num <= 15: phase = "middle"
            else:                phase = "death"

            for ball_idx, delivery in enumerate(
                over_data.get("deliveries", []), start=1
            ):
                runs       = delivery.get("runs", {})
                bat_runs   = safe_int(runs.get("batter", 0))
                extra_runs = safe_int(runs.get("extras", 0))
                total_runs = safe_int(runs.get("total",  0))

                extras      = delivery.get("extras", {})
                wide_runs   = safe_int(extras.get("wides",   0))
                noball      = safe_int(extras.get("noballs", 0))
                bye_runs    = safe_int(extras.get("byes",    0))
                legbye_runs = safe_int(extras.get("legbyes", 0))
                is_legal    = 1 if (wide_runs == 0 and noball == 0) else 0

                wickets          = delivery.get("wickets", [])
                is_wicket        = 1 if wickets else 0
                wicket_type      = np.nan
                player_dismissed = np.nan
                fielder          = np.nan

                if wickets:
                    w                = wickets[0]
                    wicket_type      = w.get("kind",       np.nan)
                    player_dismissed = w.get("player_out", np.nan)
                    fielders         = w.get("fielders",   [])
                    fielder          = (fielders[0].get("name", np.nan)
                                        if fielders else np.nan)

                delivery_rows.append({
                    "match_id"         : match_id,
                    "inning"           : inning_idx,
                    "batting_team"     : batting_team,
                    "bowling_team"     : bowling_team,
                    "over"             : over_num,
                    "ball"             : ball_idx,
                    "batter"           : delivery.get("batter",      np.nan),
                    "non_striker"      : delivery.get("non_striker", np.nan),
                    "bowler"           : delivery.get("bowler",      np.nan),
                    "wide_runs"        : wide_runs,
                    "bye_runs"         : bye_runs,
                    "legbye_runs"      : legbye_runs,
                    "noball_runs"      : noball,
                    "batsman_runs"     : bat_runs,
                    "extra_runs"       : extra_runs,
                    "total_runs"       : total_runs,
                    "player_dismissed" : player_dismissed,
                    "dismissal_kind"   : wicket_type,
                    "fielder"          : fielder,
                    "extras_type"      : np.nan,
                    "is_wicket"        : is_wicket,
                    "is_four"          : 1 if bat_runs == 4 else 0,
                    "is_six"           : 1 if bat_runs == 6 else 0,
                    "is_dot_ball"      : 1 if (bat_runs  == 0 and
                                               wide_runs == 0 and
                                               noball    == 0) else 0,
                    "is_legal_delivery": is_legal,
                    "over_phase"       : phase,
                })

    return match_row, delivery_rows


def build_dataframes_from_json(json_dir: Path,
                                seasons_to_add: list,
                                start_match_id: int) -> tuple:
    json_files = sorted(json_dir.glob("*.json"))
    print(f"Scanning {len(json_files)} JSON files "
          f"for seasons {seasons_to_add}...")

    match_rows    = []
    delivery_rows = []
    next_id       = start_match_id
    skipped       = 0

    for fp in json_files:
        try:
            with open(fp, encoding="utf-8") as f:
                peek = json.load(f)
            season = normalize_season(
                peek.get("info", {}).get("season", "")
            )
        except Exception:
            skipped += 1
            continue

        if season not in seasons_to_add:
            continue

        match_row, deliveries = parse_json_match(fp, next_id)
        if match_row is None:
            skipped += 1
            continue

        match_rows.append(match_row)
        delivery_rows.extend(deliveries)
        next_id += 1

    new_matches    = pd.DataFrame(match_rows)
    new_deliveries = pd.DataFrame(delivery_rows)

    if not new_matches.empty:
        print(f"  Parsed          : {len(new_matches)} matches, "
              f"{len(new_deliveries):,} deliveries")
        print(f"  toss_winner nulls : "
              f"{new_matches['toss_winner'].isna().sum()}")
        print(f"  winner nulls      : "
              f"{new_matches['winner'].isna().sum()}")
    print(f"  Skipped         : {skipped} files")

    return new_matches, new_deliveries


def append_new_seasons(seasons_to_add: list = None,
                        force: list = None):
    matches_path    = RAW / "matches.csv"
    deliveries_path = RAW / "deliveries.csv"

    existing_matches    = pd.read_csv(matches_path)
    existing_deliveries = pd.read_csv(
        deliveries_path, low_memory=False
    )

    # ── Normalise ID column ────────────────────────────────────────────
    if "id" in existing_matches.columns \
            and "match_id" not in existing_matches.columns:
        existing_matches = existing_matches.rename(
            columns={"id": "match_id"}
        )
        existing_matches.to_csv(matches_path, index=False)
        print("Renamed 'id' → 'match_id' in matches.csv")

    if "id" in existing_deliveries.columns \
            and "match_id" not in existing_deliveries.columns:
        existing_deliveries = existing_deliveries.rename(
            columns={"id": "match_id"}
        )
        existing_deliveries.to_csv(deliveries_path, index=False)
        print("Renamed 'id' → 'match_id' in deliveries.csv")

    existing_seasons = get_existing_seasons(matches_path)
    print(f"Existing seasons : {sorted(existing_seasons)}")

    current_year = datetime.date.today().year
    if seasons_to_add is None:
        seasons_to_add = list(range(2025, current_year + 1))

    # ── Force refresh ──────────────────────────────────────────────────
    if force:
        print(f"Force-refreshing : {force}")
        existing_matches["_season_norm"] = (
            existing_matches["season"].apply(normalize_season)
        )
        forced_match_ids = set(
            existing_matches.loc[
                existing_matches["_season_norm"].isin(force),
                "match_id"
            ].tolist()
        )
        existing_matches = existing_matches[
            ~existing_matches["_season_norm"].isin(force)
        ].drop(columns=["_season_norm"]).copy()

        existing_deliveries = existing_deliveries[
            ~existing_deliveries["match_id"].isin(forced_match_ids)
        ].copy()

        existing_seasons -= set(force)
        print(f"  Removed {len(forced_match_ids)} matches "
              f"and their deliveries.")

    seasons_to_add = [
        s for s in seasons_to_add if s not in existing_seasons
    ]

    if not seasons_to_add:
        print("All requested seasons already present.")
        print("To refresh: python update_dataset.py 2026 --force")
        return

    print(f"Seasons to add   : {seasons_to_add}")

    json_dir = download_cricsheet_json()
    next_id  = int(existing_matches["match_id"].max()) + 1

    new_matches, new_deliveries = build_dataframes_from_json(
        json_dir, seasons_to_add, next_id
    )

    if new_matches.empty:
        print(f"No matches found for {seasons_to_add}.")
        return

    # ── Align columns ──────────────────────────────────────────────────
    for col in existing_matches.columns:
        if col not in new_matches.columns:
            new_matches[col] = np.nan
    new_matches = new_matches[existing_matches.columns]

    for col in existing_deliveries.columns:
        if col not in new_deliveries.columns:
            new_deliveries[col] = np.nan
    new_deliveries = new_deliveries[existing_deliveries.columns]

    assert new_matches["match_id"].notna().all(), \
        "Null match_id in new matches after alignment — abort"

    # ── Concat and save ───────────────────────────────────────────────
    combined_matches = pd.concat(
        [existing_matches, new_matches], ignore_index=True
    )
    combined_deliveries = pd.concat(
        [existing_deliveries, new_deliveries], ignore_index=True
    )

    combined_matches.to_csv(matches_path,    index=False)
    combined_deliveries.to_csv(deliveries_path, index=False)

    print(f"\nDone:")
    print(f"  matches.csv    : {len(combined_matches):,} rows "
          f"(added {len(new_matches)})")
    print(f"  deliveries.csv : {len(combined_deliveries):,} rows "
          f"(added {len(new_deliveries):,})")
    print(f"  New seasons    : "
          f"{sorted(new_matches['season'].unique())}")
    print(f"  toss_winner nulls : "
          f"{new_matches['toss_winner'].isna().sum()}")
    print(f"  winner nulls      : "
          f"{new_matches['winner'].isna().sum()}")


# ── CLI ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    args  = [a for a in sys.argv[1:] if not a.startswith("--")]
    flags = [a for a in sys.argv[1:] if     a.startswith("--")]

    force_refresh = [int(a) for a in args] if "--force" in flags else None
    seasons       = [int(a) for a in args] if "--force" not in flags else None

    append_new_seasons(
        seasons_to_add=seasons,
        force=force_refresh,
    )