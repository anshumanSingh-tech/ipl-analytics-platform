import pandas as pd
import numpy as np

def build_batting_features(deliveries: pd.DataFrame) -> pd.DataFrame:
    
    d = deliveries[deliveries["is_wide"] == 0].copy()
    
    agg = d.groupby("batsman").agg(
        total_runs = ("batsman_runs", "sum"),
        balls_faced = ("is_legal_delivery", "sum"),
        matches_batted = ("match_id", "nunique"),
        innings = ("inning", "count"),
        fours = ("is_four", "sum"),
        sixes = ("is_six", "sum"),
        dot_balls = ("is_dot_ball", "sum"),
        dismissals = ("is_wicket", "sum"),
    ).reset_index()
    
    agg["strike_rate"] = (agg["total_runs"] / agg["balls_faced"].replace(0, np.nan) * 100).round(2)
    agg["batting_average"] = (agg["total_runs"] / agg["dismissals"].replace(0, np.nan)).round(2)
    agg["boundary_rate"] = ((agg["fours"] + agg["sixes"]) / agg["balls_faced"].replace(0, np.nan) * 100).round(2)
    agg["dot_ball_pct"] = (agg["dot_balls"] / agg["balls_faced"].replace(0, np.nan) * 100).round(2)
    agg["runs_per_match"] = (agg["total_runs"] / agg["matches_batted"].replace(0, np.nan)).round(2)
    
    for phase in ["powerplay", "middle", "death"]:
        phase_df = d[d["over_phase"] == phase].groupby("batsman").agg(
            _runs = ("batsman_runs", "sum"),
            _balls = ("is_legal_delivery", "sum"),
        ).reset_index()
        phase_df[f"sr_{phase}"] = (phase_df["_runs"] / phase_df["_balls"].replace(0, np.nan) * 100).round(2)
        agg = agg.merge(phase_df[["batsman", f"sr_{phase}"]], on="batsman", how="left")
        
    inning_scores = d.groupby(["batsman", "match_id", "inning"])["batsman_runs"].sum().reset_index()
    inning_scores.columns = ["batsman", "match_id", "inning", "innings_runs"]
    
    fifties = inning_scores[inning_scores["innings_runs"].between(50, 99)].groupby("batsman").size().rename("fifties")
    hundreds = inning_scores[inning_scores["innings_runs"] >= 100].groupby("batsman").size().rename("hundreds")
    
    agg = agg.merge(fifties, on="batsman", how="left")
    agg = agg.merge(hundreds, on="batsman", how="left")
    agg["fifties"] = agg["fifties"].fillna(0).astype(int)
    agg["hundreds"] = agg["hundreds"].fillna(0).astype(int)
    
    agg = agg[agg["balls_faced"] >= 30].copy()
    
    print(f"build_batting_features: {len(agg)} batsmans with 30+ balls faced")
    return agg.sort_values("total_runs", ascending=False).reset_index(drop=True)


def build_bowling_features(deliveries: pd.DataFrame) -> pd.DataFrame:
    d = deliveries.copy()
    
    agg = d.groupby("bowler").agg(
        balls_bowled = ("is_legal_delivery", "sum"),
        runs_conceded = ("total_runs", "sum"),
        wickets = ("is_wicket", "sum"),
        dot_balls = ("is_dot_ball", "sum"),
        wides = ("is_wide", "sum"),
        no_balls = ("is_noball", "sum"),
        matches_bowled = ("match_id", "nunique"),
        fours_conceded = ("is_four", "sum"),
        sixes_conceded = ("is_six", "sum"),
    ).reset_index()
    
    overs = agg["balls_bowled"] / 6
    agg["economy_rate"] = (agg["runs_conceded"] / overs.replace(0, np.nan)).round(2)
    agg["bowling_average"] = (agg["runs_conceded"] / agg["wickets"].replace(0, np.nan)).round(2)
    agg["bowling_sr"] = (agg["balls_bowled"] / agg["wickets"].replace(0, np.nan)).round(2)
    agg["dot_ball_pct"] = (agg["dot_balls"] / agg["balls_bowled"].replace(0, np.nan) * 100).round(2)
    agg["wickets_per_match"] = (agg["wickets"] / agg["matches_bowled"].replace(0, np.nan)).round(2)
    
    for phase in  ["powerplay", "middle", 'death']:
        phase_df = d[d["over_phase"] == phase].groupby("bowler").agg(
            _runs = ("total_runs", "sum"),
            _balls = ("is_legal_delivery", "sum"),
        ).reset_index()
        phase_df[f"economy_{phase}"] = (phase_df["_runs"] / (phase_df["_balls"] / 6).replace(0, np.nan)).round(2)
        agg = agg.merge(phase_df[["bowler", f"economy_{phase}"]], on="bowler", how="left")
        
    inning_wkt = d.groupby(["bowler", "match_id", "inning"])["is_wicket"].sum().reset_index()
    inning_wkt.columns = ["bowler", "match_id", "inning", "wickets_in_inning"]
        
    three_wkt = inning_wkt[inning_wkt["wickets_in_inning"] >= 3].groupby("bowler").size().rename("three_wicket_haul")
    five_wkt = inning_wkt[inning_wkt["wickets_in_inning"] >= 5].groupby("bowler").size().rename("five_wicket_haul")
        
    agg = agg.merge(three_wkt, on="bowler", how="left")
    agg = agg.merge(five_wkt, on="bowler", how="left")
    agg["three_wicket_haul"] = agg["three_wicket_haul"].fillna(0).astype(int)
    agg["five_wicket_haul"] = agg["five_wicket_haul"].fillna(0).astype(int)
        
    agg = agg[agg["balls_bowled"] >= 60].copy()
        
    print(f"build_bowling_features: {len(agg)} bowlers with 60+ balls bowled")
    return agg.sort_values("wickets", ascending=False).reset_index(drop=True)
    
def build_match_summary(matches: pd.DataFrame, deliveries: pd.DataFrame) -> pd.DataFrame:
    
    innings_total = deliveries.groupby(["match_id", "inning", "batting_team"]).agg(
        total_runs = ("total_runs", "sum"),
        total_wickets = ("is_wicket", "sum"),
        total_balls = ("is_legal_delivery", "sum"),
        total_fours = ("is_four", "sum"),
        total_sixes = ("is_six", "sum"),
        dot_balls = ("is_dot_ball", "sum"),
    ).reset_index()
    
    innings_total["run_rate"] = (
        innings_total["total_runs"] / (innings_total["total_balls"] / 6).replace(0, np.nan)
    ).round(2)
    
    inn1 = innings_total[innings_total["inning"] == 1].add_prefix("inn1_").rename(columns={"inn1_match_id": "match_id"})
    inn2 = innings_total[innings_total["inning"] == 2].add_prefix("inn2_").rename(columns={"inn2_match_id": "match_id"})
    
    summary = matches.merge(inn1.drop(columns=["inn1_inning"]), on="match_id", how="left")
    summary = summary.merge(inn2.drop(columns=["inn2_inning"]), on="match_id", how="left")
    
    summary["toss_winner_won"] = (summary["toss_winner"] == summary["winner"]).astype(int)
    summary["batting_first_team"] = summary["inn1_batting_team"]
    
    summary["batting_first_won"] = (
        summary["inn1_batting_team"] == summary["winner"]
    ).astype(int)
    
    summary["score_diff"] = summary["inn1_total_runs"] - summary["inn2_total_runs"]
    
    min_season = summary["season"].min()
    summary["season_num"] = summary["season"] - min_season + 1
    
    print(f"build_match_summary: {summary.shape}")
    return summary.reset_index(drop=True)

def build_team_season_stats(matches: pd.DataFrame) -> pd.DataFrame:
    
    records = []
    for season in sorted(matches["season"].unique()):
        season_df = matches[matches["season"] == season]
        all_teams = pd.concat([season_df["team1"], season_df["team2"]]).unique()
        
        for team in all_teams:
            played = season_df[(season_df["team1"] == team) | (season_df["team2"] == team)]
            won = season_df[season_df["winner"] == team]
            
            bat_first = played[played["inn1_batting_team"] == team] if "inn1_batting_team" in played.columns else pd.DataFrame()
            toss_won = played[played["toss_winner"] == team]
            toss_bat = toss_won[toss_won["toss_decision"] == "bat"]
            
            records.append({
                "season": season,
                "team": team,
                "matches_played": len(played),
                "matches_won": len(won),
                "win_pct": round(len(won) / len(played) * 100, 1) if len(played) > 0 else 0,
                "toss_wins": len(toss_won),
                "toss_bat_choice": len(toss_bat),
            })
            
    df = pd.DataFrame(records)
    print(f"build_team_season_stats: {df.shape}")
    return df