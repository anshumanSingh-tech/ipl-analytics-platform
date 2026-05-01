import joblib, json
import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

# ── Paths ─────────────────────────────────────────────────────────────
_BASE    = Path(__file__).parent.parent   # project root
MODELS   = _BASE / "data" / "processed" / "models"
DATA     = _BASE / "data" / "processed"

# ── Load artefacts once at import time ────────────────────────────────
_wp_model  = joblib.load(MODELS / "win_prob_model.pkl")
_auc_model = joblib.load(MODELS / "auction_model.pkl")

with open(MODELS / "win_prob_features.json")  as f: _wp_features  = json.load(f)
with open(MODELS / "auction_features.json")   as f: _auc_features = json.load(f)

_matches    = pd.read_csv(DATA   / "matches_clean.csv",   parse_dates=["date"])
_deliveries = pd.read_csv(DATA   / "deliveries_clean.csv", low_memory=False)


# ── Internal helpers ──────────────────────────────────────────────────

def _recent_form(team: str, past: pd.DataFrame, n: int = 5) -> float:
    t = past[(past["team1"]==team)|(past["team2"]==team)].tail(n)
    if len(t) == 0:
        return 0.5
    return round((t["winner"]==team).sum() / len(t), 4)


def _h2h_win_rate(team_a: str, team_b: str, past: pd.DataFrame) -> tuple:
    h = past[
        ((past["team1"]==team_a)&(past["team2"]==team_b))|
        ((past["team1"]==team_b)&(past["team2"]==team_a))
    ]
    if len(h) == 0:
        return 0.5, 0
    return round((h["winner"]==team_a).sum() / len(h), 4), len(h)


def _venue_win_rate(team: str, venue: str, past: pd.DataFrame) -> float:
    v  = past[past["venue"]==venue]
    vt = v[(v["team1"]==team)|(v["team2"]==team)]
    if len(vt) == 0:
        return 0.5
    return round((vt["winner"]==team).sum() / len(vt), 4)


def _team_avg_score(team: str, past_ids: list) -> float:
    d = _deliveries[_deliveries["match_id"].isin(past_ids)]
    if d.empty:
        return 150.0
    scores = d[d["batting_team"]==team].groupby("match_id")["total_runs"].sum()
    return round(scores.mean() if len(scores) > 0 else 150.0, 2)


def _team_avg_wickets(team: str, past_ids: list) -> float:
    d = _deliveries[_deliveries["match_id"].isin(past_ids)]
    if d.empty:
        return 7.0
    wkts = d[d["bowling_team"]==team].groupby("match_id")["is_wicket"].sum()
    return round(wkts.mean() if len(wkts) > 0 else 7.0, 2)


def _form_std(team: str, past: pd.DataFrame, n: int = 8) -> float:
    t = past[(past["team1"]==team)|(past["team2"]==team)].tail(n)
    if len(t) < 3:
        return 0.3
    return round(float(np.std((t["winner"]==team).astype(int).tolist())), 4)


# ── Public API ────────────────────────────────────────────────────────

def get_available_teams() -> list:
    """Return sorted list of all IPL team names in the dataset."""
    all_teams = pd.concat([
        _matches["team1"], _matches["team2"]
    ]).dropna().unique()
    return sorted(all_teams.tolist())


def get_available_venues() -> list:
    """Return sorted list of all venues in the dataset."""
    return sorted(_matches["venue"].dropna().unique().tolist())


def predict_winner(team1: str, team2: str, venue: str,
                   toss_winner: str, toss_decision: str,
                   season: int = None) -> dict:
    """
    Predict win probability for a match before it starts.

    team_A is always the toss winner — consistent with how the model
    was trained (toss_winner_won as target).

    Returns
    -------
    dict with keys:
        toss_winner      : str
        other_team       : str
        toss_winner_prob : float  (0–100)
        other_team_prob  : float  (0–100)
        predicted_winner : str
        key_factors      : dict
    """
    
    
    past     = _matches[
        _matches["winner"].notna() &
        (_matches["winner"] != "No Result")
    ].copy()
    past_ids = past["match_id"].tolist()

    team_A = toss_winner
    team_B = team2 if toss_winner == team1 else team1
    
    
    def _venue_exp(team):
        v = past[past["venue"] == venue]
        vt = v[(v["team1"] == team) | (v["team2"] == team)]
        return len(vt)
    
    A_venue_exp = _venue_exp(team_A)
    B_venue_exp = _venue_exp(team_B)
    venue_exp_diff = A_venue_exp - B_venue_exp

    A_form5    = _recent_form(team_A, past, 5)
    B_form5    = _recent_form(team_B, past, 5)
    A_form10   = _recent_form(team_A, past, 10)
    B_form10   = _recent_form(team_B, past, 10)
    A_overall  = _recent_form(team_A, past, len(past))
    B_overall  = _recent_form(team_B, past, len(past))
    A_venue    = _venue_win_rate(team_A, venue, past)
    B_venue    = _venue_win_rate(team_B, venue, past)
    A_score    = _team_avg_score(team_A,   past_ids)
    B_score    = _team_avg_score(team_B,   past_ids)
    A_wkts     = _team_avg_wickets(team_A, past_ids)
    B_wkts     = _team_avg_wickets(team_B, past_ids)
    h2h_wr, h2h_n = _h2h_win_rate(team_A, team_B, past)
    A_cons     = _form_std(team_A, past)
    B_cons     = _form_std(team_B, past)

    v_past = past[past["venue"] == venue]
    if len(v_past) >= 5 and "batting_first_won" in v_past.columns:
        vbfwr = round(
            (v_past["batting_first_won"]==1).sum() / len(v_past), 4
        )
    else:
        vbfwr = 0.5

    toss_correct = int(
        (toss_decision == "bat"   and vbfwr >= 0.5) or
        (toss_decision == "field" and vbfwr <  0.5)
    )

    features = {
        "A_form5"           : A_form5,
        "B_form5"           : B_form5,
        "form_diff5"        : round(A_form5  - B_form5,  4),
        "A_form10"          : A_form10,
        "B_form10"          : B_form10,
        "form_diff10"       : round(A_form10 - B_form10, 4),
        "A_overall_wr"      : A_overall,
        "B_overall_wr"      : B_overall,
        "overall_wr_diff"   : round(A_overall - B_overall, 4),
        "A_venue_wr"        : A_venue,
        "B_venue_wr"        : B_venue,
        "venue_wr_diff"     : round(A_venue - B_venue, 4),
        "A_venue_exp"       : A_venue_exp,
        "B_venue_exp"       : B_venue_exp,
        "venue_exp_diff"    : venue_exp_diff,
        "venue_bat_first_wr": vbfwr,
        "A_avg_score"       : A_score,
        "B_avg_score"       : B_score,
        "score_diff"        : round(A_score - B_score, 2),
        "A_avg_wickets"     : A_wkts,
        "B_avg_wickets"     : B_wkts,
        "wicket_diff"       : round(A_wkts  - B_wkts,  2),
        "h2h_wr_A"          : h2h_wr,
        "h2h_n"             : h2h_n,
        "toss_decision_bat" : 1 if toss_decision == "bat" else 0,
        "toss_correct"      : toss_correct,
        "A_consistency"     : A_cons,
        "B_consistency"     : B_cons,
        "season_stage"      : 1,
    }

    df   = pd.DataFrame([features])[_wp_features]
    prob = _wp_model.predict_proba(df)[0]
    a_prob = round(float(prob[1]) * 100, 1)
    b_prob = round(100 - a_prob, 1)

    return {
        "toss_winner"     : team_A,
        "other_team"      : team_B,
        "toss_winner_prob": a_prob,
        "other_team_prob" : b_prob,
        "predicted_winner": team_A if a_prob >= 50 else team_B,
        "key_factors"     : {
            "form_edge" : team_A if A_form5   > B_form5   else team_B,
            "venue_edge": team_A if A_venue   > B_venue   else team_B,
            "h2h_edge"  : team_A if h2h_wr    > 0.5       else team_B,
            "score_edge": team_A if A_score   > B_score   else team_B,
        },
    }


def predict_auction_value(player_stats: dict) -> dict:
    """
    Predict IPL auction price given a player stats dictionary.

    Returns
    -------
    dict with keys:
        predicted_price_cr : float
        tier               : str
    """
    defaults = {
        "total_runs":0, "batting_average":0, "strike_rate":0,
        "hundreds":0, "fifties":0, "boundary_rate":0,
        "sr_powerplay":110, "sr_death":120, "dot_ball_pct_bat":30,
        "wickets":0, "economy_rate":10.5, "bowling_average":50,
        "bowling_sr":40, "dot_ball_pct_bowl":30, "economy_death":11,
        "economy_powerplay":9, "three_wicket_haul":0,
        "matches_batted":0, "matches_bowled":0, "role":0,
    }
    defaults.update(player_stats)

    df    = pd.DataFrame([defaults])[_auc_features]
    price = round(float(_auc_model.predict(df)[0]), 2)
    price = max(0.2, price)

    if price >= 12: tier = "Icon (12 Cr+)"
    elif price >= 7: tier = "Premium (7–12 Cr)"
    elif price >= 3: tier = "Standard (3–7 Cr)"
    else:            tier = "Emerging (< 3 Cr)"

    return {"predicted_price_cr": price, "tier": tier}