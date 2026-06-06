from dash import Input, Output, State, callback, no_update, dash_table
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dash import html
import dash_bootstrap_components as dbc
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models import predict_winner, predict_auction_value

PROCESSED = Path(__file__).parent.parent / "data" / "processed"
MODELS    = PROCESSED / "models"

matches    = pd.read_csv(PROCESSED / "matches_clean.csv",  parse_dates=["date"])
batting    = pd.read_csv(PROCESSED / "batting_features.csv")
bowling    = pd.read_csv(PROCESSED / "bowling_features.csv")
deliveries = pd.read_csv(PROCESSED / "deliveries_clean.csv", low_memory=False)
valuations = pd.read_csv(MODELS    / "player_valuations.csv")
summary = pd.read_csv(PROCESSED / "match_summary.csv", parse_dates=["date"])
print("batting_first_won in summary:", "batting_first_won" in summary.columns)
print("summary shape:", summary.shape)

TEAM_COLORS = {
    "Mumbai Indians"             : "#004BA0",
    "Chennai Super Kings"        : "#F9CD05",
    "Royal Challengers Bangalore": "#EC1C24",
    "Kolkata Knight Riders"      : "#3A225D",
    "Sunrisers Hyderabad"        : "#F7A721",
    "Delhi Capitals"             : "#0078BC",
    "Punjab Kings"               : "#ED1B24",
    "Rajasthan Royals"           : "#254AA5",
}
DEFAULT_C = "#7F77DD"

def tc(team): return TEAM_COLORS.get(team, DEFAULT_C)

CHART_LAYOUT = dict(
    plot_bgcolor="white",
    paper_bgcolor="white",
    font_family="Segoe UI, Arial",
    margin=dict(t=40, b=40, l=40, r=20),
)



def register_callbacks(app):

    @app.callback(
        Output("team-metric-cards",   "children"),
        Output("wins-per-season-chart","figure"),
        Output("win-pct-chart",        "figure"),
        Output("toss-chart",           "figure"),
        Output("runs-per-over-chart",  "figure"),
        Input("team-filter",   "value"),
        Input("season-range",  "value"),
    )
    def update_team_stats(selected_teams, season_range):
        
        print(f"update_team_stats called")
        print(f"  selected_teams : {selected_teams}")
        print(f"  season_range   : {season_range}")
        print(f"  matches shape  : {matches.shape}")
        
        if not selected_teams:
            selected_teams = ["Mumbai Indians", "Chennai Super Kings"]

        s_min, s_max = season_range
        df = matches[
            (matches["season"] >= s_min) &
            (matches["season"] <= s_max)
        ].copy()

        # ── Metric cards ──────────────────────────────────────────
        total_matches = len(df)
        total_seasons = df["season"].nunique()
        total_sixes   = int(deliveries[
            deliveries["match_id"].isin(df["match_id"])
        ]["is_six"].sum())
        avg_score = deliveries[
            deliveries["match_id"].isin(df["match_id"])
        ].groupby("match_id")["total_runs"].sum().mean()

        def metric_card(value, label):
            return dbc.Col(html.Div([
                html.P(str(value), className="metric-value"),
                html.P(label,      className="metric-label"),
            ], className="metric-card"), xs=6, md=3)

        cards = [
            metric_card(total_matches,        "total matches"),
            metric_card(total_seasons,         "seasons"),
            metric_card(f"{total_sixes:,}",    "total sixes"),
            metric_card(f"{avg_score:.0f}",    "avg match score"),
        ]

        # ── Wins per season ───────────────────────────────────────
        wins = (
            df[df["winner"].isin(selected_teams)]
            .groupby(["season", "winner"])
            .size()
            .reset_index(name="wins")
            .rename(columns={"winner": "team"})
        )
        fig_wins = px.bar(
            wins, x="season", y="wins", color="team",
            barmode="group",
            color_discrete_map={t: tc(t) for t in wins["team"].unique()},
            labels={"wins": "wins", "season": "season"},
        )
        fig_wins.update_layout(**CHART_LAYOUT)
        fig_wins.update_layout(
            xaxis=dict(tickmode="linear", dtick=1, tickangle=-45),
            legend=dict(orientation="h", y=-0.25, font_size=11),
        )
        fig_wins.update_traces(marker_line_width=0)

        # ── Win percentage ────────────────────────────────────────
        records = []
        for team in selected_teams:
            played = df[(df["team1"] == team) | (df["team2"] == team)]
            won    = df[df["winner"] == team]
            if len(played) > 0:
                records.append({
                    "team"   : team,
                    "win_pct": round(len(won) / len(played) * 100, 1),
                    "played" : len(played),
                })
        wp_df  = pd.DataFrame(records).sort_values("win_pct", ascending=True)
        fig_wp = px.bar(
            wp_df, x="win_pct", y="team", orientation="h",
            color="team",
            color_discrete_map={t: tc(t) for t in wp_df["team"].unique()},
            labels={"win_pct": "win %", "team": ""},
            text="win_pct",
        )
        fig_wp.update_traces(texttemplate="%{text:.1f}%", textposition="outside",
                              marker_line_width=0)
        fig_wp.update_layout(**CHART_LAYOUT)
        fig_wp.update_layout(showlegend=False, xaxis=dict(range=[0, 100]))

        # ── Toss decisions ────────────────────────────────────────
        toss_df = (
            df[df["toss_winner"].isin(selected_teams)]
            .groupby(["toss_winner", "toss_decision"])
            .size()
            .reset_index(name="count")
            .rename(columns={"toss_winner": "team"})
        )
        fig_toss = px.bar(
            toss_df, x="team", y="count", color="toss_decision",
            barmode="group",
            color_discrete_map={"bat": "#7F77DD", "field": "#1D9E75"},
            labels={"count": "times chosen", "toss_decision": "decision"},
        )
        fig_toss.update_layout(**CHART_LAYOUT)
        fig_toss.update_layout(
            xaxis_tickangle=-30,
            legend=dict(orientation="h", y=-0.3, font_size=11),
        )
        fig_toss.update_traces(marker_line_width=0)

        # ── Runs per over ─────────────────────────────────────────
        over_avg = (
            deliveries[deliveries["match_id"].isin(df["match_id"])]
            .groupby("over")["total_runs"]
            .mean()
            .round(2)
            .reset_index()
        )
        PHASE_BG = [
            dict(type="rect", xref="x", yref="paper",
                 x0=0.5, x1=6.5, y0=0, y1=1,
                 fillcolor="#EEEDFE", opacity=0.3, layer="below", line_width=0),
            dict(type="rect", xref="x", yref="paper",
                 x0=6.5, x1=15.5, y0=0, y1=1,
                 fillcolor="#E1F5EE", opacity=0.3, layer="below", line_width=0),
            dict(type="rect", xref="x", yref="paper",
                 x0=15.5, x1=20.5, y0=0, y1=1,
                 fillcolor="#FAECE7", opacity=0.3, layer="below", line_width=0),
        ]
        fig_rpo = go.Figure()
        fig_rpo.add_trace(go.Bar(
            x=over_avg["over"], y=over_avg["total_runs"],
            marker_color="#B5D4F4", name="avg runs",
        ))
        for label, xpos in [("Powerplay", 3.5), ("Middle", 11), ("Death", 18)]:
            fig_rpo.add_annotation(
                x=xpos, y=over_avg["total_runs"].max() * 1.08,
                text=label, showarrow=False,
                font=dict(size=10, color="#444"),
            )
        fig_rpo.update_layout(
            **CHART_LAYOUT,
            shapes=PHASE_BG,
            xaxis=dict(title="over", tickmode="linear", dtick=1),
            yaxis_title="avg runs/over",
            showlegend=False,
        )

        return cards, fig_wins, fig_wp, fig_toss, fig_rpo


    # ════════════════════════════════════════════════════════════════
    # PAGE 2 CALLBACKS — PLAYER EXPLORER
    # ════════════════════════════════════════════════════════════════

    @app.callback(
        Output("batting-metric-cards", "children"),
        Output("batting-scatter-chart","figure"),
        Output("phase-sr-chart",        "figure"),
        Input("batsman-search",  "value"),
        Input("batsman-compare", "value"),
    )
    def update_batting(player1, player2):
        players = [p for p in [player1, player2] if p]

        # ── Metric cards for primary player ───────────────────────
        p1 = batting[batting["batsman"] == player1].iloc[0]

        def bat_card(value, label):
            return dbc.Col(html.Div([
                html.P(str(value), className="metric-value"),
                html.P(label,      className="metric-label"),
            ], className="metric-card"), xs=6, md=2)

        cards = [
            bat_card(f"{p1['total_runs']:,.0f}",    "career runs"),
            bat_card(f"{p1['batting_average']:.1f}","average"),
            bat_card(f"{p1['strike_rate']:.1f}",    "strike rate"),
            bat_card(int(p1["hundreds"]),             "100s"),
            bat_card(int(p1["fifties"]),              "50s"),
            bat_card(f"{p1['boundary_rate']:.1f}%",  "boundary %"),
        ]

        # ── Batting scatter: all players, highlight selected ──────
        plot_df = batting[batting["balls_faced"] >= 200].copy()
        plot_df["highlight"] = plot_df["batsman"].apply(
            lambda x: x if x in players else "others"
        )
        plot_df = plot_df.sort_values(
            "highlight",
            key=lambda s: s.map({"others": 0}).fillna(1)
        )
        color_map = {"others": "#D3D1C7"}
        colors    = ["#7F77DD", "#D85A30"]
        for i, p in enumerate(players):
            color_map[p] = colors[i % len(colors)]

        fig_scatter = px.scatter(
            plot_df,
            x="batting_average", y="strike_rate",
            color="highlight",
            color_discrete_map=color_map,
            size="total_runs", size_max=22,
            hover_name="batsman",
            hover_data={"total_runs": True, "highlight": False},
            labels={
                "batting_average": "batting average",
                "strike_rate"    : "strike rate",
                "highlight"      : "player",
            },
        )
        fig_scatter.update_layout(**CHART_LAYOUT)
        fig_scatter.update_layout(
            legend=dict(orientation="h", y=-0.2, font_size=11)
        )
        fig_scatter.update_traces(marker_line_width=0.5,
                                   marker_line_color="white")

        # ── Phase SR comparison ────────────────────────────────────
        phase_data = []
        for p in players:
            row = batting[batting["batsman"] == p]
            if row.empty:
                continue
            row = row.iloc[0]
            for phase in ["powerplay", "middle", "death"]:
                col = f"sr_{phase}"
                if col in row and pd.notna(row[col]):
                    phase_data.append({
                        "player": p,
                        "phase" : phase.capitalize(),
                        "SR"    : round(row[col], 1),
                    })

        if phase_data:
            ph_df   = pd.DataFrame(phase_data)
            fig_phase = px.bar(
                ph_df, x="phase", y="SR", color="player",
                barmode="group",
                color_discrete_map={players[0]: "#7F77DD",
                                    players[1] if len(players) > 1 else "": "#D85A30"},
                labels={"SR": "strike rate", "phase": "match phase"},
                text="SR",
            )
            fig_phase.update_traces(texttemplate="%{text:.1f}", textposition="outside",
                                     marker_line_width=0)
            fig_phase.update_layout(**CHART_LAYOUT)
            fig_phase.update_layout(
                legend=dict(orientation="h", y=-0.2, font_size=11)
            )
        else:
            fig_phase = go.Figure()
            fig_phase.update_layout(**CHART_LAYOUT)

        return cards, fig_scatter, fig_phase


    @app.callback(
        Output("bowling-metric-cards","children"),
        Output("bowling-phase-chart", "figure"),
        Input("bowler-search", "value"),
    )
    def update_bowling(player):
        row = bowling[bowling["bowler"] == player]
        if row.empty:
            return [], go.Figure()
        row = row.iloc[0]

        def bowl_card(value, label):
            return dbc.Col(html.Div([
                html.P(str(value), className="metric-value"),
                html.P(label,      className="metric-label"),
            ], className="metric-card"), xs=6, md=2)

        cards = [
            bowl_card(int(row["wickets"]),                "career wickets"),
            bowl_card(f"{row['economy_rate']:.2f}",       "economy rate"),
            bowl_card(f"{row['bowling_average']:.1f}",    "bowling avg"),
            bowl_card(f"{row['bowling_sr']:.1f}",         "bowling SR"),
            bowl_card(f"{row['dot_ball_pct']:.1f}%",      "dot ball %"),
            bowl_card(int(row["three_wkt_hauls"]) if "three_wkt_hauls" in row else "—",
                      "3-wkt hauls"),
        ]

        phases = ["powerplay", "middle", "death"]
        econ_vals = [row.get(f"economy_{p}", np.nan) for p in phases]

        fig = go.Figure(go.Bar(
            x=[p.capitalize() for p in phases],
            y=[round(e, 2) if pd.notna(e) else 0 for e in econ_vals],
            marker_color=["#7F77DD", "#1D9E75", "#D85A30"],
            text=[f"{e:.2f}" if pd.notna(e) else "N/A" for e in econ_vals],
            textposition="outside",
        ))
        fig.add_hline(y=8.0, line_dash="dot", line_color="#888",
                      annotation_text="8.0 benchmark")
        fig.update_layout(
            **CHART_LAYOUT,
            title=f"{player} — economy by phase",
            yaxis_title="economy rate",
            xaxis_title="match phase",
            showlegend=False,
        )

        return cards, fig


    @app.callback(
        Output("player-table", "data"),
        Output("player-table", "columns"),
        Input("leaderboard-type", "value"),
    )
    def update_player_table(table_type):
        if table_type == "batting":
            cols = ["batsman", "total_runs", "batting_average",
                    "strike_rate", "hundreds", "fifties",
                    "boundary_rate", "matches_batted"]
            df = batting[cols].copy().rename(columns={"batsman": "player"})
            df = df.sort_values("total_runs", ascending=False).round(2)
        else:
            cols = ["bowler", "wickets", "economy_rate",
                    "bowling_average", "bowling_sr",
                    "dot_ball_pct", "matches_bowled"]
            df = bowling[cols].copy().rename(columns={"bowler": "player"})
            df = df.sort_values("wickets", ascending=False).round(2)

        columns = [{"name": c.replace("_", " ").title(),
                    "id": c, "type": "numeric",
                    "format": {"specifier": ",.2f"}}
                   for c in df.columns]
        return df.to_dict("records"), columns


    # ════════════════════════════════════════════════════════════════
    # PAGE 3 CALLBACKS — WIN PREDICTOR
    # ════════════════════════════════════════════════════════════════

    @app.callback(
        Output("prediction-output", "children"),
        Output("win-prob-gauge",    "figure"),
        Output("historical-h2h",    "children"),
        Input("predict-btn",        "n_clicks"),
        State("pred-team1",         "value"),
        State("pred-team2",         "value"),
        State("pred-venue",         "value"),
        State("pred-toss-winner",   "value"),
        State("pred-toss-decision", "value"),
        prevent_initial_call=True,
    )
    def run_prediction(n, team1, team2, venue, toss_winner, toss_decision):
        result = predict_winner(
            team1=team1, team2=team2, venue=venue,
            toss_winner=toss_winner, toss_decision=toss_decision,
        )

        team_A  = result["toss_winner"]       
        team_B  = result["other_team"]       
        p_toss  = result["toss_winner_prob"] 
        p_other = result["other_team_prob"]   
        winner  = result["predicted_winner"]

        # ── Result card ───────────────────────────────────────────
        result_card = html.Div([
            html.P("Predicted winner", style={"color": "#888", "fontSize": "13px",
                                               "marginBottom": "4px"}),
            html.P(winner, className="win-team"),
            html.Div([
                html.Span(f"{team_A}: ", style={"fontWeight": 600}),
                html.Span(f"{p_toss}%",
                           style={"color": tc(team1), "fontWeight": 700,
                                  "fontSize": "18px"}),
                html.Span("   vs   ", style={"color": "#ccc"}),
                html.Span(f"{team_B}: ", style={"fontWeight": 600}),
                html.Span(f"{p_other}%",
                           style={"color": tc(team_B), "fontWeight": 700,
                                  "fontSize": "18px"}),
            ], style={"marginTop": "8px"}),
        ], className="prediction-result")

        # ── Gauge chart ───────────────────────────────────────────
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=p_toss,
            number={"suffix": "%", "font": {"size": 36, "color": tc(team_A)}},
            title={"text": f"{team_A} win probability", "font": {"size": 13}},
            gauge={
                "axis"      : {"range": [0, 100]},
                "bar"       : {"color": tc(team_A)},
                "steps"     : [
                    {"range": [0,  40], "color": "#FCEBEB"},
                    {"range": [40, 60], "color": "#F1EFE8"},
                    {"range": [60, 100],"color": "#E1F5EE"},
                ],
                "threshold" : {
                    "line" : {"color": "#1D9E75", "width": 3},
                    "thickness": 0.75,
                    "value": 50,
                },
            },
        ))
        fig_gauge.update_layout(
            paper_bgcolor="white",
            font_family="Segoe UI, Arial",
            margin=dict(t=60, b=20, l=30, r=30),
        )

        # ── Historical H2H ────────────────────────────────────────
        h2h = matches[
            ((matches["team1"] == team_A) & (matches["team2"] == team_B)) |
            ((matches["team1"] == team_B) & (matches["team2"] == team_A))
        ]
        h2h = h2h[h2h["winner"].isin([team_A, team_B])]
        tA_wins = int((h2h["winner"] == team_A).sum())
        tB_wins = int((h2h["winner"] == team_B).sum())

        h2h_card = dbc.Card(dbc.CardBody([
            html.H6("Historical head-to-head",
                    style={"fontWeight": 600, "marginBottom": "12px"}),
            html.Div([
                html.Div([
                    html.P(str(tA_wins), style={"fontSize": "32px",
                                                 "fontWeight": 700,
                                                 "color": tc(team_A),
                                                 "margin": 0}),
                    html.P(team_A, style={"fontSize": "12px", "color": "#888"}),
                ], style={"textAlign": "center", "flex": 1}),
                html.Div([
                    html.P(f"{len(h2h)} played",
                           style={"fontSize": "13px", "color": "#aaa",
                                  "margin": 0}),
                ], style={"textAlign": "center", "flex": 1,
                           "display": "flex", "alignItems": "center",
                           "justifyContent": "center"}),
                html.Div([
                    html.P(str(tB_wins), style={"fontSize": "32px",
                                                 "fontWeight": 700,
                                                 "color": tc(team_B),
                                                 "margin": 0}),
                    html.P(team_B, style={"fontSize": "12px", "color": "#888"}),
                ], style={"textAlign": "center", "flex": 1}),
            ], style={"display": "flex", "marginTop": "8px"}),
        ]), style={"border": "0.5px solid #e0e0e0", "marginTop": "16px"})
        
        venue_h2h = matches[
            (
            ((matches["team1"] == team_A) & (matches["team2"] == team_B)) |
            ((matches["team1"] == team_B) & (matches["team2"] == team_A))
            ) & 
            (matches["venue"] == venue)
        ]
        venue_h2h = venue_h2h[venue_h2h["winner"].isin([team_A, team_B])]
        vA_wins = int((venue_h2h["winner"] == team_A).sum())
        vB_wins = int((venue_h2h["winner"] == team_B).sum())
        total_venue = len(venue_h2h)
        
        venue_card = dbc.Card(dbc.CardBody([
            html.H6(f"Head-to-Head at {venue.split(',')[0]}",
                    style={"fontWeight": 600, "marginBottom": "12px"}),
            html.Div([
                html.Div([
                    html.P(str(vA_wins),
                           style={"fontSize": "28px", "fontWeight": 700,
                                  "color": tc(team_A), "margin": 0}),
                    html.P(team_A, style={"fontSize": "11px", "color": "#888" }),
                ], style={"textAlign": "center", "flex": 1}),
                html.Div([
                    html.P(f"{total_venue} played" if total_venue > 0
                           else "No matches\nat this value",
                           style={"fontSize": "12px", "color": "#aaa",
                                  "margin": 0, "textAlign": "center"}),
                ], style={"flex": 1, "display": "flex",
                          "alignItems": "center", "justifyContent": "center"}),
                html.Div([
                    html.P(str(vB_wins),
                           style={"fontSize": "28px", "fontWeight": 700,
                                  "color": tc(team_B), "margin": 0}),
                    html.P(team_B, style={"fontSize": "11px", "color": "#888"}),
                ], style={"textAlign": "center", "flex": 1}),
            ], style={"display": "flex"}),
            html.P(
                    "No historical data at this venue - result based on overall form only."
                    if total_venue == 0 else
                    f"Sample size: {total_venue} matches. "
                    + ("Reliable Signal." if total_venue >= 10 else "Small sample - treat with caution"),
                    style={"fontSize": "11px", "color": "#aaa",
                           "marginTop": "8px", "marginBottom": 0}
            ),
        ]), style={"border": "0.5px solid #e0e0e0", "marginTop": "12px"})

        return result_card, fig_gauge, html.Div([h2h_card, venue_card])


    # ════════════════════════════════════════════════════════════════
    # PAGE 4 CALLBACKS — AUCTION SIMULATOR
    # ════════════════════════════════════════════════════════════════

    # Slider output labels
    SLIDER_LABEL_MAP = {
        "sl-runs"    : "sl-runs-out",     "sl-avg"     : "sl-avg-out",
        "sl-sr"      : "sl-sr-out",       "sl-100s"    : "sl-100s-out",
        "sl-50s"     : "sl-50s-out",      "sl-br"      : "sl-br-out",
        "sl-pp-sr"   : "sl-pp-sr-out",    "sl-death-sr": "sl-death-sr-out",
        "sl-wkts"    : "sl-wkts-out",     "sl-econ"    : "sl-econ-out",
        "sl-bowl-avg": "sl-bowl-avg-out", "sl-dot"     : "sl-dot-out",
        "sl-d-econ"  : "sl-d-econ-out",   "sl-pp-econ" : "sl-pp-econ-out",
        "sl-mbowled" : "sl-mbowled-out",
    }

    @app.callback(
        [Output(v, "children") for v in SLIDER_LABEL_MAP.values()] +
        [
            Output("auction-price-display",  "children"),
            Output("auction-gauge",          "figure"),
            Output("similar-players-table",  "children"),
            Output("valuation-chart",        "figure"),
        ],
        [Input(k, "value") for k in SLIDER_LABEL_MAP.keys()],
    )
    def update_auction(runs, avg, sr, h100, h50, br, pp_sr, death_sr,
                       wkts, econ, bowl_avg, dot, d_econ, pp_econ, mbowled):

        labels = [runs, avg, sr, h100, h50, br, pp_sr, death_sr,
                  wkts, econ, bowl_avg, dot, d_econ, pp_econ, mbowled]

        stats = {
            "total_runs"       : runs,      "batting_average"  : avg,
            "strike_rate"      : sr,        "hundreds"         : h100,
            "fifties"          : h50,        "boundary_rate"    : br,
            "sr_powerplay"     : pp_sr,      "sr_death"         : death_sr,
            "wickets"          : wkts,       "economy_rate"     : econ,
            "bowling_average"  : bowl_avg,   "dot_ball_pct"     : dot,
            "economy_death"    : d_econ,     "economy_powerplay": pp_econ,
            "matches_bowled"   : mbowled,    "matches_batted"   : max(runs // 30, 1),
            "bowling_sr"       : 30.0,
        }
        result = predict_auction_value(stats)
        price  = result["predicted_price_cr"]
        tier   = result["tier"]

        tier_color = {
            "Icon (12 cr+)"    : "#085041",
            "Premium (7–12 cr)": "#1D9E75",
            "Standard (3–7 cr)": "#7F77DD",
            "Emerging (< 3 cr)": "#888780",
        }.get(tier, "#888780")

        price_display = html.Div([
            html.P(f"₹ {price:.2f} Cr",
                   style={"fontSize": "42px", "fontWeight": 700,
                           "color": tier_color, "margin": 0}),
            html.Span(tier, style={
                "fontSize": "13px", "padding": "4px 14px",
                "background": tier_color, "color": "white",
                "borderRadius": "999px", "fontWeight": 500,
            }),
        ], style={"textAlign": "center", "padding": "16px 0"})

        # ── Gauge ──────────────────────────────────────────────────
        fig_g = go.Figure(go.Indicator(
            mode="gauge+number",
            value=min(price, 20),
            number={"prefix": "₹ ", "suffix": " Cr",
                    "font": {"size": 28, "color": tier_color}},
            gauge={
                "axis" : {"range": [0, 20],
                           "tickvals": [0, 3, 7, 12, 20],
                           "ticktext": ["0", "3cr", "7cr", "12cr", "20cr+"]},
                "bar"  : {"color": tier_color},
                "steps": [
                    {"range": [0,  3],  "color": "#F1EFE8"},
                    {"range": [3,  7],  "color": "#EEEDFE"},
                    {"range": [7,  12], "color": "#E1F5EE"},
                    {"range": [12, 20], "color": "#9FE1CB"},
                ],
            },
        ))
        fig_g.update_layout(
            paper_bgcolor="white",
            font_family="Segoe UI, Arial",
            margin=dict(t=20, b=10, l=30, r=30),
            height=220,
        )

        # ── Similar players ────────────────────────────────────────
        sim = valuations.copy()
        sim = sim[
            (sim["predicted_price_cr"] >= price * 0.75) &
            (sim["predicted_price_cr"] <= price * 1.25)
        ].nlargest(5, "predicted_price_cr")[
            ["player", "total_runs", "wickets", "predicted_price_cr"]
        ].round(2)

        sim_table = dash_table.DataTable(
            data=sim.to_dict("records"),
            columns=[
                {"name": "Player",          "id": "player"},
                {"name": "Runs",            "id": "total_runs"},
                {"name": "Wickets",         "id": "wickets"},
                {"name": "Est. Price (Cr)", "id": "predicted_price_cr"},
            ],
            style_header={"backgroundColor": "#1a1a2e", "color": "white",
                           "fontSize": "11px", "fontWeight": 600},
            style_cell={"fontSize": "12px", "padding": "6px 10px",
                         "border": "0.5px solid #f0f0f0"},
            style_data_conditional=[
                {"if": {"row_index": "odd"}, "backgroundColor": "#fafafa"}
            ],
        )

        # ── Valuation leaderboard chart ────────────────────────────
        top_v = valuations.nlargest(25, "predicted_price_cr")[
            ["player", "predicted_price_cr", "total_runs", "wickets"]
        ].sort_values("predicted_price_cr")

        fig_val = px.bar(
            top_v, x="predicted_price_cr", y="player",
            orientation="h",
            color="predicted_price_cr",
            color_continuous_scale=["#EEEDFE", "#7F77DD", "#3C3489"],
            labels={"predicted_price_cr": "est. price (₹ Cr)", "player": ""},
            text="predicted_price_cr",
        )
        fig_val.update_traces(texttemplate="₹%{text:.1f}Cr",
                               textposition="outside")
        fig_val.update_layout(
            **CHART_LAYOUT,
            coloraxis_showscale=False,
            title="Top 25 most valuable IPL players (model estimate)",
        )

        return labels + [price_display, fig_g, sim_table, fig_val]