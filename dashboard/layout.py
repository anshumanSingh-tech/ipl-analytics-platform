from dash import dcc, html, dash_table
import dash_bootstrap_components as dbc
import pandas as pd
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))


from src.models import get_available_teams, get_available_venues

PROCESSED = Path(__file__).parent.parent / "data" / "processed"
MODELS = PROCESSED / "models"

# ── Load static data for dropdowns ───────────────────────────────────
team_season = pd.read_csv(PROCESSED / "team_season_stats.csv")
matches     = pd.read_csv(PROCESSED / "matches_clean.csv")
batting     = pd.read_csv(PROCESSED / "batting_features.csv")
bowling     = pd.read_csv(PROCESSED / "bowling_features.csv")
valuations  = pd.read_csv(MODELS    / "player_valuations.csv")

ALL_VENUES  = get_available_venues()
ALL_TEAMS   = get_available_teams()
ALL_SEASONS = sorted(matches["season"].unique().tolist())


TOP_TEAMS = [
    "Mumbai Indians", "Chennai Super Kings",
    "Royal Challengers Bangalore", "Kolkata Knight Riders",
    "Sunrisers Hyderabad", "Delhi Capitals",
    "Punjab Kings", "Rajasthan Royals",
]

# ════════════════════════════════════════════════════════════════════
# NAVBAR
# ════════════════════════════════════════════════════════════════════

def make_navbar():
    return dbc.Navbar(
        dbc.Container([
            dbc.NavbarBrand(
                [html.Span("IPL", style={"color": "#F9CD05", "fontWeight": 700}),
                 html.Span(" Analytics Platform", style={"color": "white"})],
                style={"fontSize": "18px"},
            ),
            dbc.Nav([
                dbc.NavItem(dbc.NavLink("Team Stats",       href="/",           active="exact")),
                dbc.NavItem(dbc.NavLink("Player Explorer",  href="/players",    active="exact")),
                dbc.NavItem(dbc.NavLink("Win Predictor",    href="/predict",    active="exact")),
                dbc.NavItem(dbc.NavLink("Auction Simulator",href="/auction",    active="exact")),
            ], navbar=True, className="ms-auto"),
        ], fluid=True),
        color="#1a1a2e",
        dark=True,
        sticky="top",
        style={"marginBottom": "0px"},
    )


# ════════════════════════════════════════════════════════════════════
# PAGE 1 — TEAM STATS
# ════════════════════════════════════════════════════════════════════

def layout_team_stats():
    return html.Div([
        html.Div([
            html.H1("Team Statistics", className="page-title"),
            html.P("Season-by-season performance, head-to-head records, and venue analysis.",
                   className="page-subtitle"),
        ], className="page-header"),

        dbc.Container([
            # ── Filters ───────────────────────────────────────────
            dbc.Row([
                dbc.Col([
                    html.Label("Select teams", style={"fontWeight": 500, "fontSize": "13px"}),
                    dcc.Dropdown(
                        id="team-filter",
                        options=[{"label": t, "value": t} for t in ALL_TEAMS],
                        value=TOP_TEAMS[:6],
                        multi=True,
                        placeholder="Select teams...",
                    ),
                ], md=8),
                dbc.Col([
                    html.Label("Season range", style={"fontWeight": 500, "fontSize": "13px"}),
                    dcc.RangeSlider(
                        id="season-range",
                        min=min(ALL_SEASONS), max=max(ALL_SEASONS),
                        value=[min(ALL_SEASONS), max(ALL_SEASONS)],
                        marks={s: str(s) for s in ALL_SEASONS[::2]},
                        step=1,
                        tooltip={"placement": "bottom", "always_visible": False},
                    ),
                ], md=4),
            ], className="mb-4"),

            # ── Metric cards ──────────────────────────────────────
            dbc.Row(id="team-metric-cards", className="mb-4 g-3"),

            # ── Charts ────────────────────────────────────────────
            dbc.Row([
                dbc.Col([
                    html.P("Wins per season", className="section-title"),
                    dcc.Graph(id="wins-per-season-chart", style={"height": "380px"}),
                ], md=7),
                dbc.Col([
                    html.P("Win percentage by team", className="section-title"),
                    dcc.Graph(id="win-pct-chart", style={"height": "380px"}),
                ], md=5),
            ], className="mb-4"),

            dbc.Row([
                dbc.Col([
                    html.P("Toss decision trends", className="section-title"),
                    dcc.Graph(id="toss-chart", style={"height": "340px"}),
                ], md=6),
                dbc.Col([
                    html.P("Runs per over (all seasons)", className="section-title"),
                    dcc.Graph(id="runs-per-over-chart", style={"height": "340px"}),
                ], md=6),
            ], className="mb-4"),

        ], fluid=True),
    ])




def layout_player_explorer():
    return html.Div([
        html.Div([
            html.H1("Player Explorer", className="page-title"),
            html.P("Search and compare individual player career statistics.",
                   className="page-subtitle"),
        ], className="page-header"),

        dbc.Container([
            dbc.Row([
                dbc.Col([
                    html.Label("Search batsman", style={"fontWeight": 500, "fontSize": "13px"}),
                    dcc.Dropdown(
                        id="batsman-search",
                        options=[{"label": p, "value": p}
                                 for p in sorted(batting["batsman"].unique())],
                        value=batting.nlargest(1, "total_runs")["batsman"].iloc[0],
                        placeholder="Select a batsman...",
                        clearable=False,
                    ),
                ], md=4),
                dbc.Col([
                    html.Label("Compare with (optional)", style={"fontWeight": 500, "fontSize": "13px"}),
                    dcc.Dropdown(
                        id="batsman-compare",
                        options=[{"label": p, "value": p}
                                 for p in sorted(batting["batsman"].unique())],
                        placeholder="Add a second player...",
                    ),
                ], md=4),
                dbc.Col([
                    html.Label("Search bowler", style={"fontWeight": 500, "fontSize": "13px"}),
                    dcc.Dropdown(
                        id="bowler-search",
                        options=[{"label": p, "value": p}
                                 for p in sorted(bowling["bowler"].unique())],
                        value=bowling.nlargest(1, "wickets")["bowler"].iloc[0],
                        placeholder="Select a bowler...",
                        clearable=False,
                    ),
                ], md=4),
            ], className="mb-4"),

            # ── Batting stats ──────────────────────────────────────
            html.P("Batting profile", className="section-title"),
            dbc.Row(id="batting-metric-cards", className="mb-3 g-3"),
            dbc.Row([
                dbc.Col([
                    dcc.Graph(id="batting-scatter-chart", style={"height": "400px"}),
                ], md=7),
                dbc.Col([
                    dcc.Graph(id="phase-sr-chart", style={"height": "400px"}),
                ], md=5),
            ], className="mb-4"),

            # ── Bowling stats ──────────────────────────────────────
            html.P("Bowling profile", className="section-title"),
            dbc.Row(id="bowling-metric-cards", className="mb-3 g-3"),
            dcc.Graph(id="bowling-phase-chart", style={"height": "360px"}),

            # ── Player table ───────────────────────────────────────
            html.P("Full leaderboard", className="section-title"),
            dbc.Row([
                dbc.Col([
                    dcc.RadioItems(
                        id="leaderboard-type",
                        options=[
                            {"label": " Batting", "value": "batting"},
                            {"label": " Bowling", "value": "bowling"},
                        ],
                        value="batting",
                        inline=True,
                        style={"fontSize": "13px", "marginBottom": "10px"},
                    ),
                ]),
            ]),
            dash_table.DataTable(
                id="player-table",
                page_size=15,
                sort_action="native",
                filter_action="native",
                style_table={"overflowX": "auto"},
                style_header={
                    "backgroundColor": "#1a1a2e",
                    "color": "white",
                    "fontWeight": 600,
                    "fontSize": "12px",
                    "border": "none",
                },
                style_cell={
                    "fontSize": "12px",
                    "padding": "8px 12px",
                    "border": "0.5px solid #f0f0f0",
                    "fontFamily": "Segoe UI, Arial",
                },
                style_data_conditional=[
                    {"if": {"row_index": "odd"},
                     "backgroundColor": "#fafafa"},
                ],
            ),
        ], fluid=True),
    ])


# ════════════════════════════════════════════════════════════════════
# PAGE 3 — WIN PREDICTOR
# ════════════════════════════════════════════════════════════════════

def layout_win_predictor():
    return html.Div([
        html.Div([
            html.H1("Win Probability Predictor", className="page-title"),
            html.P("Pre-match win probability powered by XGBoost — trained on 15 seasons of IPL data.",
                   className="page-subtitle"),
        ], className="page-header"),

        dbc.Container([
            dbc.Row([
                # ── Input form ────────────────────────────────────
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H5("Match setup", style={"fontWeight": 600, "marginBottom": "20px"}),

                            html.Label("Team 1 (batting side)", style={"fontSize": "13px", "fontWeight": 500}),
                            dcc.Dropdown(
                                id="pred-team1",
                                options=[{"label": t, "value": t} for t in sorted(ALL_TEAMS)],
                                value="Mumbai Indians",
                                clearable=False,
                                style={"marginBottom": "14px"},
                            ),

                            html.Label("Team 2", style={"fontSize": "13px", "fontWeight": 500}),
                            dcc.Dropdown(
                                id="pred-team2",
                                options=[{"label": t, "value": t} for t in sorted(ALL_TEAMS)],
                                value="Chennai Super Kings",
                                clearable=False,
                                style={"marginBottom": "14px"},
                            ),

                            html.Label("Venue", style={"fontSize": "13px", "fontWeight": 500}),
                            dcc.Dropdown(
                                id="pred-venue",
                                options=[{"label": v, "value": v} for v in ALL_VENUES],
                                value=ALL_VENUES[0] if ALL_VENUES else None,
                                clearable=False,
                                style={"marginBottom": "14px"},
                            ),

                            html.Label("Toss winner", style={"fontSize": "13px", "fontWeight": 500}),
                            dcc.Dropdown(
                                id="pred-toss-winner",
                                options=[{"label": t, "value": t} for t in sorted(ALL_TEAMS)],
                                value="Mumbai Indians",
                                clearable=False,
                                style={"marginBottom": "14px"},
                            ),

                            html.Label("Toss decision", style={"fontSize": "13px", "fontWeight": 500}),
                            dcc.RadioItems(
                                id="pred-toss-decision",
                                options=[
                                    {"label": "  Bat first", "value": "bat"},
                                    {"label": "  Field first", "value": "field"},
                                ],
                                value="field",
                                inline=True,
                                style={"fontSize": "13px", "marginBottom": "20px"},
                            ),

                            dbc.Button(
                                "Predict outcome",
                                id="predict-btn",
                                color="primary",
                                style={"width": "100%",
                                       "background": "#7F77DD",
                                       "border": "none",
                                       "fontWeight": 600},
                            ),
                        ])
                    ], style={"border": "0.5px solid #e0e0e0"}),
                ], md=4),

                # ── Result panel ──────────────────────────────────
                dbc.Col([
                    html.Div(id="prediction-output"),
                    dcc.Graph(id="win-prob-gauge", style={"height": "320px"}),
                    html.Div(id="historical-h2h"),
                ], md=8),
            ]),
        ], fluid=True),
    ])


# ════════════════════════════════════════════════════════════════════
# PAGE 4 — AUCTION SIMULATOR
# ════════════════════════════════════════════════════════════════════

def layout_auction_simulator():
    def stat_slider(label, id_, min_, max_, value, step=1):
        return html.Div([
            html.Div([
                html.Span(label,  style={"fontSize": "13px", "fontWeight": 500}),
                html.Span(id=f"{id_}-out", style={"fontSize": "13px",
                                                    "color": "#7F77DD",
                                                    "fontWeight": 600,
                                                    "float": "right"}),
            ], style={"display": "flex", "justifyContent": "space-between"}),
            dcc.Slider(
                id=id_, min=min_, max=max_, value=value, step=step,
                marks=None,
                tooltip={"placement": "bottom", "always_visible": False},
            ),
        ], style={"marginBottom": "18px"})

    return html.Div([
        html.Div([
            html.H1("Auction Value Simulator", className="page-title"),
            html.P("Adjust player stats with the sliders to see predicted IPL auction price in real-time.",
                   className="page-subtitle"),
        ], className="page-header"),

        dbc.Container([
            dbc.Row([
                # ── Sliders ───────────────────────────────────────
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H5("Batting stats",
                                    style={"fontWeight": 600, "marginBottom": "16px"}),
                            stat_slider("Career runs",    "sl-runs",    0, 8000, 2000, 50),
                            stat_slider("Batting average","sl-avg",     0, 80,   28,   0.5),
                            stat_slider("Strike rate",    "sl-sr",      60, 220, 130,  0.5),
                            stat_slider("Centuries",      "sl-100s",    0, 10,   1,    1),
                            stat_slider("Half-centuries", "sl-50s",     0, 60,   10,   1),
                            stat_slider("Boundary rate (%)", "sl-br",   0, 30,   12,   0.5),
                            stat_slider("Powerplay SR",   "sl-pp-sr",   0, 220, 130,  0.5),
                            stat_slider("Death SR",       "sl-death-sr",0, 300, 160,  0.5),
                        ])
                    ], style={"border": "0.5px solid #e0e0e0", "marginBottom": "16px"}),

                    dbc.Card([
                        dbc.CardBody([
                            html.H5("Bowling stats",
                                    style={"fontWeight": 600, "marginBottom": "16px"}),
                            stat_slider("Career wickets",    "sl-wkts",     0, 200, 30,   1),
                            stat_slider("Economy rate",      "sl-econ",     5, 14,  8.5,  0.1),
                            stat_slider("Bowling average",   "sl-bowl-avg", 10, 60, 35,   0.5),
                            stat_slider("Dot ball %",        "sl-dot",      0, 60,  20,   0.5),
                            stat_slider("Death economy",     "sl-d-econ",   5, 16,  9.5,  0.1),
                            stat_slider("Powerplay economy", "sl-pp-econ",  5, 14,  8.0,  0.1),
                            stat_slider("Matches bowled",    "sl-mbowled",  0, 150, 40,   1),
                        ])
                    ], style={"border": "0.5px solid #e0e0e0"}),
                ], md=5),

                # ── Live result ───────────────────────────────────
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H5("Predicted auction value",
                                    style={"fontWeight": 600,
                                           "textAlign": "center",
                                           "marginBottom": "20px"}),
                            html.Div(id="auction-price-display",
                                     style={"textAlign": "center"}),
                            dcc.Graph(id="auction-gauge",
                                      style={"height": "260px"},
                                      config={"displayModeBar": False}),
                            html.Hr(),
                            html.P("Similar players at this price",
                                   className="section-title"),
                            html.Div(id="similar-players-table"),
                        ])
                    ], style={"border": "0.5px solid #e0e0e0", "position": "sticky", "top": "70px"}),
                ], md=7),
            ]),

            html.Br(),
            html.P("All-time player valuations leaderboard", className="section-title"),
            dcc.Graph(id="valuation-chart", style={"height": "420px"}),

        ], fluid=True),
    ])