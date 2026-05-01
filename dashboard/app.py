# dashboard/app.py
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
DASHBOARD = Path(__file__).parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(DASHBOARD))

import dash
from dash import Input, Output, html, dcc
import dash_bootstrap_components as dbc

from layout    import (make_navbar,
                       layout_team_stats,
                       layout_player_explorer,
                       layout_win_predictor,
                       layout_auction_simulator)
from callbacks import register_callbacks

# ── Initialise app ────────────────────────────────────────────────────
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    suppress_callback_exceptions=True,
    meta_tags=[{"name": "viewport",
                "content": "width=device-width, initial-scale=1"}],
)
app.title  = "IPL Analytics Platform"
server     = app.server

# ── Root layout ───────────────────────────────────────────────────────
# dcc.Location MUST be first child — it triggers the routing callback
app.layout = html.Div([
    dcc.Location(id="url", refresh=False),
    make_navbar(),
    html.Div(id="page-content"),
])

# ── Page routing callback ─────────────────────────────────────────────
@app.callback(
    Output("page-content", "children"),
    Input("url", "pathname"),
)
def render_page(pathname):
    print(f"render_page called with pathname: {pathname}")
    if pathname == "/players":
        return layout_player_explorer()
    if pathname == "/predict":
        return layout_win_predictor()
    if pathname == "/auction":
        return layout_auction_simulator()
    return layout_team_stats()

register_callbacks(app)

if __name__ == "__main__":
    app.run(debug=True, port=8050)