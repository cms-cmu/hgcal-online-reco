"""Sidebar layout: all user-facing controls.

Returns a ``dash.html.Div`` that can be dropped into any layout.
All control IDs are defined as module-level constants so panels can
import them without circular dependencies.
"""

from __future__ import annotations

from dash import dcc, html

from hgcal_viewer.data import pdg as PDG

# ---------------------------------------------------------------------------
# Stable control IDs  (imported by panels and app.py)
# ---------------------------------------------------------------------------

ID_EVENT          = "ctrl-event"
ID_WAFER_MODE     = "ctrl-wafer-mode"
ID_MIN_ENERGY     = "ctrl-min-energy"
ID_RECHIT_MODE    = "ctrl-rechit-mode"
ID_MAX_RECHITS    = "ctrl-max-rechits"
ID_SHOW_CLUSTERS  = "ctrl-show-clusters"
ID_PDG_FILTER     = "ctrl-pdg-filter"
ID_ENDCAP_FILTER  = "ctrl-endcap-filter"
ID_ENERGY_LOG     = "ctrl-energy-log"


# ---------------------------------------------------------------------------
# Layout builder
# ---------------------------------------------------------------------------

def sidebar(events: list[int], max_energy: float) -> html.Div:
    """Build the full sidebar Div.  *events* is the list of available event
    indices; *max_energy* sets the upper bound of the energy slider."""

    label_style = {"fontWeight": "600", "marginTop": "12px", "display": "block"}

    return html.Div(
        [
            html.H3("HGCAL Viewer", style={"marginBottom": "4px"}),
            html.Hr(style={"margin": "6px 0"}),

            # ── Event selector ──────────────────────────────────────────────
            html.Label("Event", style=label_style),
            dcc.Dropdown(
                id=ID_EVENT,
                options=[{"label": str(e), "value": e} for e in events],
                value=events[0],
                clearable=False,
            ),

            # ── Particle-type filter ─────────────────────────────────────────
            html.Label("Particle types", style=label_style),
            dcc.Checklist(
                id=ID_PDG_FILTER,
                options=PDG.category_options(),
                value=PDG.all_category_indices(),
                labelStyle={"display": "flex", "alignItems": "center", "gap": "6px"},
                inputStyle={"accentColor": "#457b9d"},
            ),

            # ── Endcap filter ────────────────────────────────────────────────
            html.Label("Endcap", style=label_style),
            dcc.RadioItems(
                id=ID_ENDCAP_FILTER,
                options=[
                    {"label": "Both",     "value": "both"},
                    {"label": "+z only",  "value": "pos"},
                    {"label": "−z only",  "value": "neg"},
                ],
                value="both",
                inline=True,
                inputStyle={"marginRight": "4px"},
                labelStyle={"marginRight": "12px"},
            ),

            html.Hr(style={"margin": "10px 0"}),

            # ── Wafer display ────────────────────────────────────────────────
            html.Label("Wafer display", style=label_style),
            dcc.Dropdown(
                id=ID_WAFER_MODE,
                options=[
                    {"label": "None",             "value": "none"},
                    {"label": "All (union)",       "value": "all"},
                    {"label": "Active only",       "value": "active"},
                    {"label": "Associated only",   "value": "associated"},
                ],
                value="active",
                clearable=False,
            ),

            html.Label("Min wafer energy", style=label_style),
            dcc.Slider(
                id=ID_MIN_ENERGY,
                min=0.0,
                max=max_energy,
                step=max_energy / 100.0,
                value=0.0,
                tooltip={"placement": "bottom", "always_visible": False},
                marks=None,
            ),

            html.Hr(style={"margin": "10px 0"}),

            # ── RecHits ──────────────────────────────────────────────────────
            html.Label("RecHits", style=label_style),
            dcc.Dropdown(
                id=ID_RECHIT_MODE,
                options=[
                    {"label": "None",          "value": "none"},
                    {"label": "Matched only",  "value": "matched"},
                    {"label": "All",           "value": "all"},
                ],
                value="matched",
                clearable=False,
            ),

            html.Label("Max RecHits shown", style=label_style),
            dcc.Input(
                id=ID_MAX_RECHITS,
                type="number",
                value=15000,
                min=1000,
                step=1000,
                style={"width": "100%"},
            ),

            html.Hr(style={"margin": "10px 0"}),

            # ── Cluster impact points ────────────────────────────────────────
            dcc.Checklist(
                id=ID_SHOW_CLUSTERS,
                options=[{"label": "Show cluster impact points", "value": "show"}],
                value=["show"],
            ),

            html.Hr(style={"margin": "10px 0"}),

            # ── Energy axis scale ────────────────────────────────────────────
            html.Label("Energy axis", style=label_style),
            dcc.RadioItems(
                id=ID_ENERGY_LOG,
                options=[
                    {"label": "Linear", "value": "linear"},
                    {"label": "Log",    "value": "log"},
                ],
                value="log",
                inline=True,
                inputStyle={"marginRight": "4px"},
                labelStyle={"marginRight": "12px"},
            ),
        ],
        style={
            "width": "280px",
            "minWidth": "280px",
            "padding": "12px 14px",
            "overflowY": "auto",
            "height": "100vh",
            "boxSizing": "border-box",
            "borderRight": "1px solid #dee2e6",
            "fontSize": "13px",
        },
    )
