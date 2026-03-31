"""Energy distribution panel: overlaid histograms of wafer energy per PDG category."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

from hgcal_viewer.data.geometry import UnionGeom
from hgcal_viewer.data.loader import EventPayload
from hgcal_viewer.data import pdg as PDG


def build_figure(
    union: UnionGeom,
    data: EventPayload,
    active_cats: list[int],
    endcap: str,
    log_scale: bool,
    min_energy: float,
) -> go.Figure:
    energy  = data["union_energy"]
    present = data["union_present"]
    cat     = data["union_cat"]
    zside   = union["zside"]
    active_cat_set = set(active_cats) if active_cats is not None else set()
    active_cats = active_cats or []

    # Endcap mask
    if endcap == "pos":
        ec = zside > 0
    elif endcap == "neg":
        ec = zside < 0
    else:
        ec = np.ones(zside.shape[0], dtype=bool)

    base = present & ec & (energy >= float(min_energy))

    fig = go.Figure()

    # One histogram trace per active PDG category (excludes UNASSOC_IDX, handled below)
    for ci in active_cats:
        if ci == PDG.UNASSOC_IDX:
            continue
        cmask = base & (cat == ci)
        vals = energy[cmask]
        if vals.size == 0:
            continue
        plot_vals = np.log1p(vals) if log_scale else vals
        fig.add_trace(go.Histogram(
            x=plot_vals,
            name=PDG.LABELS[ci],
            marker_color=PDG.COLOURS[ci],
            opacity=0.65,
            nbinsx=60,
            hovertemplate=f"{PDG.LABELS[ci]}<br>bin=%{{x:.3f}}<br>count=%{{y}}<extra></extra>",
        ))

    # "Active" = all present wafers in this event (no category filter)
    if PDG.UNASSOC_IDX in active_cat_set:
        vals = energy[base]
        if vals.size:
            plot_vals = np.log1p(vals) if log_scale else vals
            fig.add_trace(go.Histogram(
                x=plot_vals,
                name=PDG.LABELS[PDG.UNASSOC_IDX],
                marker_color=PDG.COLOURS[PDG.UNASSOC_IDX],
                opacity=0.5,
                nbinsx=60,
                hovertemplate=f"{PDG.LABELS[PDG.UNASSOC_IDX]}<br>bin=%{{x:.3f}}<br>count=%{{y}}<extra></extra>",
            ))

    xlabel = "log(1 + energy)" if log_scale else "Wafer energy"
    fig.update_layout(
        barmode="overlay",
        title={"text": "Wafer energy distribution", "font": {"size": 13}},
        xaxis_title=xlabel,
        yaxis_title="Wafer count",
        legend={"orientation": "h", "y": -0.25, "font": {"size": 10}},
        margin={"l": 50, "r": 10, "t": 40, "b": 80},
        height=320,
    )
    return fig
