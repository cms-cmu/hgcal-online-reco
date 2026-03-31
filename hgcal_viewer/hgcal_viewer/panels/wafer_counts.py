"""Wafer counts panel: stacked bar of associated / active-unassociated / inactive
wafers per detector layer, filtered by PDG category and endcap."""

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
) -> go.Figure:
    cat     = data["union_cat"]
    present = data["union_present"]
    layers  = union["layer"]
    zside   = union["zside"]
    active_cat_set = set(active_cats) if active_cats is not None else set()

    # Endcap mask
    if endcap == "pos":
        ec = zside > 0
    elif endcap == "neg":
        ec = zside < 0
    else:
        ec = np.ones(zside.shape[0], dtype=bool)

    unique_layers = np.sort(np.unique(layers[ec]))

    # Per-layer counts: associated (by PDG cat), active-unassociated, inactive
    # "associated" = cat in active_cats and cat != OTHER_IDX
    # "unassociated" = present but cat == OTHER_IDX or cat not in active_cats
    # "inactive" = not present

    assoc_counts:   dict[int, np.ndarray] = {}   # cat_idx → per-layer count
    unassoc_counts: np.ndarray = np.zeros(unique_layers.shape[0], dtype=int)
    inactive_counts: np.ndarray = np.zeros(unique_layers.shape[0], dtype=int)

    for li, layer in enumerate(unique_layers):
        lmask = (layers == layer) & ec
        for wi in np.where(lmask)[0]:
            c = int(cat[wi])
            if not present[wi]:
                inactive_counts[li] += 1
            elif c in active_cat_set and c != PDG.UNASSOC_IDX:
                assoc_counts.setdefault(c, np.zeros(unique_layers.shape[0], dtype=int))
                assoc_counts[c][li] += 1
            else:
                unassoc_counts[li] += 1

    fig = go.Figure()

    # Inactive (grey, bottom)
    fig.add_trace(go.Bar(
        x=unique_layers,
        y=inactive_counts,
        name="Inactive",
        marker_color="#dee2e6",
        hovertemplate="Layer %{x}<br>Inactive: %{y}<extra></extra>",
    ))

    # Active-unassociated
    fig.add_trace(go.Bar(
        x=unique_layers,
        y=unassoc_counts,
        name="Active (unassociated)",
        marker_color="#adb5bd",
        hovertemplate="Layer %{x}<br>Active unassoc: %{y}<extra></extra>",
    ))

    # Associated, one trace per PDG category
    for ci, counts in sorted(assoc_counts.items()):
        fig.add_trace(go.Bar(
            x=unique_layers,
            y=counts,
            name=PDG.LABELS[ci],
            marker_color=PDG.COLOURS[ci],
            hovertemplate=f"Layer %{{x}}<br>{PDG.LABELS[ci]}: %{{y}}<extra></extra>",
        ))

    fig.update_layout(
        barmode="stack",
        title={"text": "Wafer counts per layer", "font": {"size": 13}},
        xaxis_title="Layer",
        yaxis_title="Wafer count",
        legend={"orientation": "h", "y": -0.25, "font": {"size": 10}},
        margin={"l": 50, "r": 10, "t": 40, "b": 80},
        height=320,
    )
    return fig
