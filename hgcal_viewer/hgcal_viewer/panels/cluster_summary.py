"""Cluster summary panel: η–φ scatter of MergedSimCluster impact points,
coloured by PDG category, sized by pT."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

from hgcal_viewer.data.loader import EventPayload
from hgcal_viewer.data import pdg as PDG


def build_figure(
    data: EventPayload,
    active_cats: list[int],
    endcap: str,
) -> go.Figure:
    cl_pdg    = data["cl_pdg"]
    cl_eta    = data["cl_eta"]
    cl_phi    = data["cl_phi"]
    cl_pt     = data["cl_pt"]
    cl_energy = data["cl_energy"]
    cl_z      = data["cl_z"]
    active_cat_set = set(active_cats) if active_cats is not None else set()
    active_cats = active_cats or []

    # Endcap mask
    if endcap == "pos":
        ec = cl_z >= 0
    elif endcap == "neg":
        ec = cl_z < 0
    else:
        ec = np.ones(cl_pdg.shape[0], dtype=bool)

    # Only trainable clusters (sumHitEnergy > 0)
    trainable = (cl_energy > 0) & ec

    cat_arr = np.array([PDG.pdg_to_idx(int(p)) for p in cl_pdg])

    fig = go.Figure()

    for ci in active_cats:
        cmask = trainable & (cat_arr == ci)
        if not cmask.any():
            continue
        # Marker size proportional to sqrt(pT), clamped to [3, 18]
        sizes = np.clip(3.0 + 4.0 * np.sqrt(np.maximum(cl_pt[cmask], 0.0)), 3, 18)
        fig.add_trace(go.Scatter(
            x=cl_eta[cmask],
            y=cl_phi[cmask],
            mode="markers",
            name=PDG.LABELS[ci],
            marker={
                "color": PDG.COLOURS[ci],
                "size": sizes,
                "opacity": 0.75,
                "line": {"width": 0.4, "color": "#333"},
            },
            hovertemplate=(
                f"<b>{PDG.LABELS[ci]}</b><br>"
                "η=%{x:.3f}  φ=%{y:.3f}<br>"
                "pT=%{customdata[0]:.3f} GeV<br>"
                "E=%{customdata[1]:.3f} GeV<extra></extra>"
            ),
            customdata=np.column_stack([cl_pt[cmask], cl_energy[cmask]]),
        ))

    fig.update_layout(
        title={"text": "Clusters: η–φ (size ∝ √pT)", "font": {"size": 13}},
        xaxis_title="η",
        yaxis_title="φ (rad)",
        legend={"orientation": "h", "y": -0.25, "font": {"size": 10}},
        margin={"l": 50, "r": 10, "t": 40, "b": 80},
        height=320,
    )
    return fig
