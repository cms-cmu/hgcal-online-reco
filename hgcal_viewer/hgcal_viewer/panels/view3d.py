"""3D scatter panel: wafer geometry + RecHits + cluster impact points."""

from __future__ import annotations

import math

import numpy as np
import plotly.graph_objects as go

from hgcal_viewer.data.geometry import UnionGeom, hex_vertices
from hgcal_viewer.data.loader import EventPayload
from hgcal_viewer.data import pdg as PDG


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _endcap_mask(zside: np.ndarray, endcap: str) -> np.ndarray:
    if endcap == "pos":
        return zside > 0
    if endcap == "neg":
        return zside < 0
    return np.ones(zside.shape[0], dtype=bool)


def _wafer_selection(
    union: UnionGeom,
    data: EventPayload,
    mode: str,
    min_energy: float,
    active_cats: set[int],
    endcap: str,
) -> np.ndarray:
    """Return boolean mask over union wafers for fill/outline modes."""
    energy  = data["union_energy"]
    present = data["union_present"]
    cat     = data["union_cat"]
    ec_mask = _endcap_mask(union["zside"], endcap)

    if mode == "none":
        return np.zeros(present.shape[0], dtype=bool)
    if mode == "all":
        base = ec_mask
    elif mode == "active":
        base = present & (energy >= float(min_energy)) & ec_mask
    elif mode == "associated":
        assoc = (cat >= 0) & (cat != PDG.UNASSOC_IDX)
        base  = assoc & (energy >= float(min_energy)) & ec_mask
    else:
        return np.zeros(present.shape[0], dtype=bool)

    # Further restrict to selected particle categories
    cat_ok = np.zeros(present.shape[0], dtype=bool)
    for ci in active_cats:
        cat_ok |= (cat == ci)
    # Always include inactive wafers (cat == -1) when showing all/active
    if mode in ("all", "active"):
        cat_ok |= (cat == -1)

    return base & cat_ok


# ---------------------------------------------------------------------------
# Public figure builder
# ---------------------------------------------------------------------------

def build_figure(
    union: UnionGeom,
    data: EventPayload,
    event_index: int,
    global_radius: float,
    max_energy_log: float,
    wafer_mode: str,
    min_energy: float,
    rechit_mode: str,
    max_rechits: int,
    show_clusters: bool,
    active_cats: list[int],
    endcap: str,
) -> go.Figure:
    fig = go.Figure()
    active_cat_set = set(active_cats) if active_cats is not None else set()

    energy  = data["union_energy"]
    cat     = data["union_cat"]

    # ── Wafer outlines ───────────────────────────────────────────────────────
    outline_mask = _wafer_selection(
        union, data, wafer_mode, min_energy, active_cat_set, endcap
    )
    if outline_mask.any():
        ox, oy, oz = [], [], []
        for x, y, z in zip(
            union["x"][outline_mask],
            union["y"][outline_mask],
            union["z"][outline_mask],
        ):
            verts = hex_vertices(float(x), float(y), float(z), global_radius)
            ox.extend([v[0] for v in verts] + [verts[0][0], None])
            oy.extend([v[1] for v in verts] + [verts[0][1], None])
            oz.extend([v[2] for v in verts] + [verts[0][2], None])
        fig.add_trace(go.Scatter3d(
            x=ox, y=oy, z=oz,
            mode="lines",
            name="Wafer outlines",
            line={"color": "#555", "width": 1},
            hoverinfo="skip",
        ))

    # ── Wafer fill (coloured by PDG category) ────────────────────────────────
    fill_mask = _wafer_selection(
        union, data, wafer_mode, min_energy, active_cat_set, endcap
    )
    if fill_mask.any():
        # Split by category so each gets its own colour + legend entry
        fill_idx = np.where(fill_mask)[0]
        cats_present = np.unique(cat[fill_idx])
        for ci in cats_present:
            cmask = fill_mask & (cat == ci)
            if not cmask.any():
                continue
            loge = np.log1p(energy[cmask])
            colour = PDG.COLOURS[ci] if ci >= 0 else "#cccccc"
            label  = PDG.LABELS[ci]  if ci >= 0 else "inactive"
            fig.add_trace(go.Scatter3d(
                x=union["x"][cmask],
                y=union["y"][cmask],
                z=union["z"][cmask],
                mode="markers",
                name=f"Wafer – {label}",
                marker={
                    "size": 2,
                    "color": loge,
                    "colorscale": [[0, "#ffffff"], [1, colour]],
                    "cmin": 0.0,
                    "cmax": max_energy_log,
                    "opacity": 0.75,
                },
                hovertemplate=(
                    f"<b>{label}</b><br>"
                    "x=%{x:.1f}  y=%{y:.1f}  z=%{z:.1f}<br>"
                    "energy=%{customdata:.4f}<extra></extra>"
                ),
                customdata=energy[cmask],
            ))

    # ── RecHits ──────────────────────────────────────────────────────────────
    if rechit_mode != "none":
        rh_cl = data["rh_cluster_idx"]
        ec_rh = data["rh_z"] >= 0 if endcap == "pos" else (
                data["rh_z"] < 0  if endcap == "neg" else
                np.ones(rh_cl.shape[0], dtype=bool))

        if rechit_mode == "matched":
            rh_mask = (rh_cl >= 0) & ec_rh
        else:
            rh_mask = ec_rh

        # Filter by active PDG categories via cluster pdg
        cl_pdg = data["cl_pdg"]
        cat_rh = np.full(rh_cl.shape[0], PDG.UNASSOC_IDX, dtype=np.int32)
        valid  = (rh_cl >= 0) & (rh_cl < cl_pdg.size)
        cat_rh[valid] = np.array([PDG.pdg_to_idx(int(p)) for p in cl_pdg[rh_cl[valid]]])
        rh_mask &= np.isin(cat_rh, list(active_cat_set))

        idx = np.where(rh_mask)[0]
        if idx.size > int(max_rechits):
            rng = np.random.default_rng(42)
            idx = rng.choice(idx, size=int(max_rechits), replace=False)
        if idx.size:
            fig.add_trace(go.Scatter3d(
                x=data["rh_x"][idx],
                y=data["rh_y"][idx],
                z=data["rh_z"][idx],
                mode="markers",
                name="RecHits",
                marker={"size": 1.5, "color": "#1f77b4", "opacity": 0.5},
                hovertemplate=(
                    "RecHit<br>x=%{x:.1f}  y=%{y:.1f}  z=%{z:.1f}<br>"
                    "energy=%{customdata:.4f}<extra></extra>"
                ),
                customdata=data["rh_energy"][idx],
            ))

    # ── Cluster impact points ────────────────────────────────────────────────
    if show_clusters and data["cl_x"].size:
        cl_pdg = data["cl_pdg"]
        ec_cl  = (data["cl_z"] >= 0 if endcap == "pos" else
                  data["cl_z"] < 0  if endcap == "neg" else
                  np.ones(cl_pdg.shape[0], dtype=bool))
        cat_cl = np.array([PDG.pdg_to_idx(int(p)) for p in cl_pdg])
        cl_mask = ec_cl & np.isin(cat_cl, list(active_cat_set))
        if cl_mask.any():
            colours_cl = [PDG.COLOURS[c] for c in cat_cl[cl_mask]]
            fig.add_trace(go.Scatter3d(
                x=data["cl_x"][cl_mask],
                y=data["cl_y"][cl_mask],
                z=data["cl_z"][cl_mask],
                mode="markers",
                name="Cluster impact pts",
                marker={
                    "size": 5,
                    "color": colours_cl,
                    "symbol": "diamond",
                    "line": {"width": 0.5, "color": "#333"},
                },
                hovertemplate=(
                    "<b>%{customdata[0]}</b><br>"
                    "pT=%{customdata[1]:.2f} GeV  η=%{customdata[2]:.2f}  φ=%{customdata[3]:.2f}<br>"
                    "E=%{customdata[4]:.3f} GeV<extra></extra>"
                ),
                customdata=np.column_stack([
                    [PDG.pdg_to_label(int(p)) for p in cl_pdg[cl_mask]],
                    data["cl_pt"][cl_mask],
                    data["cl_eta"][cl_mask],
                    data["cl_phi"][cl_mask],
                    data["cl_energy"][cl_mask],
                ]),
            ))

    fig.update_layout(
        scene={
            "xaxis_title": "x (cm)",
            "yaxis_title": "y (cm)",
            "zaxis_title": "z (cm)",
            "aspectmode": "data",
        },
        margin={"l": 0, "r": 0, "t": 30, "b": 0},
        uirevision="keep-camera",
        legend={"orientation": "h", "y": -0.02, "font": {"size": 11}},
        title={"text": f"Event {event_index}", "font": {"size": 14}},
    )
    return fig
