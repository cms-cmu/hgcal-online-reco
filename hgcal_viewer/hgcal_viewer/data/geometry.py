"""Union wafer geometry: loading, radius estimation, hex vertex helpers."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Dict

import numpy as np
from scipy.spatial import cKDTree


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

UnionGeom = Dict[str, np.ndarray]
"""Keys: layer (int32), zside (int8), x (float32), y (float32), z (float32)."""


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_union_geometry(csv_path: str | Path) -> UnionGeom:
    """Load wafer_mapping.csv into typed numpy arrays."""
    rows = []
    with open(csv_path, "r", newline="") as fh:
        for row in csv.DictReader(fh):
            rows.append(row)
    if not rows:
        raise ValueError(f"No rows in wafer mapping CSV: {csv_path}")
    return {
        "layer": np.array([int(r["layer"]) for r in rows], dtype=np.int32),
        "zside": np.array([int(r["zside"]) for r in rows], dtype=np.int8),
        "x": np.array([float(r["x"]) for r in rows], dtype=np.float32),
        "y": np.array([float(r["y"]) for r in rows], dtype=np.float32),
        "z": np.array([float(r["z"]) for r in rows], dtype=np.float32),
    }


def load_meta(meta_path: str | Path) -> dict:
    """Load wafer_mapping_meta.json."""
    with open(meta_path, "r", encoding="utf-8") as fh:
        return json.load(fh)


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def hex_vertices(
    x: float, y: float, z: float, r: float
) -> list[tuple[float, float, float]]:
    """Six (x,y,z) corners of a flat-top hexagon centred at (x,y,z)."""
    angles = np.deg2rad(np.arange(0, 360, 60))
    return [(x + r * math.cos(a), y + r * math.sin(a), z) for a in angles]


def estimate_global_radius(
    xy: np.ndarray, layers: np.ndarray, zside: np.ndarray
) -> float:
    """0.5 × median nearest-neighbour spacing per (layer, zside)."""
    if xy.shape[0] < 2:
        return 1.0
    nn_all: list[float] = []
    for layer in np.unique(layers):
        for side in (-1, 1):
            mask = (layers == layer) & (zside == side)
            if mask.sum() < 2:
                continue
            pts = xy[mask]
            tree = cKDTree(pts)
            dists, _ = tree.query(pts, k=2)
            nn = dists[:, 1]
            nn = nn[np.isfinite(nn)]
            if nn.size:
                nn_all.extend(nn.tolist())
    if not nn_all:
        return 1.0
    return 0.5 * float(np.median(np.asarray(nn_all)))


# ---------------------------------------------------------------------------
# Wafer-geometry export (one-time setup)
# ---------------------------------------------------------------------------

def build_union_geometry_from_root(
    root_path: str | Path,
    csv_out: str | Path,
    meta_out: str | Path,
) -> None:
    """
    Scan all events in *root_path*, collect unique (layer,u,v,zside) → (x,y,z)
    mappings, write wafer_mapping.csv and wafer_mapping_meta.json.

    This replicates export_wafer_mapping.py as a library call.
    """
    import uproot
    from collections import OrderedDict

    mapping: dict[tuple[int, int, int, int], tuple[float, float, float]] = {}

    branches = [
        "L1THGCAL_wafer_layer",
        "L1THGCAL_wafer_waferu",
        "L1THGCAL_wafer_waferv",
        "L1THGCAL_wafer_zside",
        "L1THGCAL_wafer_x",
        "L1THGCAL_wafer_y",
        "L1THGCAL_wafer_z",
    ]

    with uproot.open(root_path) as f:
        # Use awkward output for compatibility with both TTrees and RNTuples.
        for chunk in f["Events"].iterate(branches, step_size=200, library="ak"):
            for i_evt in range(len(chunk)):
                l_evt = chunk["L1THGCAL_wafer_layer"][i_evt]
                u_evt = chunk["L1THGCAL_wafer_waferu"][i_evt]
                v_evt = chunk["L1THGCAL_wafer_waferv"][i_evt]
                zs_evt = chunk["L1THGCAL_wafer_zside"][i_evt]
                x_evt = chunk["L1THGCAL_wafer_x"][i_evt]
                y_evt = chunk["L1THGCAL_wafer_y"][i_evt]
                z_evt = chunk["L1THGCAL_wafer_z"][i_evt]
                for l, u, v, zsi, x, y, z in zip(l_evt, u_evt, v_evt, zs_evt, x_evt, y_evt, z_evt):
                    key = (int(l), int(u), int(v), int(zsi))
                    if key not in mapping:
                        mapping[key] = (float(x), float(y), float(z))

    sorted_items = OrderedDict(
        sorted(mapping.items(), key=lambda kv: (kv[0][3], kv[0][0], kv[0][1], kv[0][2]))
    )

    with open(csv_out, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["layer", "u", "v", "zside", "x", "y", "z"])
        for (layer, u, v, zside), (x, y, z) in sorted_items.items():
            writer.writerow([layer, u, v, zside, f"{x:.6f}", f"{y:.6f}", f"{z:.6f}"])

    xy = np.array([[v[0], v[1]] for v in sorted_items.values()], dtype=np.float64)
    layers_arr = np.array([k[0] for k in sorted_items.keys()], dtype=np.int16)
    zside_arr = np.array([k[3] for k in sorted_items.keys()], dtype=np.int8)
    global_radius = estimate_global_radius(xy, layers_arr, zside_arr)

    meta = {
        "global_radius": global_radius,
        "radius_method": "0.5 * median nearest-neighbor spacing per (layer,zside)",
        "source_root": str(root_path),
        "unique_wafers": len(sorted_items),
    }
    with open(meta_out, "w") as fh:
        json.dump(meta, fh, indent=2, sort_keys=True)
        fh.write("\n")

    print(f"Wrote {csv_out}  ({len(sorted_items)} unique wafers)")
    print(f"Wrote {meta_out}  (global_radius={global_radius:.6f})")
