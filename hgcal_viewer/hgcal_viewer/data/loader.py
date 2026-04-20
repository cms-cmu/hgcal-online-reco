"""Per-event data loading and association logic.

All ROOT I/O lives here.  The rest of the package works with plain numpy
dicts returned by ``load_event``.

Event payload keys
------------------
rh_x, rh_y, rh_z        : RecHit positions (float32)
rh_energy                : RecHit energy    (float32)
rh_cluster_idx           : MergedSimCluster index per RecHit, -1 = unmatched (int32)

cl_x, cl_y, cl_z         : MergedSimCluster impact-point positions (float32)
cl_pdg                   : MergedSimCluster pdgId (int32)
cl_pt, cl_eta, cl_phi    : kinematics (float32)
cl_energy                : sumHitEnergy (float32)

union_energy             : per-union-wafer energy mapped from event wafers (float32)
union_present            : bool mask — wafer was active this event
union_cat                : per-union-wafer PDG category index (-1 = inactive/unassociated)
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import uproot
from scipy.spatial import cKDTree

from hgcal_viewer.data.geometry import UnionGeom
from hgcal_viewer.data import pdg as PDG


# ---------------------------------------------------------------------------
# Branch list
# ---------------------------------------------------------------------------

_BRANCHES = [
    "RecHitHGC_x", "RecHitHGC_y", "RecHitHGC_z", "RecHitHGC_energy",
    "RecHitHGC_MergedSimClusterBestMatchIdx",
    "MergedSimCluster_impactPoint_x",
    "MergedSimCluster_impactPoint_y",
    "MergedSimCluster_impactPoint_z",
    "MergedSimCluster_pdgId",
    "MergedSimCluster_pt",
    "MergedSimCluster_eta",
    "MergedSimCluster_phi",
    "MergedSimCluster_sumHitEnergy",
    "L1THGCAL_wafer_x", "L1THGCAL_wafer_y", "L1THGCAL_wafer_z",
    "L1THGCAL_wafer_layer", "L1THGCAL_wafer_energy",
]

EventPayload = Dict[str, np.ndarray]


# ---------------------------------------------------------------------------
# Geometry key helpers  (mirrors export_wafer_mapping / interactive_event_viewer)
# ---------------------------------------------------------------------------

def _build_key(
    layer: int, zside: int, x: float, y: float, z: float, digits: int = 4
) -> tuple[int, int, float, float, float]:
    return (layer, zside, round(x, digits), round(y, digits), round(z, digits))


def _map_event_energy(
    union: UnionGeom,
    wf_layer: np.ndarray,
    wf_x: np.ndarray,
    wf_y: np.ndarray,
    wf_z: np.ndarray,
    wf_energy: np.ndarray,
    digits: int = 4,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (energy_per_union_wafer, present_mask)."""
    key_to_idx: dict[tuple, int] = {}
    for idx, (layer, zside, x, y, z) in enumerate(
        zip(union["layer"], union["zside"], union["x"], union["y"], union["z"])
    ):
        key_to_idx[_build_key(int(layer), int(zside), float(x), float(y), float(z), digits)] = idx

    energy = np.zeros(union["x"].shape[0], dtype=np.float32)
    present = np.zeros(union["x"].shape[0], dtype=bool)
    for layer, x, y, z, e in zip(wf_layer, wf_x, wf_y, wf_z, wf_energy):
        zside = 1 if float(z) >= 0 else -1
        key = _build_key(int(layer), int(zside), float(x), float(y), float(z), digits)
        idx = key_to_idx.get(key)
        if idx is not None:
            energy[idx] = float(e)
            present[idx] = True
    return energy, present


def _assign_wafer_categories(
    union: UnionGeom,
    present: np.ndarray,
    rh_x: np.ndarray,
    rh_y: np.ndarray,
    rh_z: np.ndarray,
    rh_cluster_idx: np.ndarray,
    cl_pdg: np.ndarray,
    global_radius: float,
) -> np.ndarray:
    """
    Return int array of length N_union_wafers.

    Value = PDG category index of the nearest matched RecHit cluster.
    Value = PDG.UNASSOC_IDX if active but no matched RecHit within radius.
    Value = -1             if wafer is not present/active.
    """
    cat = np.full(union["x"].shape[0], -1, dtype=np.int32)

    active_idx = np.where(present)[0]
    if active_idx.size == 0:
        return cat

    matched_rh = rh_cluster_idx >= 0
    if not matched_rh.any():
        cat[active_idx] = PDG.UNASSOC_IDX
        return cat

    # Build KD-tree of matched RecHit positions
    rh_pts = np.column_stack([rh_x[matched_rh], rh_y[matched_rh], rh_z[matched_rh]])
    rh_cl_idx = rh_cluster_idx[matched_rh]
    tree = cKDTree(rh_pts)

    wf_pts = np.column_stack([
        union["x"][active_idx],
        union["y"][active_idx],
        union["z"][active_idx],
    ])
    dists, nn_idx = tree.query(wf_pts, k=1)

    for i, (dist, ni) in enumerate(zip(dists, nn_idx)):
        union_i = active_idx[i]
        if dist <= global_radius:
            cl_i = int(rh_cl_idx[ni])
            if 0 <= cl_i < cl_pdg.size:
                cat[union_i] = PDG.pdg_to_idx(int(cl_pdg[cl_i]))
            else:
                cat[union_i] = PDG.RARE_IDX
        else:
            cat[union_i] = PDG.UNASSOC_IDX

    return cat


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class EventCache:
    """LRU-style in-memory cache keyed by (root_path, event_index)."""

    def __init__(self, maxsize: int = 20) -> None:
        self._maxsize = maxsize
        self._store: dict[tuple, EventPayload] = {}
        self._order: list[tuple] = []

    def get(self, key: tuple) -> EventPayload | None:
        return self._store.get(key)

    def put(self, key: tuple, payload: EventPayload) -> None:
        if key in self._store:
            self._order.remove(key)
        self._store[key] = payload
        self._order.append(key)
        while len(self._order) > self._maxsize:
            evict = self._order.pop(0)
            del self._store[evict]


_CACHE = EventCache(maxsize=20)


def load_event(
    root_path: str,
    event_index: int,
    union: UnionGeom,
    global_radius: float,
) -> EventPayload:
    """Load and return a fully-annotated event payload (cached)."""
    key = (root_path, event_index)
    cached = _CACHE.get(key)
    if cached is not None:
        return cached

    with uproot.open(root_path) as f:
        # Awkward output is robust across both TTree and RNTuple inputs.
        arr = f["Events"].arrays(
            _BRANCHES,
            entry_start=event_index,
            entry_stop=event_index + 1,
            library="ak",
        )

    rh_x       = np.asarray(arr["RecHitHGC_x"][0],       dtype=np.float32)
    rh_y       = np.asarray(arr["RecHitHGC_y"][0],       dtype=np.float32)
    rh_z       = np.asarray(arr["RecHitHGC_z"][0],       dtype=np.float32)
    rh_energy  = np.asarray(arr["RecHitHGC_energy"][0],  dtype=np.float32)
    rh_cl_idx  = np.asarray(arr["RecHitHGC_MergedSimClusterBestMatchIdx"][0], dtype=np.int32)

    cl_x       = np.asarray(arr["MergedSimCluster_impactPoint_x"][0],   dtype=np.float32)
    cl_y       = np.asarray(arr["MergedSimCluster_impactPoint_y"][0],   dtype=np.float32)
    cl_z       = np.asarray(arr["MergedSimCluster_impactPoint_z"][0],   dtype=np.float32)
    cl_pdg     = np.asarray(arr["MergedSimCluster_pdgId"][0],           dtype=np.int32)
    cl_pt      = np.asarray(arr["MergedSimCluster_pt"][0],              dtype=np.float32)
    cl_eta     = np.asarray(arr["MergedSimCluster_eta"][0],             dtype=np.float32)
    cl_phi     = np.asarray(arr["MergedSimCluster_phi"][0],             dtype=np.float32)
    cl_energy  = np.asarray(arr["MergedSimCluster_sumHitEnergy"][0],    dtype=np.float32)

    wf_x       = np.asarray(arr["L1THGCAL_wafer_x"][0],      dtype=np.float32)
    wf_y       = np.asarray(arr["L1THGCAL_wafer_y"][0],      dtype=np.float32)
    wf_z       = np.asarray(arr["L1THGCAL_wafer_z"][0],      dtype=np.float32)
    wf_layer   = np.asarray(arr["L1THGCAL_wafer_layer"][0],  dtype=np.int32)
    wf_energy  = np.asarray(arr["L1THGCAL_wafer_energy"][0], dtype=np.float32)

    union_energy, union_present = _map_event_energy(
        union, wf_layer, wf_x, wf_y, wf_z, wf_energy
    )
    union_cat = _assign_wafer_categories(
        union, union_present, rh_x, rh_y, rh_z, rh_cl_idx, cl_pdg, global_radius
    )

    payload: EventPayload = {
        "rh_x": rh_x, "rh_y": rh_y, "rh_z": rh_z,
        "rh_energy": rh_energy, "rh_cluster_idx": rh_cl_idx,
        "cl_x": cl_x, "cl_y": cl_y, "cl_z": cl_z,
        "cl_pdg": cl_pdg, "cl_pt": cl_pt,
        "cl_eta": cl_eta, "cl_phi": cl_phi, "cl_energy": cl_energy,
        "union_energy": union_energy,
        "union_present": union_present,
        "union_cat": union_cat,
    }
    _CACHE.put(key, payload)
    return payload


def count_events(root_path: str) -> int:
    """Return number of events in the ROOT file."""
    with uproot.open(root_path) as f:
        return int(f["Events"].num_entries)


def scan_max_energy(root_path: str, events: list[int]) -> float:
    """Quick scan for global max wafer energy across the given event list."""
    max_e = 0.0
    with uproot.open(root_path) as f:
        tree = f["Events"]
        # Read contiguous ranges in a single batch for efficiency.
        sorted_evs = sorted(set(events))
        # Build contiguous runs to minimise the number of tree.arrays calls.
        runs: list[tuple[int, int]] = []
        for ev in sorted_evs:
            if runs and ev == runs[-1][1]:
                runs[-1] = (runs[-1][0], ev + 1)
            else:
                runs.append((ev, ev + 1))
        for start, stop in runs:
            arr = tree.arrays(
                ["L1THGCAL_wafer_energy"],
                entry_start=start, entry_stop=stop,
                library="ak",
            )
            for i in range(len(arr)):
                wf_e = np.asarray(arr["L1THGCAL_wafer_energy"][i])
                if wf_e.size:
                    max_e = max(max_e, float(wf_e.max()))
    return max(max_e, 1.0)
