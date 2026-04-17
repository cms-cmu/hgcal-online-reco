"""PDG ID → human label, short label, and colour palette.

Physics process is pp → τ⁺τ⁻, so the dominant species are:
  γ (55%), e± (30%), π± (10%), n/p (5%), μ± (1%), rare (<2%).
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Canonical category definitions
# ---------------------------------------------------------------------------

# Each entry: (pdgIds, label, short, hex_colour)
_CATEGORIES: list[tuple[tuple[int, ...], str, str, str]] = [
    ((22,),          "γ photon",    "γ",   "#f4a261"),   # orange
    ((11,),          "e⁻ electron", "e⁻",  "#2a9d8f"),   # teal
    ((-11,),         "e⁺ positron", "e⁺",  "#57cc99"),   # green
    ((211,),         "π⁺ pion",     "π⁺",  "#e76f51"),   # red-orange
    ((-211,),        "π⁻ pion",     "π⁻",  "#e9c46a"),   # yellow
    ((2112,),        "neutron",     "n",   "#a8dadc"),   # light blue
    ((2212,),        "proton",      "p",   "#457b9d"),   # steel blue
    ((13,),          "μ⁻ muon",     "μ⁻",  "#9b5de5"),   # purple
    ((-13,),         "μ⁺ muon",     "μ⁺",  "#c77dff"),   # lavender
]

# "rare" = any pdgId not in the 9 named species (~1.3% of trainable clusters:
#  K mesons, π⁰, Λ baryons, and other τ⁺τ⁻ decay products)
_RARE_COLOUR = "#6c757d"
_RARE_LABEL  = "Rare particles"
_RARE_SHORT  = "rare"

# "unassociated" = active wafer whose nearest matched RecHit is beyond
#  global_radius — background/noise wafers, ~4k–4.3k per event
_UNASSOC_COLOUR = "#ced4da"
_UNASSOC_LABEL  = "Active"
_UNASSOC_SHORT  = "active"

# ---------------------------------------------------------------------------
# Build lookup tables once at import time
# ---------------------------------------------------------------------------

# pdgId (int) → category index (0-based); unknown pdgId → RARE_IDX
_PDG_TO_IDX: dict[int, int] = {}
for _idx, (_ids, _label, _short, _colour) in enumerate(_CATEGORIES):
    for _pid in _ids:
        _PDG_TO_IDX[_pid] = _idx

RARE_IDX:    int = len(_CATEGORIES)       # unknown pdgId from a matched cluster
UNASSOC_IDX: int = len(_CATEGORIES) + 1  # active wafer, no matched RecHit nearby

LABELS:  list[str] = [c[1] for c in _CATEGORIES] + [_RARE_LABEL,  _UNASSOC_LABEL]
SHORTS:  list[str] = [c[2] for c in _CATEGORIES] + [_RARE_SHORT,  _UNASSOC_SHORT]
COLOURS: list[str] = [c[3] for c in _CATEGORIES] + [_RARE_COLOUR, _UNASSOC_COLOUR]

# All known pdgIds (for reference)
KNOWN_PDGIDS: list[int] = [ids[0] for ids, *_ in _CATEGORIES]


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def pdg_to_idx(pdg_id: int) -> int:
    """Return category index for *pdg_id*; unknown pdgId → RARE_IDX."""
    return _PDG_TO_IDX.get(int(pdg_id), RARE_IDX)


def pdg_to_label(pdg_id: int) -> str:
    return LABELS[pdg_to_idx(pdg_id)]


def pdg_to_colour(pdg_id: int) -> str:
    return COLOURS[pdg_to_idx(pdg_id)]


def category_options() -> list[dict]:
    """Dash checklist options: one entry per category including rare + unassociated."""
    options = []
    for idx, (label, colour) in enumerate(zip(LABELS, COLOURS)):
        options.append({
            "label": label,
            "value": idx,
            "style": {"color": colour},
        })
    return options


def all_category_indices() -> list[int]:
    return list(range(len(LABELS)))
