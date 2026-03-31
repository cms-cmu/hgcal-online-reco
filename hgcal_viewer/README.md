# HGCAL Viewer

An interactive browser-based dashboard for exploring HGCAL simulation events.
Built with [Dash](https://dash.plotly.com/) and [Plotly](https://plotly.com/python/),
it reads CMS NanoAOD-style ROOT files via [uproot](https://uproot.readthedocs.io/).

---

## Requirements

- Python ≥ 3.10 (tested on 3.12)
- A ROOT file containing the `Events` tree with the branches listed below
- `wafer_mapping.csv` and `wafer_mapping_meta.json` (generated once per ROOT file — see [Usage → Step 1](#step-1--generate-the-wafer-geometry-mapping-once-per-root-file))

**Python dependencies** (installed automatically with the package):

| Package | Version |
|---------|---------|
| dash | ≥ 4.0 |
| plotly | ≥ 6.0 |
| uproot | ≥ 5.0 |
| numpy | ≥ 1.24 |
| scipy | ≥ 1.10 |

---

## Quick start (from a downloaded zip)

### 1 — Unzip and enter the directory

After downloading and unzipping, you will have a folder called `hgcal_viewer/`.
Open a terminal and `cd` into that folder (the one that contains `pyproject.toml`):

```bash
unzip hgcal_viewer.zip
cd hgcal_viewer
```

> **Tip:** If you are re-extracting over an existing directory, use
> `unzip -o hgcal_viewer.zip` to overwrite without being prompted.

### 2 — Create a virtual environment and install

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install .
```

This installs all dependencies and registers the `hgcal-viewer` command.

> **Tip — Debian/Ubuntu:** if `python3 -m venv` fails with
> *"ensurepip is not available"*, install the missing package first:
> ```bash
> sudo apt install python3-venv
> ```
> or use [uv](https://github.com/astral-sh/uv) instead:
> ```bash
> curl -LsSf https://astral.sh/uv/install.sh | sh
> uv venv .venv && source .venv/bin/activate
> uv pip install .
> ```

### 3 — Smoke test (no browser needed)

```bash
hgcal-viewer --help
```

If this prints the usage message, the installation is working correctly.

### 4 — End-to-end quick reference

All commands below must be run from the **same directory** — the `hgcal_viewer/`
folder that contains `pyproject.toml`.  The mapping files are written to and
read from the **current working directory**.

```bash
# One-time geometry setup (writes wafer_mapping.csv + wafer_mapping_meta.json here)
hgcal-viewer prepare --root /path/to/events.root

# Launch the dashboard (reads mapping files from the current directory)
hgcal-viewer --root /path/to/events.root

# Then open http://127.0.0.1:8050/ in a browser
```

---

## Usage

> **Working directory:** always run `hgcal-viewer` from the directory where
> your mapping files (`wafer_mapping.csv`, `wafer_mapping_meta.json`) live
> (or will be written).  The commands below assume you are in the `hgcal_viewer/`
> folder (the one containing `pyproject.toml`).

### Step 1 — Generate the wafer geometry mapping (once per ROOT file)

Run this once to build the two geometry files the dashboard needs:

```bash
hgcal-viewer prepare --root path/to/events.root
```

This writes two files in the **current directory**:

- `wafer_mapping.csv` — union wafer positions (layer, u, v, zside, x, y, z)
- `wafer_mapping_meta.json` — derived geometry metadata (global hex radius, etc.)

> **Note:** `prepare` scans every event in the ROOT file to build the union
> geometry.  For a 1 000-event file this takes roughly 30–60 seconds.

Custom output paths:

```bash
hgcal-viewer prepare --root events.root \
    --csv-out my_mapping.csv \
    --meta-out my_mapping_meta.json
```

### Step 2 — Launch the dashboard

```bash
hgcal-viewer --root path/to/events.root
```

Or equivalently:

```bash
python -m hgcal_viewer --root path/to/events.root
```

Then open **<http://127.0.0.1:8050/>** in a browser.

#### Common options

| Flag | Default | Description |
|------|---------|-------------|
| `--root` | *(required)* | Input ROOT file |
| `--events` | `0-49` | Event range or comma-separated list, e.g. `0-99` or `0,5,12`; ranges and individual indices can be mixed, e.g. `0-2,5,8-10` |
| `--csv` | `wafer_mapping.csv` | Wafer mapping CSV |
| `--meta` | `wafer_mapping_meta.json` | Wafer mapping metadata JSON |
| `--host` | `127.0.0.1` | Bind address |
| `--port` | `8050` | Port |
| `--debug` | off | Enable Dash hot-reload |

#### Example

```bash
hgcal-viewer --root events.root --events 0-99 --port 8080 --debug
```

---

## Creating a distributable zip (for developers)

A helper script is included to produce a clean `hgcal_viewer.zip` that users
can download and install.  Run it from anywhere — it automatically excludes
build artifacts, virtual environments, and generated geometry files:

```bash
bash hgcal_viewer/make_zip.sh              # writes hgcal_viewer.zip next to the source dir
bash hgcal_viewer/make_zip.sh /path/out    # writes to /path/out/hgcal_viewer.zip
```

The resulting zip extracts to a single `hgcal_viewer/` folder and contains
only the source files needed for installation.

---

## Dashboard panels

### Sidebar controls

| Control | Description |
|---------|-------------|
| **Event** | Select which event to display |
| **Particle types** | Checklist to show/hide individual PDG categories |
| **Endcap** | Show both endcaps, +z only, or −z only |
| **Wafer display** | Which wafers to draw (fills *and* outlines together): None / All (union) / Active only / Associated only |
| **Min wafer energy** | Slider to hide wafers below an energy threshold |
| **RecHits** | None / Matched only / All |
| **Max RecHits shown** | Cap on randomly-sampled RecHits rendered (default 15 000) |
| **Show cluster impact points** | Toggle MergedSimCluster impact-point markers |
| **Energy axis** | Linear or log(1 + E) scale for the energy distribution plot |

### Main panels

**3D view** — Interactive 3D scatter of wafer fills (coloured by PDG category,
intensity ∝ log energy), wafer outlines, RecHits, and cluster impact points.

**Wafer counts per layer** — Stacked bar chart showing inactive / active /
associated wafers broken down by PDG category for each detector layer.

**Wafer energy distribution** — Overlaid histograms of wafer energy per PDG
category. The *Active* trace covers all present wafers; *Rare particles* covers
only wafers matched to rare-species clusters.

**Clusters: η–φ** — Scatter of MergedSimCluster impact points in η–φ space,
coloured by particle type and sized by √pT.

---

## Particle categories

The physics process is pp → τ⁺τ⁻. Wafers and clusters are assigned to one of
eleven categories:

| Category | PDG ID(s) | Approx. fraction |
|----------|-----------|-----------------|
| γ photon | 22 | ~55% |
| e⁻ electron | 11 | ~30% |
| e⁺ positron | −11 | |
| π⁺ pion | 211 | ~10% |
| π⁻ pion | −211 | |
| neutron | 2112 | ~5% |
| proton | 2212 | |
| μ⁻ muon | 13 | ~1% |
| μ⁺ muon | −13 | |
| Rare particles | all others | <2% |
| Active | active wafer, no nearby matched RecHit | ~4 000–4 300 per event |

---

## Project layout

```
hgcal_viewer/               ← distribution root (unzip here, run pip install .)
├── pyproject.toml
├── README.md
├── make_zip.sh             ← helper to create a clean distributable zip
└── hgcal_viewer/           ← Python package
    ├── app.py              # Dash app factory and CLI entry point
    ├── filters.py          # Sidebar layout and control IDs
    ├── data/
    │   ├── geometry.py     # Union wafer geometry loading and hex helpers
    │   ├── loader.py       # ROOT I/O, event caching, wafer–cluster association
    │   └── pdg.py          # PDG ID → label / colour lookup tables
    └── panels/
        ├── view3d.py        # 3D scatter panel
        ├── wafer_counts.py  # Per-layer stacked bar panel
        ├── energy_dist.py   # Energy distribution histogram panel
        └── cluster_summary.py  # η–φ cluster scatter panel
```

---

## Expected ROOT branches

The `Events` tree must contain the following branches.
Branches marked *(prepare only)* are only needed for the one-time `prepare` step;
the rest are required at runtime by the dashboard.

```
RecHitHGC_x / _y / _z / _energy
RecHitHGC_MergedSimClusterBestMatchIdx
MergedSimCluster_impactPoint_x / _y / _z
MergedSimCluster_pdgId / _pt / _eta / _phi / _sumHitEnergy
L1THGCAL_wafer_x / _y / _z / _layer / _energy
L1THGCAL_wafer_waferu / _waferv / _zside   (prepare only)
```

---

## Troubleshooting

**`ROOT file not found: <path>`**
The path passed to `--root` does not exist or is not accessible.  Check the
path and ensure the file is readable.  Both the `prepare` subcommand and the
serve command validate the file before doing any work.

**`wafer_mapping.csv not found`**
Run `hgcal-viewer prepare --root <file.root>` from the directory where you
want the mapping files written before launching the viewer.  The mapping files
must be in the **current working directory** when you run `hgcal-viewer`.

**`No valid events in the requested range`**
The `--events` value must be a valid range or list within `[0, N)` where *N*
is the number of events in the ROOT file.  Also printed when a reversed range
is given (e.g. `5-3`); the end must be ≥ the start.  Check the event count with:
```bash
python3 -c "import uproot; f=uproot.open('events.root'); print(f['Events'].num_entries)"
```

**Invalid `--events` format**
Only non-negative integers are accepted.  The following will print an error and exit:
- Negative indices: `-1`, `-5`
- Open-ended ranges: `0-`
- Reversed ranges: `5-3` (end < start)

Use the form `0-49`, `0,5,12`, or mixed `0-2,5,8-10`.

**Slow startup with a large `--events` range**
At startup the viewer scans the requested events to determine the global
energy scale.  For very large ranges (> 500 events) consider narrowing the
range with `--events` or using a smaller subset for exploration.

**`prepare` is slow**
`prepare` reads every event in the ROOT file to build the union geometry.
For a 1 000-event file this takes roughly 30–60 seconds.  The result is cached in
`wafer_mapping.csv` / `wafer_mapping_meta.json` and only needs to be run once
per ROOT file.

**Dashboard shows empty panels after deselecting all particle types**
Deselecting all checkboxes in the *Particle types* sidebar control is handled
gracefully — all panels render as empty rather than crashing.

**`pip install .` fails with "package directory does not exist"**
Make sure you are running `pip install .` from inside the `hgcal_viewer/`
folder that contains `pyproject.toml` (not from a parent directory).
