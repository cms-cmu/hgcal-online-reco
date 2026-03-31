"""HGCAL Viewer — Dash app factory and CLI entry point.

Usage
-----
    # One-time geometry setup (if wafer_mapping.csv doesn't exist yet):
    hgcal-viewer prepare --root path/to/file.root

    # Launch dashboard:
    hgcal-viewer --root path/to/file.root [--events 0-49] [--csv wafer_mapping.csv] [--meta wafer_mapping_meta.json]
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
from dash import Dash, Input, Output, dcc, html

from hgcal_viewer.data.geometry import (
    load_meta,
    load_union_geometry,
    build_union_geometry_from_root,
)
from hgcal_viewer.data.loader import count_events, load_event, scan_max_energy
from hgcal_viewer.filters import (
    ID_ENDCAP_FILTER,
    ID_ENERGY_LOG,
    ID_EVENT,
    ID_MAX_RECHITS,
    ID_MIN_ENERGY,
    ID_PDG_FILTER,
    ID_RECHIT_MODE,
    ID_SHOW_CLUSTERS,
    ID_WAFER_MODE,
    sidebar,
)
from hgcal_viewer.panels import cluster_summary, energy_dist, view3d, wafer_counts


# ---------------------------------------------------------------------------
# Argument parsing helpers
# ---------------------------------------------------------------------------

def _parse_events(value: str, n_total: int) -> list[int]:
    events: list[int] = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-", 1)
            try:
                lo_i, hi_i = int(lo), int(hi)
            except ValueError:
                raise ValueError(
                    f"Invalid event range '{part}'. "
                    "Use integers like '0-49' or a comma-separated list like '0,5,12'."
                )
            if lo_i < 0 or hi_i < 0:
                raise ValueError(
                    f"Negative indices are not allowed in range '{part}'. "
                    "Use non-negative integers like '0-49'."
                )
            if hi_i < lo_i:
                raise ValueError(
                    f"Invalid event range '{part}': end ({hi_i}) must be >= start ({lo_i})."
                )
            events.extend(range(lo_i, min(hi_i + 1, n_total)))
        else:
            try:
                idx = int(part)
            except ValueError:
                raise ValueError(
                    f"Invalid event index '{part}'. "
                    "Use integers like '0-49' or a comma-separated list like '0,5,12'."
                )
            if idx < 0:
                raise ValueError(
                    f"Negative event index '{idx}' is not allowed. "
                    "Use non-negative integers like '0,5,12'."
                )
            events.append(idx)
    seen: set[int] = set()
    unique: list[int] = []
    for e in events:
        if 0 <= e < n_total and e not in seen:
            seen.add(e)
            unique.append(e)
    return unique


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

def create_app(
    root_path: str,
    csv_path: str,
    meta_path: str,
    events: list[int],
) -> Dash:
    union        = load_union_geometry(csv_path)
    meta         = load_meta(meta_path)
    global_radius = float(meta["global_radius"])
    max_energy   = scan_max_energy(root_path, events)
    max_energy_log = float(np.log1p(max_energy))

    app = Dash(__name__, title="HGCAL Viewer")

    app.layout = html.Div(
        [
            # ── Sidebar ──────────────────────────────────────────────────────
            sidebar(events, max_energy),

            # ── Main panel grid ──────────────────────────────────────────────
            html.Div(
                [
                    # Row 1: 3D view (full width)
                    html.Div(
                        dcc.Graph(
                            id="graph-3d",
                            style={"height": "560px"},
                            config={"scrollZoom": True},
                        ),
                        style={"width": "100%"},
                    ),
                    # Row 2: three side panels
                    html.Div(
                        [
                            html.Div(
                                dcc.Graph(id="graph-wafer-counts", style={"height": "320px"}),
                                style={"flex": "1", "minWidth": "0"},
                            ),
                            html.Div(
                                dcc.Graph(id="graph-energy-dist", style={"height": "320px"}),
                                style={"flex": "1", "minWidth": "0"},
                            ),
                            html.Div(
                                dcc.Graph(id="graph-cluster-summary", style={"height": "320px"}),
                                style={"flex": "1", "minWidth": "0"},
                            ),
                        ],
                        style={"display": "flex", "gap": "4px", "width": "100%"},
                    ),
                ],
                style={"flex": "1", "minWidth": "0", "padding": "8px", "overflowY": "auto"},
            ),
        ],
        style={"display": "flex", "height": "100vh", "fontFamily": "sans-serif"},
    )

    # ── Single master callback ───────────────────────────────────────────────
    @app.callback(
        Output("graph-3d",              "figure"),
        Output("graph-wafer-counts",    "figure"),
        Output("graph-energy-dist",     "figure"),
        Output("graph-cluster-summary", "figure"),
        Input(ID_EVENT,         "value"),
        Input(ID_WAFER_MODE,    "value"),
        Input(ID_MIN_ENERGY,    "value"),
        Input(ID_RECHIT_MODE,   "value"),
        Input(ID_MAX_RECHITS,   "value"),
        Input(ID_SHOW_CLUSTERS, "value"),
        Input(ID_PDG_FILTER,    "value"),
        Input(ID_ENDCAP_FILTER, "value"),
        Input(ID_ENERGY_LOG,    "value"),
    )
    def update_all(
        event_idx,
        wafer_mode,
        min_energy,
        rechit_mode,
        max_rechits,
        show_clusters_val,
        pdg_filter,
        endcap,
        energy_log,
    ):
        data = load_event(
            root_path=root_path,
            event_index=int(event_idx),
            union=union,
            global_radius=global_radius,
        )

        active_cats   = list(pdg_filter or [])
        show_clusters = bool(show_clusters_val and "show" in show_clusters_val)
        log_scale     = energy_log == "log"
        min_e         = float(min_energy or 0.0)
        max_rh        = int(max_rechits or 15000)

        fig_3d = view3d.build_figure(
            union=union,
            data=data,
            event_index=int(event_idx),
            global_radius=global_radius,
            max_energy_log=max_energy_log,
            wafer_mode=wafer_mode,
            min_energy=min_e,
            rechit_mode=rechit_mode,
            max_rechits=max_rh,
            show_clusters=show_clusters,
            active_cats=active_cats,
            endcap=endcap,
        )

        fig_counts = wafer_counts.build_figure(
            union=union,
            data=data,
            active_cats=active_cats,
            endcap=endcap,
        )

        fig_energy = energy_dist.build_figure(
            union=union,
            data=data,
            active_cats=active_cats,
            endcap=endcap,
            log_scale=log_scale,
            min_energy=min_e,
        )

        fig_clusters = cluster_summary.build_figure(
            data=data,
            active_cats=active_cats,
            endcap=endcap,
        )

        return fig_3d, fig_counts, fig_energy, fig_clusters

    return app


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="hgcal-viewer",
        description="Interactive HGCAL event visualization dashboard",
    )
    sub = parser.add_subparsers(dest="command")

    # ── prepare subcommand ───────────────────────────────────────────────────
    prep = sub.add_parser(
        "prepare",
        help="Generate wafer_mapping.csv and wafer_mapping_meta.json from a ROOT file.",
    )
    prep.add_argument("--root",     required=True, help="Input ROOT file")
    prep.add_argument("--csv-out",  default="wafer_mapping.csv",
                      help="Output CSV path (default: wafer_mapping.csv)")
    prep.add_argument("--meta-out", default="wafer_mapping_meta.json",
                      help="Output metadata JSON path (default: wafer_mapping_meta.json)")

    # ── serve (default) ──────────────────────────────────────────────────────
    parser.add_argument("--root",   default=None, help="ROOT file to visualize")
    parser.add_argument("--csv",    default="wafer_mapping.csv",
                        help="Wafer mapping CSV (default: wafer_mapping.csv)")
    parser.add_argument("--meta",   default="wafer_mapping_meta.json",
                        help="Wafer mapping metadata JSON (default: wafer_mapping_meta.json)")
    parser.add_argument("--events", default="0-49",
                        help="Event range/list, e.g. '0-49' or '0,5,12'")
    parser.add_argument("--host",   default="127.0.0.1",
                        help="Host address to bind (default: 127.0.0.1)")
    parser.add_argument("--port",   type=int, default=8050,
                        help="Port to bind the dashboard server (default: 8050)")
    parser.add_argument("--debug",  action="store_true",
                        help="Enable Dash hot-reload / debug mode")

    return parser


def main() -> None:
    parser = _build_arg_parser()
    args   = parser.parse_args()

    # ── prepare ─────────────────────────────────────────────────────────────
    if args.command == "prepare":
        if not Path(args.root).exists():
            raise SystemExit(f"ROOT file not found: {args.root}")
        build_union_geometry_from_root(args.root, args.csv_out, args.meta_out)
        return

    # ── serve ────────────────────────────────────────────────────────────────
    if not args.root:
        parser.error("--root is required to launch the viewer")

    if not Path(args.root).exists():
        raise SystemExit(f"ROOT file not found: {args.root}")

    if not Path(args.csv).exists():
        raise SystemExit(
            f"Wafer mapping CSV not found: {args.csv}\n"
            "Run:  hgcal-viewer prepare --root <file.root>  first."
        )
    if not Path(args.meta).exists():
        raise SystemExit(
            f"Wafer mapping meta JSON not found: {args.meta}\n"
            "Run:  hgcal-viewer prepare --root <file.root>  first."
        )

    n_total = count_events(args.root)
    try:
        events = _parse_events(args.events, n_total)
    except ValueError as exc:
        raise SystemExit(str(exc)) from None
    if not events:
        raise SystemExit("No valid events in the requested range.")

    print(f"ROOT file : {args.root}  ({n_total} events total)")
    print(f"Showing   : {len(events)} events  [{events[0]}–{events[-1]}]")
    print(f"Geometry  : {args.csv}")
    print(f"Launching : http://{args.host}:{args.port}/")

    app = create_app(
        root_path=args.root,
        csv_path=args.csv,
        meta_path=args.meta,
        events=events,
    )
    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
