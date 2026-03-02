#!/usr/bin/env python
"""
Minimum Working Example: Load Event Data from NPZ and ROOT Files

Shows how to:
1. Load latent space from NPZ file for a specific event
2. Load MergedSimCluster data from ROOT file for the same event
3. Access wafer-level information

Usage:
    python load_events.py
"""

import numpy as np
import uproot
import awkward as ak

# ============================================================================
# CONFIGURATION
# ============================================================================

CONFIG = {
    'root_file': 'data/output_Phase2_HGCalL1T_Clustering_1.root',
    'npz_file': 'data/output_Phase2_HGCalL1T_Clustering_1_latent.npz',
    'event_id': 42,
}

# ============================================================================
# LOAD NPZ DATA
# ============================================================================

def load_npz_event(npz_file, event_id):
    """Load latent space for a specific event from NPZ file"""

    data = np.load(npz_file)

    # Arrays in NPZ file
    latent = data['latent']          # (N_wafers, 16) - latent space
    conditions = data['conditions']  # (N_wafers, 8)  - condition features
    event_index = data['event_index']  # (N_wafers,) - event ID
    elink_id = data['elink_id']      # (N_wafers,) - encoder ID (2,3,4,5)

    # Filter to specific event
    mask = event_index == event_id

    return {
        'latent': latent[mask],
        'conditions': conditions[mask],
        'elink_id': elink_id[mask],
        'n_wafers': np.sum(mask)
    }

# ============================================================================
# LOAD ROOT DATA
# ============================================================================

def load_root_event(root_file, event_id):
    """Load MergedSimCluster and wafer data from ROOT file for a specific event"""

    with uproot.open(root_file) as f:
        tree = f['Events']

        # Define branches to read
        branches = [
            # Wafer information
            "L1THGCAL_wafer_layer",
            "L1THGCAL_wafer_eta",
            "L1THGCAL_wafer_waferv",
            "L1THGCAL_wafer_waferu",

            # MergedSimCluster information
            "MergedSimCluster_pt",
            "MergedSimCluster_eta",
            "MergedSimCluster_phi",
            "MergedSimCluster_sumHitEnergy",
        ]

        # Load single event
        data = tree.arrays(
            branches,
            library="ak",
            entry_start=event_id,
            entry_stop=event_id + 1
        )

        # Extract data (index [0] because we loaded only one event)
        return {
            'wafers': {
                'layer': ak.to_numpy(data["L1THGCAL_wafer_layer"][0]),
                'eta': ak.to_numpy(data["L1THGCAL_wafer_eta"][0]),
                'wafer_v': ak.to_numpy(data["L1THGCAL_wafer_waferv"][0]),
                'wafer_u': ak.to_numpy(data["L1THGCAL_wafer_waferu"][0]),
                'n_wafers': len(data["L1THGCAL_wafer_layer"][0])
            },
            'sim_clusters': {
                'pt': ak.to_numpy(data["MergedSimCluster_pt"][0]),
                'eta': ak.to_numpy(data["MergedSimCluster_eta"][0]),
                'phi': ak.to_numpy(data["MergedSimCluster_phi"][0]),
                'energy': ak.to_numpy(data["MergedSimCluster_sumHitEnergy"][0]),
                'n_clusters': len(data["MergedSimCluster_pt"][0])
            }
        }

# ============================================================================
# MAIN
# ============================================================================

def main():
    event_id = CONFIG['event_id']

    print("\n" + "="*70)
    print(f"LOADING EVENT {event_id}")
    print("="*70)

    # Load from NPZ
    print("\n1. Loading latent space from NPZ file...")
    npz_data = load_npz_event(CONFIG['npz_file'], event_id)

    print(f"   Total wafers: {npz_data['n_wafers']}")
    print(f"   Latent space shape: {npz_data['latent'].shape}")
    print(f"   Conditions shape: {npz_data['conditions'].shape}")
    print(f"   eLinks used: {np.unique(npz_data['elink_id'])}")

    # Breakdown by eLink
    print("\n   Per-eLink breakdown:")
    for elink in np.unique(npz_data['elink_id']):
        n = np.sum(npz_data['elink_id'] == elink)
        print(f"     eLink {elink}: {n} wafers")

    # Load from ROOT
    print(f"\n2. Loading data from ROOT file...")
    root_data = load_root_event(CONFIG['root_file'], event_id)

    print(f"   Total wafers in event (unfiltered): {root_data['wafers']['n_wafers']}")
    print(f"   SimClusters in event: {root_data['sim_clusters']['n_clusters']}")

    # Verify eLink filtering
    print(f"\n   Verifying eLink filters on ROOT data:")
    layers = root_data['wafers']['layer']

    # Count wafers per eLink filter
    elink_configs = {
        2: (layers < 7) | (layers > 13),
        3: layers == 13,
        4: (layers == 7) | (layers == 11),
        5: (layers <= 11) & (layers >= 5),
    }

    total_filtered = 0
    for elink_id, mask in elink_configs.items():
        n_wafers = np.sum(mask)
        total_filtered += n_wafers
        print(f"     eLink {elink_id}: {n_wafers} wafers")

    print(f"   Total after all eLink filters: {total_filtered}")
    print(f"   NPZ total: {npz_data['n_wafers']}")

    if total_filtered == npz_data['n_wafers']:
        print(f"   ✓ Match!")
    else:
        print(f"   ⚠️  Mismatch: {abs(total_filtered - npz_data['n_wafers'])} wafer difference")

    # Show first few SimClusters
    if root_data['sim_clusters']['n_clusters'] > 0:
        print("\n   First 5 MergedSimClusters:")
        for i in range(min(5, root_data['sim_clusters']['n_clusters'])):
            pt = root_data['sim_clusters']['pt'][i]
            eta = root_data['sim_clusters']['eta'][i]
            phi = root_data['sim_clusters']['phi'][i]
            energy = root_data['sim_clusters']['energy'][i]
            print(f"     Cluster {i}: pT={pt:.2f}, eta={eta:.2f}, phi={phi:.2f}, E={energy:.2f}")

    # Show example latent space values
    print(f"\n3. Example latent space values (first wafer):")
    print(f"   Latent (first 8 dims): {npz_data['latent'][0, :8]}")
    print(f"   Conditions: {npz_data['conditions'][0]}")

    # Usage examples
    print("\n" + "="*70)
    print("USAGE EXAMPLES")
    print("="*70)
    print("""
# Load NPZ file
data = np.load('data/output_Phase2_HGCalL1T_Clustering_1_latent.npz')

# Get event 42, eLink 2
mask = (data['event_index'] == 42) & (data['elink_id'] == 2)
latent_event42_elink2 = data['latent'][mask]

# Get all wafers from event 42
mask = data['event_index'] == 42
latent_event42_all = data['latent'][mask]
elinks_event42 = data['elink_id'][mask]

# Load ROOT file
with uproot.open('data/output_Phase2_HGCalL1T_Clustering_1.root') as f:
    tree = f['Events']
    data = tree.arrays(['MergedSimCluster_pt', 'MergedSimCluster_eta'],
                       entry_start=42, entry_stop=43)
    pt = data['MergedSimCluster_pt'][0]
    eta = data['MergedSimCluster_eta'][0]
""")

if __name__ == "__main__":
    main()
