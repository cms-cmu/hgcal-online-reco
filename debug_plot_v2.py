import uproot
import awkward as ak
import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np
from matplotlib.colors import LogNorm

hep.style.use(hep.style.CMS) # Commented out to test if this is the cause

try:
    print("Opening ROOT file...")
    with uproot.open("data/output_Phase2_HGCalL1T_Clustering_1.root") as file:
        tree = file["Events"]
        
        print("Reading MergedSimCluster_eta...")
        eta_ak = tree["MergedSimCluster_eta"].array(library="ak")
        print(f"Eta AK Type: {type(eta_ak)}")
        
        print("Flattening Eta...")
        eta = ak.flatten(eta_ak)
        print(f"Eta Flattened Type: {type(eta)}")
        print(f"Eta Shape: {len(eta)}")

        print("Reading MergedSimCluster_phi...")
        phi_ak = tree["MergedSimCluster_phi"].array(library="ak")
        print("Reading MergedSimCluster_sumHitEnergy...")
        energy_ak = tree["MergedSimCluster_sumHitEnergy"].array(library="ak")
        phi = ak.flatten(phi_ak)
        energy = ak.flatten(energy_ak)
        
        print("Converting to numpy...")
        eta_np = ak.to_numpy(eta)
        phi_np = ak.to_numpy(phi)
        energy_np = ak.to_numpy(energy)
        
        print("Plotting histogram...")
        fig, ax = plt.subplots(1, 2, figsize=(20, 9))
            
        # 1. Occupancy Plot (Counts)
        h1 = ax[0].hist2d(eta_np, phi_np, bins=(100, 100), cmap='viridis', norm=LogNorm(vmin=1))
        fig.colorbar(h1[3], ax=ax[0], label='Counts')
        ax[0].set_title('MergedSimCluster Occupancy (Eta vs Phi)')
        hep.cms.label(ax=ax[0], data=False, label="Simulation", rlabel="")
            
        # 2. Energy Weighted Plot
        mask = energy_np > 0
        h2 = ax[1].hist2d(eta_np[mask], phi_np[mask], bins=(100, 100), weights=energy_np[mask], cmap='plasma', norm=LogNorm(vmin=1e-2))
        fig.colorbar(h2[3], ax=ax[1], label='Total Energy (GeV)')
        ax[1].set_title('Energy Weighted Distribution')
        hep.cms.label(ax=ax[1], data=False, label="Simulation", rlabel="")
        
        plt.tight_layout()
        
        plt.savefig('debug_plot_eta_phi.png')
        print("Saved debug_plot_eta_phi.png")

except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
