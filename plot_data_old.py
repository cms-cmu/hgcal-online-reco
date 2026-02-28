print("Starting script...")
import uproot
import sys
import numpy as np
import matplotlib.pyplot as plt
import mplhep as hep
import awkward as ak

# Set CMS style
# hep.style.use(hep.style.CMS)

from matplotlib.colors import LogNorm

def plot_eta_phi(root_file):
    print(f"Plotting Eta vs Phi from {root_file}...")
    try:
        with uproot.open(root_file) as file:
            tree = file["Events"]
            
            # Load Data
            print("  Reading Data...")
            eta_ak = tree["MergedSimCluster_eta"].array(library="ak")
            phi_ak = tree["MergedSimCluster_phi"].array(library="ak")
            energy_ak = tree["MergedSimCluster_sumHitEnergy"].array(library="ak")
            
            print("  Flattening and converting to numpy...")
            eta = ak.to_numpy(ak.flatten(eta_ak))
            phi = ak.to_numpy(ak.flatten(phi_ak))
            energy = ak.to_numpy(ak.flatten(energy_ak))
            
            print(f"  Plotting {len(eta)} points...")
            
            # Create Plot
            print("  Initializing figure...")
            fig, ax = plt.subplots(1, 2, figsize=(20, 9))
            
            # 1. Occupancy Plot (Counts)
            print("  Creating Occupancy Plot...")
            # Use vmin=1 to avoid log(0) issues for empty bins
            h1 = ax[0].hist2d(eta, phi, bins=(100, 100), cmap='viridis', norm=LogNorm(vmin=1))
            print("  Adding colorbar 1...")
            fig.colorbar(h1[3], ax=ax[0], label='Counts')
            ax[0].set_xlabel(r'$\eta$')
            ax[0].set_ylabel(r'$\phi$')
            ax[0].set_title('MergedSimCluster Occupancy (Eta vs Phi)')
            print("  Adding CMS label 1...")
            # hep.cms.label(ax=ax[0], data=False, label="Simulation", rlabel="")
            
            # Add stats to plot 1
            print("  Adding stats text...")
            text_str = f"Entries: {len(eta)}\nMean $\eta$: {np.mean(eta):.2f}\nMean $\phi$: {np.mean(phi):.2f}"
            props = dict(boxstyle='round', facecolor='white', alpha=0.8)
            ax[0].text(0.05, 0.95, text_str, transform=ax[0].transAxes, fontsize=12,
                    verticalalignment='top', bbox=props)

            # 2. Energy Weighted Plot
            print("  Creating Energy Plot...")
            sys.stdout.flush()
            
            # Filter for positive energy just to be safe with LogNorm
            mask = energy > 0
            eta_pos = eta[mask]
            phi_pos = phi[mask]
            energy_pos = energy[mask]
            
            print(f"  Filtered {len(energy) - len(energy_pos)} zero/negative energy entries.")
            sys.stdout.flush()

            # However, if a bin sum is 0, LogNorm needs vmin.
            # We also set vmin to a small positive value.
            h2 = ax[1].hist2d(eta_pos, phi_pos, bins=(100, 100), weights=energy_pos, cmap='plasma', norm=LogNorm(vmin=1e-2))
            print("  Adding colorbar 2...")
            sys.stdout.flush()
            fig.colorbar(h2[3], ax=ax[1], label='Total Energy (GeV)')
            ax[1].set_xlabel(r'$\eta$')
            ax[1].set_ylabel(r'$\phi$')
            ax[1].set_title('Energy Weighted Distribution')
            print("  Adding CMS label 2...")
            sys.stdout.flush()
            # hep.cms.label(ax=ax[1], data=False, label="Simulation", rlabel="")

            print("  Saving figure...")
            sys.stdout.flush()
            plt.tight_layout()
            plt.savefig('plot_eta_phi.png')
            print("Saved plot_eta_phi.png")
            sys.stdout.flush()
            plt.close()
            
    except Exception as e:
        print(f"Error creating Eta-Phi plot: {e}")
        import traceback
        traceback.print_exc()
            
    except Exception as e:
        print(f"Error creating Eta-Phi plot: {e}")

def plot_latent(npz_file):
    print(f"Plotting Latent Space from {npz_file}...")
    try:
        data = np.load(npz_file)
        latent = data['latent'] # Shape: (N_wafers, 16)
        
        # 1. Distribution of all latent values
        plt.figure(figsize=(10, 6))
        plt.hist(latent.flatten(), bins=100, log=True)
        plt.xlabel('Latent Value')
        plt.ylabel('Count (Log Scale)')
        plt.title('Distribution of Latent Space Values')
        plt.savefig('plot_latent_dist.png')
        print("Saved plot_latent_dist.png")
        plt.close()
        
        # 2. Heatmap of first 100 wafers
        plt.figure(figsize=(12, 8))
        plt.imshow(latent[:100], aspect='auto', cmap='viridis', interpolation='nearest')
        plt.colorbar(label='Latent Value')
        plt.xlabel('Latent Dimension (0-15)')
        plt.ylabel('Wafer Index (0-99)')
        plt.title('Latent Space Vectors (First 100 Wafers)')
        plt.savefig('plot_latent_heatmap.png')
        print("Saved plot_latent_heatmap.png")
        plt.close()

    except Exception as e:
        print(f"Error creating Latent Space plot: {e}")

def main():
    root_file = "data/output_Phase2_HGCalL1T_Clustering_1.root"
    npz_file = "data/output_Phase2_HGCalL1T_Clustering_1_latent.npz"
    
    plot_eta_phi(root_file)
    plot_latent(npz_file)

if __name__ == "__main__":
    main()
