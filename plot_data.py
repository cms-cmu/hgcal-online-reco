import uproot
import awkward as ak
import numpy as np
import matplotlib.pyplot as plt
import mplhep as hep
from matplotlib.colors import LogNorm
import sys

# Set CMS style
hep.style.use(hep.style.CMS)

def plot_eta_phi(root_file):
    print(f"Plotting Eta vs Phi from {root_file}...")
    sys.stdout.flush()
    
    try:
        with uproot.open(root_file) as file:
            tree = file["Events"]
            
            print("  Reading data...")
            sys.stdout.flush()
            
            # Using awkward array with library="ak" explicitly
            eta_ak = tree["MergedSimCluster_eta"].array(library="ak")
            phi_ak = tree["MergedSimCluster_phi"].array(library="ak")
            energy_ak = tree["MergedSimCluster_sumHitEnergy"].array(library="ak")
            
            print("  Flattening...")
            sys.stdout.flush()
            
            eta = ak.flatten(eta_ak)
            phi = ak.flatten(phi_ak)
            energy = ak.flatten(energy_ak)
            
            print("  Converting to numpy...")
            sys.stdout.flush()
            
            eta_np = ak.to_numpy(eta)
            phi_np = ak.to_numpy(phi)
            energy_np = ak.to_numpy(energy)
            
            print(f"  Plotting {len(eta_np)} points...")
            sys.stdout.flush()
            
            fig, ax = plt.subplots(1, 2, figsize=(20, 9))
                
            # 1. Occupancy Plot (Counts)
            h1 = ax[0].hist2d(eta_np, phi_np, bins=(100, 100), cmap='viridis', norm=LogNorm(vmin=1))
            cbar1 = fig.colorbar(h1[3], ax=ax[0])
            cbar1.set_label('Counts', fontsize=14)
            cbar1.ax.tick_params(labelsize=12)
            
            ax[0].set_xlabel(r'$\eta$', fontsize=16)
            ax[0].set_ylabel(r'$\phi$', fontsize=16)
            ax[0].set_title('MergedSimCluster Occupancy', fontsize=20, pad=10)
            ax[0].tick_params(axis='both', which='major', labelsize=14)
            hep.cms.label(ax=ax[0], data=False, label="Simulation", rlabel="")
            
            # Add stats
            text_str = f"Entries: {len(eta_np)}\nMean $\eta$: {np.mean(eta_np):.2f}\nMean $\phi$: {np.mean(phi_np):.2f}"
            props = dict(boxstyle='round', facecolor='white', alpha=0.8)
            ax[0].text(0.05, 0.95, text_str, transform=ax[0].transAxes, fontsize=12,
                    verticalalignment='top', bbox=props)
                
            # 2. Energy Weighted Plot
            mask = energy_np > 0
            h2 = ax[1].hist2d(eta_np[mask], phi_np[mask], bins=(100, 100), weights=energy_np[mask], cmap='plasma', norm=LogNorm(vmin=1e-2))
            cbar2 = fig.colorbar(h2[3], ax=ax[1])
            cbar2.set_label('Total Energy (GeV)', fontsize=14)
            cbar2.ax.tick_params(labelsize=12)
            
            ax[1].set_xlabel(r'$\eta$', fontsize=16)
            ax[1].set_ylabel(r'$\phi$', fontsize=16)
            ax[1].set_title('Energy Weighted Distribution', fontsize=20, pad=10)
            ax[1].tick_params(axis='both', which='major', labelsize=14)
            hep.cms.label(ax=ax[1], data=False, label="Simulation", rlabel="")
            
            plt.tight_layout()
            
            plt.savefig('plot_eta_phi.png')
            print("Saved plot_eta_phi.png")
            sys.stdout.flush()
            plt.close()

    except Exception as e:
        print(f"Error creating Eta-Phi plot: {e}")
        import traceback
        traceback.print_exc()

def plot_mergedsimcluster_kinematics(root_file):
    print(f"Plotting MergedSimCluster Kinematics from {root_file}...")
    sys.stdout.flush()
    try:
        with uproot.open(root_file) as file:
            tree = file["Events"]
            pt_ak = tree["MergedSimCluster_pt"].array(library="ak")
            eta_ak = tree["MergedSimCluster_eta"].array(library="ak")
            phi_ak = tree["MergedSimCluster_phi"].array(library="ak")
            energy_ak = tree["MergedSimCluster_sumHitEnergy"].array(library="ak")
            
            pt = ak.to_numpy(ak.flatten(pt_ak))
            eta = ak.to_numpy(ak.flatten(eta_ak))
            phi = ak.to_numpy(ak.flatten(phi_ak))
            energy = ak.to_numpy(ak.flatten(energy_ak))
            
            fig, ax = plt.subplots(2, 2, figsize=(16, 12))
            
            ax[0, 0].hist(pt, bins=100, log=True, color='blue', alpha=0.7)
            ax[0, 0].set_xlabel('pT (GeV)')
            ax[0, 0].set_ylabel('Counts')
            ax[0, 0].set_title('MergedSimCluster pT')
            
            ax[0, 1].hist(eta, bins=100, color='green', alpha=0.7)
            ax[0, 1].set_xlabel(r'$\eta$')
            ax[0, 1].set_ylabel('Counts')
            ax[0, 1].set_title(r'MergedSimCluster $\eta$')
            
            ax[1, 0].hist(phi, bins=100, color='red', alpha=0.7)
            ax[1, 0].set_xlabel(r'$\phi$')
            ax[1, 0].set_ylabel('Counts')
            ax[1, 0].set_title(r'MergedSimCluster $\phi$')
            
            ax[1, 1].hist(energy, bins=100, log=True, color='purple', alpha=0.7)
            ax[1, 1].set_xlabel('Energy (GeV)')
            ax[1, 1].set_ylabel('Counts')
            ax[1, 1].set_title('MergedSimCluster Total Energy')
            
            plt.tight_layout()
            plt.savefig('plot_mergedsimcluster_kinematics.png')
            print("Saved plot_mergedsimcluster_kinematics.png")
            plt.close()
    except Exception as e:
        print(f"Error creating kinematics plot: {e}")

def plot_latent(npz_file):
    print(f"Plotting Latent Space from {npz_file}...")
    sys.stdout.flush()
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
        
        # 2. L1 Latent Space Images (Grid of 4x4)
        num_images = 16  # Plot the first 16 latent vectors as 4x4 images
        fig, axes = plt.subplots(4, 4, figsize=(12, 12))
        fig.suptitle('L1 Latent Space (4x4 representations of first 16 wafers)', fontsize=16)
        
        for i, ax in enumerate(axes.flatten()):
            if i < len(latent):
                if latent[i].size == 16:
                    img = latent[i].reshape(4, 4)
                    ax.imshow(img, cmap='viridis', aspect='auto')
                    ax.set_title(f'Wafer {i}')
                else:
                    ax.text(0.5, 0.5, 'Invalid shape', ha='center', va='center')
            ax.axis('off')
        
        plt.tight_layout()
        plt.savefig('plot_latent_images.png')
        print("Saved plot_latent_images.png")
        sys.stdout.flush()
        plt.close()

    except Exception as e:
        print(f"Error creating Latent Space plot: {e}")

def main():
    start_msg = "Starting data plotting..."
    print(start_msg)
    sys.stdout.flush()
    
    root_file = "data/output_Phase2_HGCalL1T_Clustering_1.root"
    npz_file = "data/output_Phase2_HGCalL1T_Clustering_1_latent.npz"
    
    plot_eta_phi(root_file)
    plot_mergedsimcluster_kinematics(root_file)
    plot_latent(npz_file)

if __name__ == "__main__":
    main()
