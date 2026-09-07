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

        # Compute sparsity statistics
        nonzero_mask = np.any(latent != 0, axis=1)
        n_nonzero_wafers = np.sum(nonzero_mask)
        print(f"  Latent stats: {n_nonzero_wafers}/{len(latent)} wafers have non-zero values "
              f"({100*n_nonzero_wafers/len(latent):.2f}%)")
        print(f"  Value range: [{latent.min():.4f}, {latent.max():.4f}]")

        # 1. Distribution of NON-ZERO latent values (exclude the dominant zero peak)
        nonzero_vals = latent.flatten()
        nonzero_vals = nonzero_vals[nonzero_vals != 0]
        fig, ax = plt.subplots(1, 2, figsize=(18, 6))

        ax[0].hist(latent.flatten(), bins=100, log=True, color='steelblue', alpha=0.8)
        ax[0].set_xlabel('Latent Value')
        ax[0].set_ylabel('Count (Log Scale)')
        ax[0].set_title('All Latent Values (incl. zeros)')
        ax[0].axvline(0, color='red', linestyle='--', alpha=0.5)
        hep.cms.label(ax=ax[0], data=False, label="Simulation", rlabel="")

        if len(nonzero_vals) > 0:
            ax[1].hist(nonzero_vals, bins=80, log=True, color='darkorange', alpha=0.8)
            ax[1].set_xlabel('Latent Value')
            ax[1].set_ylabel('Count (Log Scale)')
            ax[1].set_title(f'Non-Zero Latent Values (N={len(nonzero_vals):,})')
            hep.cms.label(ax=ax[1], data=False, label="Simulation", rlabel="")
        else:
            ax[1].text(0.5, 0.5, 'All values are zero', ha='center', va='center',
                       fontsize=16, transform=ax[1].transAxes)

        plt.tight_layout()
        plt.savefig('plot_latent_dist.png', dpi=150)
        print("  Saved plot_latent_dist.png")
        plt.close()

        # 2. L1 Latent Space Images — select 16 wafers with HIGHEST activity
        l1_norms = np.sum(np.abs(latent), axis=1)
        top_indices = np.argsort(l1_norms)[-16:][::-1]  # Top 16 by L1 norm

        fig, axes = plt.subplots(4, 4, figsize=(14, 14))
        fig.suptitle('L1 Latent Space (top 16 most active wafers, 4x4)', fontsize=16, y=1.01)

        vmax = np.max(np.abs(latent[top_indices])) if len(top_indices) > 0 else 1
        for i, ax in enumerate(axes.flatten()):
            if i < len(top_indices):
                idx = top_indices[i]
                img = latent[idx].reshape(4, 4)
                im = ax.imshow(img, cmap='RdBu_r', aspect='auto', vmin=-vmax, vmax=vmax)
                ax.set_title(f'Wafer {idx}\n(L1={l1_norms[idx]:.2f})', fontsize=10)
            ax.axis('off')

        fig.colorbar(im, ax=axes, shrink=0.6, label='Latent Value')
        plt.tight_layout()
        plt.savefig('plot_latent_images.png', dpi=150)
        print("  Saved plot_latent_images.png")
        sys.stdout.flush()
        plt.close()

        # 3. Latent heatmap — show activation pattern across a sample of non-zero wafers
        if n_nonzero_wafers > 0:
            nonzero_indices = np.where(nonzero_mask)[0]
            sample_size = min(200, len(nonzero_indices))
            sample_idx = np.sort(np.random.choice(nonzero_indices, sample_size, replace=False))
            sample_latent = latent[sample_idx]

            plt.figure(figsize=(12, 8))
            vabs = np.max(np.abs(sample_latent))
            plt.imshow(sample_latent.T, cmap='RdBu_r', aspect='auto', vmin=-vabs, vmax=vabs)
            plt.colorbar(label='Latent Value')
            plt.xlabel(f'Wafer Index (sampled {sample_size} non-zero wafers)')
            plt.ylabel('Latent Dimension')
            plt.title(f'Latent Activations Heatmap ({n_nonzero_wafers:,} non-zero wafers / {len(latent):,} total)')
            plt.yticks(range(16), [f'Dim {i}' for i in range(16)])
            hep.cms.label(data=False, label="Simulation", rlabel="")
            plt.tight_layout()
            plt.savefig('plot_latent_heatmap.png', dpi=150)
            print("  Saved plot_latent_heatmap.png")
            plt.close()

    except Exception as e:
        print(f"Error creating Latent Space plot: {e}")
        import traceback
        traceback.print_exc()

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
