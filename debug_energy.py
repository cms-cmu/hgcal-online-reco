import uproot
import awkward as ak
import numpy as np

try:
    print("Opening ROOT file...")
    with uproot.open("data/output_Phase2_HGCalL1T_Clustering_1.root") as file:
        tree = file["Events"]
        
        print("Reading MergedSimCluster_sumHitEnergy...")
        energy_ak = tree["MergedSimCluster_sumHitEnergy"].array(library="ak")
        
        print("Flattening energy...")
        energy = ak.to_numpy(ak.flatten(energy_ak))
        
        print(f"Energy Shape: {energy.shape}")
        print(f"Min: {np.min(energy)}, Max: {np.max(energy)}, Mean: {np.mean(energy)}")
        print(f"Number of zeros: {np.sum(energy == 0)}")
        print(f"Number of negative values: {np.sum(energy < 0)}")

except Exception as e:
    print(f"Error: {e}")
