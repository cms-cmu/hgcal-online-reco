import uproot
import awkward as ak

try:
    with uproot.open("data/output_Phase2_HGCalL1T_Clustering_1.root") as file:
        tree = file["Events"]
        print("Opening tree...")
        
        print("Reading MergedSimCluster_eta...")
        eta = tree["MergedSimCluster_eta"].array(library="ak")
        print(f"Eta Type: {type(eta)}")
        print(f"Eta: {eta}")

        print("Reading MergedSimCluster_phi...")
        phi = tree["MergedSimCluster_phi"].array(library="ak")
        print(f"Phi Type: {type(phi)}")
        print(f"Phi: {phi}")

except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
