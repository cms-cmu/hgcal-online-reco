import uproot
import sys

try:
    print("Imported uproot")
    path = "data/output_Phase2_HGCalL1T_Clustering_1.root"
    print(f"Opening {path}...")
    with uproot.open(path) as file:
        print("Opened successfully.")
        print(f"Keys: {file.keys()}")
except Exception as e:
    print(f"Error: {e}")
    sys.exit(1)
