import uproot
import numpy as np
import glob
import os

def inspect_root(file_path):
    print(f"\n--- Inspecting ROOT file: {file_path} ---")
    try:
        with uproot.open(file_path) as file:
            print(f"Keys in file: {file.keys()}")
            # Assuming 'Events' tree based on README, but good to check
            if "Events;1" in file.keys() or "Events" in file.keys():
                tree = file["Events"]
                print(f"Branches in 'Events' tree:")
                for branch in tree.keys():
                    print(f"  - {branch}")
            else:
                print("No 'Events' tree found. Keys available:", file.keys())
    except Exception as e:
        print(f"Error reading ROOT file: {e}")

def inspect_npz(file_path):
    print(f"\n--- Inspecting NPZ file: {file_path} ---")
    try:
        with np.load(file_path) as data:
            print("Files in archive:")
            for key in data.files:
                print(f"  - {key}: shape={data[key].shape}, dtype={data[key].dtype}")
    except Exception as e:
        print(f"Error reading NPZ file: {e}")

def main():
    data_dir = "data"
    
    # Find ROOT files
    root_files = glob.glob(os.path.join(data_dir, "*.root"))
    if root_files:
        inspect_root(root_files[0])
    else:
        print("No ROOT files found in data directory.")

    # Find NPZ files
    npz_files = glob.glob(os.path.join(data_dir, "*.npz"))
    if npz_files:
        inspect_npz(npz_files[0])
    else:
        print("No NPZ files found in data directory.")

if __name__ == "__main__":
    main()
