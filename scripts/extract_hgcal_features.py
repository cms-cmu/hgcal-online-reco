# Needed packages to run minimal example:
# pip install uproot awkward numpy

import uproot
import awkward as ak

filename = "output_Phase2_HGCalL1T_Clustering_1.root"

f = uproot.open(filename)

f.keys()
print("Keys in the ROOT file:")
print(f.keys())  # Output:

print("Available branches in Events:")
print(f["Events"].keys())

# Example: Accessing a specific branch
branch_name = "CaloPart_eta"

print(f"Accessing branch: {branch_name}")

print(f["Events"][branch_name])
print("Arrays in the branch:")
print(f["Events"][branch_name].arrays())
print("Fields in the arrays:")
print(f["Events"][branch_name].arrays().fields)
print("Data in the branch:")
print(f["Events"][branch_name].arrays()[branch_name])
print("Type of the data:")
print(type(f["Events"][branch_name].arrays()[branch_name]))

if type(f["Events"][branch_name].arrays()[branch_name]) == ak.highlevel.Array:
    print("The branch data is an Awkward Array.")

# Use awkward functions to manipulate or analyze the data as needed
