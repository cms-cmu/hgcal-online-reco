# CERN Data Analysis Environment

## Environment Setup
A micromamba environment named `cern_analysis` has been created with all necessary packages.

### Activate:
```bash
micromamba activate cern_analysis
```

### Installed packages:
- **uproot** (≥5.0) & **awkward** (≥2.0) - Reading ROOT files
- **aiohttp** & **requests** - Remote file streaming
- **vector**, **hist**, **mplhep** - Particle physics analysis tools
- **numpy**, **pandas**, **matplotlib** - Standard data analysis

## Usage

### 1. Download Data locally (Optional)
You can download the dataset using the provided shell script, which supports parallel downloads and is compatible with Slurm.

**Run locally:**
```bash
./download.sh
```

**Submit to Slurm:**
```bash
sbatch download.sh
```
*Note: `data/output_Phase2_HGCalL1T_Clustering_1.root` is already downloaded.*

### 2. In Your Own Analysis Scripts
To stream data remotely using `uproot`, simply use the URL:

```python
import uproot

# Remote URL
url = "https://cernbox.cern.ch/remote.php/dav/public-files/bE1UoOQJry001ih/data/output_Phase2_HGCalL1T_Clustering_6.root"

# Open remote file (timeout is recommended)
file = uproot.open(url, timeout=30)

# Access data (lazy loading only reads what you need!)
tree = file["Events"]
arrays = tree.arrays(["nCaloPart", "CaloPart_energy"], entry_stop=100)
```

## Available Remote Files
Base URL: `https://cernbox.cern.ch/remote.php/dav/public-files/bE1UoOQJry001ih/data/`

- `output_Phase2_HGCalL1T_Clustering_1.root` (608 MB) [Local Copy Available]
- `output_Phase2_HGCalL1T_Clustering_6.root` (614 MB)
- `output_Phase2_HGCalL1T_Clustering_12.root` (618 MB)
- `output_Phase2_HGCalL1T_Clustering_26.root` (622 MB)
- `output_Phase2_HGCalL1T_Clustering_38.root` (613 MB)
- Plus several `.npz` files...
ster```: The *simulated* HGCAL clusters that we want to reconstruct using the wafer inputs. The ROOT file stores many physical quantities of the cluster, but the most important target variables to reconstruct are:
  - ```charge```: Total electric charge of the cluster.
  - ```nHits```: Number of wafer-level hits (energy deposits) associated to the cluster.
  - ```pdgId```: Particle-Data-Group ID, an integer indicating the type of particle that initiated the cluster shower.
  - ```eta```: Pseudorapidity (geometric coordinate) of the cluster
  - ```phi```: Azimuthal angle (geometric coordinate) of the cluster
  - ```mass```: Mass of the cluster
  - ```pt```: Transverse momentum of the cluster
  - ```sumHitEnergy```: Sum of simulated hit energies associated to the cluster.


### NPZ files

For each ROOT file, there is a corresponding NPZ file with similar naming (eg. ```_1.root``` --> ```_1_latent.npz```). That NPZ file contains the latent space vectors for all wafers in all 1000 events, produced by the auto-encoder models. These latent space vectors constitute the input to the cluster reconstruction.

The NPZ files contain the following arrays:
- ```latent``` - Shape: (N_total_wafers, 16). The 16-dimensional latent space representation for each wafer. Each row is one wafer.
- ```conditions``` - Shape: (N_total_wafers, 8). The 8 conditional features used for the auto-encoders. Together with the latenst space representations, these can be used for cluster reconstructions.
- ```event_index``` - Shape: (N_total_wafers,). Integer array tracking collision event each wafer belongs to, in order to use the corresponding event from the ROOT files.
- ```elink_id```- Shape: (N_total_wafers,). Integer array indicating which encoder was used [2, 3, 4, or 5] for encoding the wafer.

**Disclaimer**: Some wafers are encoded by multiple encoder models. As a result, the NPZ files can multiple latent space vectors corresponding to the same wafer. This will be addressed soon!

## Loading events

To access both the inputs (latent space vectors in the NPZ files) and the targets (MergedSimClusters in the ROOT files), we need to load the collections in both files consistently. An example with verbose output is provided:
```
python load_events.py
```


