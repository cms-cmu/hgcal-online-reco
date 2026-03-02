#!/bin/bash

#SBATCH --job-name=download_data
#SBATCH --output=download_%j.log
#SBATCH --time=24:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=leoyao@andrew.cmu.edu
#SBATCH --cpus-per-task=8
#SBATCH --mem=4G

set -e

# Check for required commands
command -v parallel >/dev/null 2>&1 || { echo >&2 "I require parallel but it's not installed.  Aborting."; exit 1; }
command -v wget >/dev/null 2>&1 || { echo >&2 "I require wget but it's not installed.  Aborting."; exit 1; }

# Set default number of jobs if SLURM_CPUS_PER_TASK is not set
JOBS=${SLURM_CPUS_PER_TASK:-1}
echo "Using $JOBS parallel jobs."

mkdir -p data
cd data
seq 1 47 | parallel -j "$JOBS" --halt now,fail=1 "wget -qc https://cernbox.cern.ch/remote.php/dav/public-files/bE1UoOQJry001ih/data/output_Phase2_HGCalL1T_Clustering_{}_latent.npz; wget -qc https://cernbox.cern.ch/remote.php/dav/public-files/bE1UoOQJry001ih/data/output_Phase2_HGCalL1T_Clustering_{}.root"
cd -

mkdir -p encoders
cd encoders
seq 2 5 | parallel -j "$JOBS" --halt now,fail=1 wget -qc https://cernbox.cern.ch/remote.php/dav/public-files/bE1UoOQJry001ih/encoders/encoder_model_NoBiasModel_elink_{}.hdf5
cd -

parallel -j "$JOBS" --halt now,fail=1 wget -qc ::: \
    https://cernbox.cern.ch/remote.php/dav/public-files/bE1UoOQJry001ih/encode_wafers.py \
    https://cernbox.cern.ch/remote.php/dav/public-files/bE1UoOQJry001ih/environment.yml \
    https://cernbox.cern.ch/remote.php/dav/public-files/bE1UoOQJry001ih/load_events.py \
    https://cernbox.cern.ch/remote.php/dav/public-files/bE1UoOQJry001ih/README.md