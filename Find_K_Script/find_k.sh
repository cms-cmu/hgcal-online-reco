#!/usr/bin/env bash
#SBATCH --job-name=find_k
#SBATCH --qos=cpu_light
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=dylankan@andrew.cmu.edu
#SBATCH --chdir=/home/export/dylankan/hgcal_online_reco/Find_K_Script
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-cpu=4
#SBATCH --time=4:00:00
#SBATCH --output=Find_K_Script/logs/slurm_%j.log

set -euo pipefail
source "$HOME/miniconda3/etc/profile.d/conda.sh"

SCRIPT_DIR=/home/export/dylankan/hgcal_online_reco/Find_K_Script
conda run -n detector_env root -l -b -q "$SCRIPT_DIR/find_k.cpp" 