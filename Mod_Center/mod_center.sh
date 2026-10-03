#!/usr/bin/env bash
#SBATCH --job-name=mod_center
#SBATCH --gres=mps:100
#SBATCH --qos=heavy
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=dylankan@andrew.cmu.edu
#SBATCH --chdir=/home/export/dylankan/hgcal_online_reco/Mod_Center
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=4G
#SBATCH --time=1:00:00
#SBATCH --output=logs/slurm_%j.log
set -euo pipefail

HOME_DIR=/home/export/dylankan/hgcal_online_reco
SCRIPT_DIR="$HOME_DIR/Mod_Center"
LOG_DIR="$SCRIPT_DIR/logs"
mkdir -p "$LOG_DIR"

source "$HOME/miniconda3/etc/profile.d/conda.sh"

echo "--- K-center SetTransformer (100 epochs) ---"
conda run -n detector_env python -u "$SCRIPT_DIR/K_center_transformer_train.py" \
    --device cuda:0 \
    --root_data_dir "$HOME_DIR/data" \
    --epochs 100 \
    --batch_size 16 \
    --root_max_wafers 10000 \
    --root_max_k 150 \
    --final_test_size 5000 
