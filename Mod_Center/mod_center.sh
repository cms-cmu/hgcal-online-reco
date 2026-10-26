#!/usr/bin/env bash
#SBATCH --job-name=mod_center
#SBATCH --gres=mps:50
#SBATCH --qos=medium
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=dylankan@andrew.cmu.edu
#SBATCH --chdir=/home/export/dylankan/hgcal_online_reco/Mod_Center
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=4G
#SBATCH --time=8:00:00
#SBATCH --output=logs/slurm_%j.log
set -euo pipefail

HOME_DIR=/home/export/dylankan/hgcal_online_reco
SCRIPT_DIR="$HOME_DIR/Mod_Center"
LOG_DIR="$SCRIPT_DIR/logs"
mkdir -p "$LOG_DIR"

TOTAL_EPOCHS=100
LATEST="$SCRIPT_DIR/checkpoint/model_latest.pth"

source "$HOME/miniconda3/etc/profile.d/conda.sh"

if [[ -f "$LATEST" ]]; then
    LAST_EPOCH=$(conda run -n detector_env python -c \
        "import torch; print(torch.load('$LATEST', map_location='cpu', weights_only=False)['epoch'])")
    echo "Found checkpoint at epoch $LAST_EPOCH"
    if (( LAST_EPOCH + 1 >= TOTAL_EPOCHS )); then
        echo "Training already complete. Nothing to do."
        exit 0
    fi
else
    echo "No checkpoint found - starting from scratch"
fi

echo "--- K-center SetTransformer ($TOTAL_EPOCHS epochs) ---"
conda run --no-capture-output -n detector_env python -u "$SCRIPT_DIR/K_center_transformer_train.py" \
    --device cuda:0 \
    --root_data_dir "$HOME_DIR/data" \
    --epochs "$TOTAL_EPOCHS" \
    --resume_from "$LATEST" \
    --batch_size 16 \
    --root_max_wafers 9250 \
    --root_max_k 95 \
    --final_test_size 5000 
