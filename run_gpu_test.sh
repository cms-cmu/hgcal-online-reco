#!/usr/bin/env bash
#SBATCH --job-name=transform_reco
#SBATCH --gres=gpu:1
#SBATCH --qos=light
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=leoyao@andrew.cmu.edu
#SBATCH --chdir=/home/export/leoyao/transform_reco
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=logs/slurm_%j.log
set -euo pipefail

SCRIPT_DIR=/home/export/leoyao/transform_reco
LOG_DIR="$SCRIPT_DIR/logs"
mkdir -p "$LOG_DIR"

source "$HOME/miniconda3/etc/profile.d/conda.sh"

echo "=== GPU test started: $(date) ==="
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
echo ""

echo "--- K-center SetTransformer (15 epochs) ---"
conda run -n detector_env python -u "$SCRIPT_DIR/K_center_transformer_train.py" \
    --device cuda:0 \
    --root_data_dir "$SCRIPT_DIR/data" \
    --epochs 15 \
    --batch_size 16 \
    --root_max_wafers 600 \
    --root_max_k 10 \
    --final_test_size 5000 \
    2>&1 | tee "$LOG_DIR/kcenter_$(date +%Y%m%d_%H%M%S).log"

echo ""
echo "--- Offset GPTEncoderModel (12 epochs) ---"
conda run -n detector_env python -u "$SCRIPT_DIR/offset_transformer_train.py" \
    --device cuda:0 \
    --root_data_dir "$SCRIPT_DIR/data" \
    --epochs 12 \
    --batch_size 16 \
    --root_max_wafers 600 \
    --root_max_k 10 \
    --final_test_size 15000 \
    2>&1 | tee "$LOG_DIR/offset_$(date +%Y%m%d_%H%M%S).log"

echo ""
echo "=== GPU test finished: $(date) ==="
