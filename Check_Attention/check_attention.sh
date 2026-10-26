#!/usr/bin/env bash
#SBATCH --job-name=check_attention
#SBATCH --gres=mps:25
#SBATCH --qos=light
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=dylankan@andrew.cmu.edu
#SBATCH --chdir=/home/export/dylankan/hgcal_online_reco/Check_Attention
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=4G
#SBATCH --time=1:00:00
#SBATCH --output=logs/slurm_%j.log
set -euo pipefail

HOME_DIR=/home/export/dylankan/hgcal_online_reco
SCRIPT_DIR="$HOME_DIR/Check_Attention"
LOG_DIR="$SCRIPT_DIR/logs"
mkdir -p "$LOG_DIR"

export PYTHONPATH="$HOME_DIR:${PYTHONPATH:-}"

source "$HOME/miniconda3/etc/profile.d/conda.sh"

echo "Checking Attention Type"
conda run -n detector_env python -u "$SCRIPT_DIR/check_attention.py" 