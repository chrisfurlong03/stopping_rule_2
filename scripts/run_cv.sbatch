#!/bin/bash
#SBATCH --job-name=onset-cv
#SBATCH --partition=general
#SBATCH --cpus-per-task=25
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=outputs/logs/%x-%j.out
# Full leave-one-year-out CV, final model, metrics, bootstrap and figures.
# Run from the repo root:  sbatch scripts/run_cv.sbatch
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
uv run stopping-rule-2 train
uv run stopping-rule-2 evaluate
