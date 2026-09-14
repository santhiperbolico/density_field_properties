#!/bin/bash
#SBATCH -A 16cores
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --job-name=fastpm_unit_ics
#SBATCH --error=output/fastpm_unit_ics.log
#SBATCH --output=output/fastpm_unit_ics.out
#SBATCH --partition=all
#SBATCH --exclude=epi
#SBATCH --time=04:00:00

set -euo pipefail
cd /home/arnes/santiago_arranz/density_field_properties

export OMP_NUM_THREADS=16
export PYTHONPATH=/home/arnes/santiago_arranz/density_field_properties/src
source /home/arnes/miniconda3/etc/profile.d/conda.sh
conda activate density_field_properties

OUTPUT_DIR="output/fastpm_unit_seed_check"
N_HALOS=500000
N_GRID=256

mkdir -p "${OUTPUT_DIR}"

echo "[INFO] [$(date '+%Y-%m-%d %H:%M:%S')] Starting FastPM vs UNIT IC seed check (r(k))"

srun python scripts/verify_fastpm_unit_ics.py \
    --n-halos "${N_HALOS}" \
    --n-grid "${N_GRID}" \
    --output-dir "${OUTPUT_DIR}"

echo "[INFO] [$(date '+%Y-%m-%d %H:%M:%S')] Finished IC seed check"
