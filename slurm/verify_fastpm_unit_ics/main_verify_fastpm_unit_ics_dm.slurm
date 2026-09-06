#!/bin/bash
#SBATCH -A 16cores
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --job-name=fastpm_unit_ics_dm
#SBATCH --error=output/fastpm_unit_ics_dm.log
#SBATCH --output=output/fastpm_unit_ics_dm.out
#SBATCH --partition=all
#SBATCH --exclude=epi
#SBATCH --time=08:00:00

set -euo pipefail
cd /home/arnes/santiago_arranz/density_field_properties

export OMP_NUM_THREADS=16
export PYTHONPATH=/home/arnes/santiago_arranz/density_field_properties/src
source /home/arnes/miniconda3/etc/profile.d/conda.sh
conda activate density_field_properties

OUTPUT_DIR="output/fastpm_unit_seed_check_dm"
N_GRID=256
DM_BATCH_SIZE=5000000

mkdir -p "${OUTPUT_DIR}"

echo "[INFO] [$(date '+%Y-%m-%d %H:%M:%S')] Starting FastPM vs UNIT IC seed check (DM r(k))"

srun python scripts/verify_fastpm_unit_ics.py \
    --tracer dm \
    --n-grid "${N_GRID}" \
    --dm-batch-size "${DM_BATCH_SIZE}" \
    --output-dir "${OUTPUT_DIR}"

echo "[INFO] [$(date '+%Y-%m-%d %H:%M:%S')] Finished DM IC seed check"
