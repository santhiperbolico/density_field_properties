#!/bin/bash
#SBATCH -A 16cores
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --job-name=halo_hmf_central
#SBATCH --error=output/halo_hmf_a1_central_min20mp.log
#SBATCH --output=output/halo_hmf_a1_central_min20mp.out
#SBATCH --partition=all
#SBATCH --exclude=epi
#SBATCH --time=16:00:00

set -euo pipefail
cd /home/arnes/santiago_arranz/density_field_properties

export OMP_NUM_THREADS=16
export PYTHONPATH=/home/arnes/santiago_arranz/density_field_properties/src
source /home/arnes/miniconda3/etc/profile.d/conda.sh
conda activate density_field_properties

OUTPUT_DIR="output/halo_hmf_a1_central_min20mp"
MIN_M200B_TIMES_MP=20

mkdir -p "${OUTPUT_DIR}"

echo "[INFO] [$(date '+%Y-%m-%d %H:%M:%S')] Starting HMF comparison (centrals, M > ${MIN_M200B_TIMES_MP} mp)"

srun python scripts/compare_halo_hmf.py \
    --min-m200b-times-mp "${MIN_M200B_TIMES_MP}" \
    --output-dir "${OUTPUT_DIR}"

echo "[INFO] [$(date '+%Y-%m-%d %H:%M:%S')] Finished HMF comparison (centrals, min mass cut)"
