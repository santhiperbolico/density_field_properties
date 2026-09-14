#!/bin/bash
#SBATCH -A 16cores
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --job-name=halo_hmf_a1
#SBATCH --error=output/halo_hmf_a1.log
#SBATCH --output=output/halo_hmf_a1.out
#SBATCH --partition=all
#SBATCH --exclude=epi
#SBATCH --time=08:00:00

set -euo pipefail
cd /home/arnes/santiago_arranz/density_field_properties

export OMP_NUM_THREADS=16
export PYTHONPATH=/home/arnes/santiago_arranz/density_field_properties/src
source /home/arnes/miniconda3/etc/profile.d/conda.sh
conda activate density_field_properties

OUTPUT_DIR="output/halo_hmf_a1"

mkdir -p "${OUTPUT_DIR}"

echo "[INFO] [$(date '+%Y-%m-%d %H:%M:%S')] Starting UNIT vs FastPM halo mass function comparison (a=1)"

srun python scripts/compare_halo_hmf.py \
    --include-subhalos \
    --output-dir "${OUTPUT_DIR}"

echo "[INFO] [$(date '+%Y-%m-%d %H:%M:%S')] Finished halo mass function comparison"
