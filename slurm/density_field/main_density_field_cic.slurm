#!/bin/bash
#SBATCH -A 16cores
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --job-name=density_field_cic
#SBATCH --error=output/density_field_cic/density_field_cic.log
#SBATCH --output=output/density_field_cic/density_field_cic.out
#SBATCH --partition=all
#SBATCH --exclude=epi
#SBATCH --time=30-00:00:00

set -euo pipefail
cd /home/arnes/santiago_arranz/density_field_properties

export OMP_NUM_THREADS=16
export PYTHONPATH=/home/arnes/santiago_arranz/density_field_properties/src
source /home/arnes/miniconda3/etc/profile.d/conda.sh
conda activate density_field_properties

DM_PARTICLES_FILE="/home/arnes/santiago_arranz/density_field_properties/output/data/dm_particles_0.5_128.dat"
OUTPUT_PATH="output/density_field_cic"

mkdir -p "${OUTPUT_PATH}"

echo "[INFO] [$(date '+%Y-%m-%d %H:%M:%S')] Starting CIC density field calculation"

srun python scripts/main_density_field_cic.py \
    --dm_particles_file "${DM_PARTICLES_FILE}" \
    --path "${OUTPUT_PATH}"
