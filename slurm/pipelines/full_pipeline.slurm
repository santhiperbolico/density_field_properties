#!/bin/bash
#SBATCH -A 16cores
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --job-name=density_field_pipeline
#SBATCH --error=output/density_field_pipeline/density_field_pipeline.err
#SBATCH --output=output/density_field_pipeline/density_field_pipeline.out
#SBATCH --partition=all
#SBATCH --exclude=epi
#SBATCH --time=30-00:00:00

set -euo pipefail
cd /home/arnes/santiago_arranz/density_field_properties

export OMP_NUM_THREADS=16
export PYTHONPATH=/home/arnes/santiago_arranz/density_field_properties/src
source /home/arnes/miniconda3/etc/profile.d/conda.sh
conda activate density_field_properties

DM_PARTICLES_FILE="/path/to/dm_particles.dat"
HALO_FILE="/path/to/halo_catalog.dat"
OUTPUT_PATH="output/density_field_pipeline"
HALO_CATALOG_NAME="rockstar"

DM_BASENAME="$(basename "${DM_PARTICLES_FILE}" .dat)"
DENSITY_FILE="${OUTPUT_PATH}/${DM_BASENAME}_density"
DENSITY_INFO="${OUTPUT_PATH}/${DM_BASENAME}_density_info.txt"

mkdir -p "${OUTPUT_PATH}"

echo "[INFO] [$(date '+%Y-%m-%d %H:%M:%S')] Step 1/2: CIC density field"

srun python scripts/main_density_field_cic.py \
    --dm_particles_file "${DM_PARTICLES_FILE}" \
    --path "${OUTPUT_PATH}"

echo "[INFO] [$(date '+%Y-%m-%d %H:%M:%S')] Step 2/2: Tidal tensor and environment descriptors"

srun python scripts/main_tidal_tensor_field.py \
    --density_file "${DENSITY_FILE}" \
    --density_info "${DENSITY_INFO}" \
    --halo_file "${HALO_FILE}" \
    --halo_catalog_name "${HALO_CATALOG_NAME}" \
    --path "${OUTPUT_PATH}"
