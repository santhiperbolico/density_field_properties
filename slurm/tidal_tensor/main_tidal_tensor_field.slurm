#!/bin/bash
#SBATCH -A 16cores
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --job-name=tidal_tensor_field
#SBATCH --error=output/tidal_tensor_field/tidal_tensor_field.log
#SBATCH --output=output/tidal_tensor_field/tidal_tensor_field.out
#SBATCH --partition=all
#SBATCH --exclude=epi
#SBATCH --time=30-00:00:00

set -euo pipefail
cd /home/arnes/santiago_arranz/density_field_properties

export OMP_NUM_THREADS=16
export PYTHONPATH=/home/arnes/santiago_arranz/density_field_properties/src
source /home/arnes/miniconda3/etc/profile.d/conda.sh
conda activate density_field_properties

DENSITY_FILE="output/density_field_cic/dm_particles_0_density"
DENSITY_INFO="output/density_field_cic/dm_particles_0_density_info.txt"
HALO_FILE="/data8/adrian/fastpm_N2048_kmax0_25/fnl0/fastpm_N1/fof/fof_0.5000/LL-0.200"
OUTPUT_PATH="output/tidal_tensor_field"
HALO_CATALOG_NAME="fastpm"

mkdir -p "${OUTPUT_PATH}"

echo "[INFO] [$(date '+%Y-%m-%d %H:%M:%S')] Starting tidal tensor and environment descriptors"

srun python scripts/main_tidal_tensor_field.py \
    --density_file "${DENSITY_FILE}" \
    --density_info "${DENSITY_INFO}" \
    --halo_file "${HALO_FILE}" \
    --halo_catalog_name "${HALO_CATALOG_NAME}" \
    --path "${OUTPUT_PATH}" \
    --read_from_path
