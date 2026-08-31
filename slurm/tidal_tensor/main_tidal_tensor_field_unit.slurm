#!/bin/bash
#SBATCH -A 16cores
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --job-name=tidal_unit
#SBATCH --error=output/unit_files/tidal_tensor_field.log
#SBATCH --output=output/unit_files/tidal_tensor_field.out
#SBATCH --partition=all
#SBATCH --exclude=epi
#SBATCH --time=30-00:00:00

set -euo pipefail
cd /home/arnes/santiago_arranz/density_field_properties

export OMP_NUM_THREADS=16
export PYTHONPATH=/home/arnes/santiago_arranz/density_field_properties/src
source /home/arnes/miniconda3/etc/profile.d/conda.sh
conda activate density_field_properties

DENSITY_FILE="output/unit_files/dm_particles_0_density"
DENSITY_INFO="output/unit_files/dm_particles_0_density_info.txt"
HALO_FILE_BZ2="/data5/UNITSIM/fixedAmp_InvPhase_001/ROCKSTAR/outputs/hlists/hlist_1.00000.list.bz2"
HALO_FILE="output/unit_files/hlist_1.00000.list"
OUTPUT_PATH="output/unit_files"
HALO_CATALOG_NAME="rockstar"
BATCH_SIZE=100000

mkdir -p "${OUTPUT_PATH}"

if [[ ! -f "${HALO_FILE}" ]]; then
    echo "[INFO] [$(date '+%Y-%m-%d %H:%M:%S')] Decompressing UNIT hlist (one-time, large file)"
    bzcat "${HALO_FILE_BZ2}" > "${HALO_FILE}"
fi

if [[ -d "${OUTPUT_PATH}/tidal_anisotropy" ]]; then
    echo "[INFO] [$(date '+%Y-%m-%d %H:%M:%S')] Removing stale tidal_anisotropy output"
    rm -rf "${OUTPUT_PATH}/tidal_anisotropy"
fi

if [[ -d "${OUTPUT_PATH}/tidal_tensor" ]]; then
    echo "[INFO] [$(date '+%Y-%m-%d %H:%M:%S')] Removing stale tidal_tensor output for full rerun"
    rm -rf "${OUTPUT_PATH}/tidal_tensor"
fi

echo "[INFO] [$(date '+%Y-%m-%d %H:%M:%S')] UNIT: full tidal tensor + environment descriptors"

srun python scripts/main_tidal_tensor_field.py \
    --density_file "${DENSITY_FILE}" \
    --density_info "${DENSITY_INFO}" \
    --halo_file "${HALO_FILE}" \
    --halo_catalog_name "${HALO_CATALOG_NAME}" \
    --batch_size "${BATCH_SIZE}" \
    --path "${OUTPUT_PATH}"
