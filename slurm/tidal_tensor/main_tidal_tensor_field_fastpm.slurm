#!/bin/bash
#SBATCH -A 16cores
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --job-name=tidal_fastpm
#SBATCH --error=output/fast_pm_bigfile/tidal_tensor_field.log
#SBATCH --output=output/fast_pm_bigfile/tidal_tensor_field.out
#SBATCH --partition=all
#SBATCH --exclude=epi
#SBATCH --time=30-00:00:00

set -euo pipefail
cd /home/arnes/santiago_arranz/density_field_properties

export OMP_NUM_THREADS=16
export PYTHONPATH=/home/arnes/santiago_arranz/density_field_properties/src
source /home/arnes/miniconda3/etc/profile.d/conda.sh
conda activate density_field_properties

DENSITY_FILE="output/fast_pm_bigfile/snap_1.0000_density"
DENSITY_INFO="output/fast_pm_bigfile/snap_1.0000_density_info.txt"
HALO_FILE="/data21/users/mruiz/fastpm_MN5/fastpm_tfm/rockstar_out_pm/out_8.list"
OUTPUT_PATH="output/fast_pm_bigfile"
HALO_CATALOG_NAME="rockstar"
BATCH_SIZE=100000

mkdir -p "${OUTPUT_PATH}"

if [[ -d "${OUTPUT_PATH}/tidal_anisotropy" ]]; then
    echo "[INFO] [$(date '+%Y-%m-%d %H:%M:%S')] Removing stale tidal_anisotropy output (issue #12)"
    rm -rf "${OUTPUT_PATH}/tidal_anisotropy"
fi

echo "[INFO] [$(date '+%Y-%m-%d %H:%M:%S')] FastPM: tidal descriptors rerun (reuse tidal_tensor HDF5)"

srun python scripts/main_tidal_tensor_field.py \
    --density_file "${DENSITY_FILE}" \
    --density_info "${DENSITY_INFO}" \
    --halo_file "${HALO_FILE}" \
    --halo_catalog_name "${HALO_CATALOG_NAME}" \
    --batch_size "${BATCH_SIZE}" \
    --path "${OUTPUT_PATH}" \
    --read_from_path
