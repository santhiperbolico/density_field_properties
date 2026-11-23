#!/bin/bash
#SBATCH -A 16cores
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --job-name=tidal_mn5
#SBATCH --array=0-1
#SBATCH --error=output/tidal_mn5/tidal_mn5_%a.log
#SBATCH --output=output/tidal_mn5/tidal_mn5_%a.out
#SBATCH --partition=all
#SBATCH --exclude=epi
#SBATCH --time=30-00:00:00

set -euo pipefail
cd /home/arnes/santiago_arranz/density_field_properties

export OMP_NUM_THREADS=16
export PYTHONPATH=/home/arnes/santiago_arranz/density_field_properties/src
source /home/arnes/miniconda3/etc/profile.d/conda.sh
conda activate density_field_properties

CONFIG="config/older/tidal_anisotropy_mn5.json"
OUTPUT_PATH="output/tidal_mn5"
UNIT_WORK="${OUTPUT_PATH}/unit"

TIDAL_TARGETS=(unit fastpm)
TARGET="${TIDAL_TARGETS[$SLURM_ARRAY_TASK_ID]}"

UNIT_HLIST_BZ2="/data21/UNITSIM/fixedAmp_InvPhase_001/ROCKSTAR/outputs/hlists/hlist_1.00000.list.bz2"
UNIT_HLIST="${UNIT_WORK}/hlist_1.00000.list"
UNIT_DM_BZ2="/data21/UNITSIM/fixedAmp_InvPhase_001/DM_PARTICLES/dm_particles_0.5_128.bz2"
UNIT_DM="${UNIT_WORK}/dm_particles_0.5_128.dat"

mkdir -p "${OUTPUT_PATH}" "${UNIT_WORK}"

if [[ "${TARGET}" == "unit" ]]; then
    if [[ ! -f "${UNIT_HLIST}" ]]; then
        echo "[INFO] [$(date '+%Y-%m-%d %H:%M:%S')] Decompressing UNIT hlist (required for byte-offset batch reads)"
        bzcat "${UNIT_HLIST_BZ2}" > "${UNIT_HLIST}"
    fi

    if [[ ! -f "${UNIT_DM}" ]]; then
        echo "[INFO] [$(date '+%Y-%m-%d %H:%M:%S')] Decompressing UNIT DM particles (one-time, large file)"
        bzcat "${UNIT_DM_BZ2}" > "${UNIT_DM}"
    fi
fi

echo "[INFO] [$(date '+%Y-%m-%d %H:%M:%S')] CIC + tidal tensor + tidal anisotropy (target=${TARGET})"

srun python pipelines/run_haloscope_enrichment.py \
    --config "${CONFIG}" \
    --tidal-targets "${TARGET}"

echo "[INFO] [$(date '+%Y-%m-%d %H:%M:%S')] Finished target=${TARGET}: ${OUTPUT_PATH}/${TARGET}/tidal_anisotropy/"
