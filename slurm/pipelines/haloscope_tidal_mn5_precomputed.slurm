#!/bin/bash
#SBATCH -A 16cores
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --job-name=haloscope_mn5
#SBATCH --error=output/tidal_mn5/haloscope/haloscope_mn5.log
#SBATCH --output=output/tidal_mn5/haloscope/haloscope_mn5.out
#SBATCH --partition=all
#SBATCH --exclude=epi
#SBATCH --time=30-00:00:00

set -euo pipefail
cd /home/arnes/santiago_arranz/density_field_properties

export OMP_NUM_THREADS=16
export MPLBACKEND=Agg
export PYTHONPATH=/home/arnes/santiago_arranz/density_field_properties/src
source /home/arnes/miniconda3/etc/profile.d/conda.sh
conda activate density_field_properties

CONFIG="config/older/tidal_anisotropy_mn5.json"
OUTPUT_PATH="output/tidal_mn5/haloscope"
UNIT_WORK="output/tidal_mn5/unit"
UNIT_HLIST_BZ2="/data21/UNITSIM/fixedAmp_InvPhase_001/ROCKSTAR/outputs/hlists/hlist_1.00000.list.bz2"
UNIT_HLIST="${UNIT_WORK}/hlist_1.00000.list"
FASTPM_LIST="output/rockstar_find_parents/fastpm_mn5_tfm/out_8_with_pid.list"
UNIT_DESCRIPTORS="${UNIT_WORK}/tidal_anisotropy"
FASTPM_DESCRIPTORS="output/tidal_mn5/fastpm/tidal_anisotropy"

mkdir -p "${OUTPUT_PATH}" "${UNIT_WORK}"

if [[ ! -f "${UNIT_HLIST}" ]]; then
    echo "[INFO] [$(date '+%Y-%m-%d %H:%M:%S')] Decompressing UNIT hlist into ${UNIT_HLIST}"
    bzcat "${UNIT_HLIST_BZ2}" > "${UNIT_HLIST}"
fi

for required_path in \
    "${FASTPM_LIST}" \
    "${UNIT_DESCRIPTORS}" \
    "${FASTPM_DESCRIPTORS}"; do
    if [[ ! -e "${required_path}" ]]; then
        echo "[ERROR] Missing prerequisite: ${required_path}" >&2
        exit 1
    fi
done

echo "[INFO] [$(date '+%Y-%m-%d %H:%M:%S')] Haloscope preprocess + enrichment (precomputed tidal descriptors)"
echo "[INFO] Config: ${CONFIG}"

srun python pipelines/run_haloscope_enrichment.py --config "${CONFIG}"

echo "[INFO] [$(date '+%Y-%m-%d %H:%M:%S')] Done. Enriched catalog: ${OUTPUT_PATH}/fastpm_out_8_haloscope_tidal_enriched.parquet"
