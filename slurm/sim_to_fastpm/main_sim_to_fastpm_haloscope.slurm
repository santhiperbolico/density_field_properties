#!/bin/bash
#SBATCH -A 16cores
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --job-name=sim_to_fastpm_haloscope
#SBATCH --error=output/sim_to_fastpm_haloscope/sim_to_fastpm_haloscope.log
#SBATCH --output=output/sim_to_fastpm_haloscope/sim_to_fastpm_haloscope.out
#SBATCH --partition=all
#SBATCH --exclude=epi
#SBATCH --time=30-00:00:00
#SBATCH --mem=128G

set -euo pipefail
cd /home/arnes/santiago_arranz/density_field_properties

export OMP_NUM_THREADS=16
export PYTHONPATH=/home/arnes/santiago_arranz/density_field_properties/src
source /home/arnes/miniconda3/etc/profile.d/conda.sh
conda activate density_field_properties

OUTPUT_PATH="output/sim_to_fastpm_haloscope"
MIN_BIN_SIZE=10

mkdir -p "${OUTPUT_PATH}"

echo "[INFO] [$(date '+%Y-%m-%d %H:%M:%S')] Starting SIM (UNIT) to FastPM Haloscope (full catalogs)"

srun python scripts/run_sim_to_fastpm_haloscope.py \
    --max-sim-halos 0 \
    --max-fastpm-halos 0 \
    --min-bin-size "${MIN_BIN_SIZE}" \
    --output-dir "${OUTPUT_PATH}"

echo "[INFO] [$(date '+%Y-%m-%d %H:%M:%S')] Finished Haloscope enrichment"
