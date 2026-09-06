#!/bin/bash
#SBATCH -A 16cores
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --job-name=haloscope_tidal_smoke
#SBATCH --error=output/sim_to_fastpm_haloscope_tidal_smoke/sim_to_fastpm_haloscope_tidal_smoke.log
#SBATCH --output=output/sim_to_fastpm_haloscope_tidal_smoke/sim_to_fastpm_haloscope_tidal_smoke.out
#SBATCH --partition=all
#SBATCH --exclude=epi
#SBATCH --time=04:00:00

set -euo pipefail
cd /home/arnes/santiago_arranz/density_field_properties

export OMP_NUM_THREADS=8
export MPLBACKEND=Agg
export PYTHONPATH=/home/arnes/santiago_arranz/density_field_properties/src
source /home/arnes/miniconda3/etc/profile.d/conda.sh
conda activate density_field_properties

OUTPUT_PATH="output/sim_to_fastpm_haloscope_tidal_smoke"

mkdir -p "${OUTPUT_PATH}"

echo "[INFO] [$(date '+%Y-%m-%d %H:%M:%S')] Starting tidal Haloscope QUICK_RUN smoke (+ assembly bias)"

srun python scripts/run_sim_to_fastpm_haloscope_tidal.py \
    --repo-root /home/arnes/santiago_arranz/density_field_properties \
    --quick-run \
    --assembly-bias \
    --output-dir "${OUTPUT_PATH}"

echo "[INFO] [$(date '+%Y-%m-%d %H:%M:%S')] Finished tidal Haloscope smoke run"
