#!/bin/bash
#SBATCH -A 16cores
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --job-name=haloscope_pipe_mn5
#SBATCH --error=output/tidal_mn5/haloscope/haloscope_pipeline_mn5.log
#SBATCH --output=output/tidal_mn5/haloscope/haloscope_pipeline_mn5.out
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

CONFIG="config/pipeline_haloscope_mn5_taurus.json"

echo "[INFO] [$(date '+%Y-%m-%d %H:%M:%S')] Haloscope pipeline (${CONFIG})"
srun python pipelines/run_haloscope_enrichment.py --config "${CONFIG}"

echo "[INFO] [$(date '+%Y-%m-%d %H:%M:%S')] HMF comparison (${CONFIG})"
srun python scripts/compare_mass_calibration_catalogs.py --config "${CONFIG}"

echo "[INFO] [$(date '+%Y-%m-%d %H:%M:%S')] Done."
