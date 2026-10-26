#!/bin/bash
#SBATCH -A 16cores
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --job-name=hmf_4cat_taurus
#SBATCH --error=output/hmf_four_catalogs_taurus/hmf_4cat_taurus.log
#SBATCH --output=output/hmf_four_catalogs_taurus/hmf_4cat_taurus.out
#SBATCH --partition=all
#SBATCH --exclude=epi
#SBATCH --time=4:00:00

set -euo pipefail
cd /home/arnes/santiago_arranz/density_field_properties

export OMP_NUM_THREADS=8
export PYTHONPATH=/home/arnes/santiago_arranz/density_field_properties/src
source /home/arnes/miniconda3/etc/profile.d/conda.sh
conda activate density_field_properties

mkdir -p output/hmf_four_catalogs_taurus

echo "[INFO] [$(date '+%Y-%m-%d %H:%M:%S')] Four-catalog HMF (taurus_full /data21)"

srun python scripts/plot_hmf_local_sample_four_catalogs.py \
    --config config/hmf_four_catalogs_taurus.json

echo "[INFO] [$(date '+%Y-%m-%d %H:%M:%S')] Done. See output/hmf_four_catalogs_taurus/"
