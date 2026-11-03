#!/bin/bash
#SBATCH -A 16cores
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --job-name=mass_cal_4cat
#SBATCH --error=output/mass_calibration_four_catalogs_taurus/mass_cal_4cat.log
#SBATCH --output=output/mass_calibration_four_catalogs_taurus/mass_cal_4cat.out
#SBATCH --partition=all
#SBATCH --exclude=epi
#SBATCH --time=7-00:00:00

set -euo pipefail
cd /home/arnes/santiago_arranz/density_field_properties

export OMP_NUM_THREADS=16
export PYTHONPATH=/home/arnes/santiago_arranz/density_field_properties/src
source /home/arnes/miniconda3/etc/profile.d/conda.sh
conda activate density_field_properties

mkdir -p output/mass_calibration_four_catalogs_taurus

echo "[INFO] [$(date '+%Y-%m-%d %H:%M:%S')] Mass calibration benchmark (MN5 full box, four catalogs)"
echo "[INFO] Config: config/mass_calibration_four_catalogs_taurus.json"
echo "[INFO] Catalog paths aligned with config/hmf_four_catalogs_taurus.json"

srun python scripts/compare_mass_calibration_catalogs.py \
    --config config/mass_calibration_four_catalogs_taurus.json

echo "[INFO] [$(date '+%Y-%m-%d %H:%M:%S')] Done. See output/mass_calibration_four_catalogs_taurus/"
