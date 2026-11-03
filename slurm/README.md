# Slurm

Plantillas `sbatch` para el cluster. Todas asumen `cd` al directorio del repo y activación del entorno conda `density_field_properties`.

## Estructura

| Carpeta | Contenido |
|---------|-----------|
| `density_field/` | Construcción del campo de densidad CIC (`main_density_field_cic.py`) |
| `tidal_tensor/` | Tensor de marea y descriptores tidales: `main_tidal_tensor_field_fastpm.slurm`, `main_tidal_tensor_field_unit.slurm` |
| `sim_to_fastpm/` | Haloscope SIM→FastPM: `main_sim_to_fastpm_haloscope.slurm`, `main_sim_to_fastpm_haloscope_tidal.slurm`, `main_sim_to_fastpm_haloscope_tidal_smoke.slurm` |
| `pipelines/` | CIC + tensor de marea (`tidal_anisotropy_mn5_from_scratch.slurm`); Haloscope MN5 con descriptores ya generados (`haloscope_tidal_mn5_precomputed.slurm`) |
| `rockstar/` | Post-proceso Rockstar `find_parents` (columna PID / host halos) |
| `compare_halo_hmf/` | HMF y calibración de masas multi-catálogo (local_sample / MN5 Taurus) |

## Convenciones

- Logs `#SBATCH --error` / `--output` bajo `output/` (directorio existente), p. ej. `output/density_field_cic/density_field_cic.out`.
- Scripts Python en `scripts/`.
- Editar las variables de rutas de entrada al inicio de cada `.slurm` antes de enviar el job.
- `PYTHONPATH` apunta a `src/` para importar el paquete `density_field_properties`.

## Ejemplos

```bash
sbatch slurm/density_field/main_density_field_cic.slurm
sbatch slurm/tidal_tensor/main_tidal_tensor_field_fastpm.slurm
sbatch slurm/tidal_tensor/main_tidal_tensor_field_unit.slurm
sbatch slurm/pipelines/full_pipeline.slurm
sbatch slurm/sim_to_fastpm/main_sim_to_fastpm_haloscope_tidal_smoke.slurm
sbatch slurm/sim_to_fastpm/main_sim_to_fastpm_haloscope_tidal.slurm
sbatch slurm/rockstar/find_parents_fastpm_mn5_out8.slurm
sbatch slurm/pipelines/tidal_anisotropy_mn5_from_scratch.slurm
sbatch slurm/pipelines/haloscope_tidal_mn5_precomputed.slurm
sbatch slurm/compare_halo_hmf/plot_hmf_four_catalogs_taurus.slurm
sbatch slurm/compare_halo_hmf/compare_mass_calibration_four_catalogs_taurus.slurm
```

Salida por defecto de `find_parents_fastpm_mn5_out8.slurm`:

`output/rockstar_find_parents/fastpm_mn5_tfm/out_8_with_pid.list`

(El catálogo de entrada en `/data21/users/mruiz/...` es solo lectura.)

Tras el fix de la issue #12, regenerar descriptores tidales:

- **FastPM:** `main_tidal_tensor_field_fastpm.slurm` — reutiliza `tidal_tensor/` (`--read_from_path`), borra y reescribe `tidal_anisotropy/`.
- **UNIT:** `main_tidal_tensor_field_unit.slurm` — pipeline completo (tensor + descriptores). Descomprime `hlist_1.00000.list.bz2` en `output/unit_files/` la primera vez (~100 GB descomprimido).
