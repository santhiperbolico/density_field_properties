# Slurm

Plantillas `sbatch` para el cluster. Todas asumen `cd` al directorio del repo y activación del entorno conda `density_field_properties`.

## Estructura

| Carpeta | Contenido |
|---------|-----------|
| `density_field/` | Construcción del campo de densidad CIC (`main_density_field_cic.py`) |
| `tidal_tensor/` | Tensor de marea y descriptores tidales: `main_tidal_tensor_field_fastpm.slurm`, `main_tidal_tensor_field_unit.slurm` |
| `pipelines/` | Pipeline completo: CIC + tensor de marea |

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
```

Tras el fix de la issue #12, regenerar descriptores tidales:

- **FastPM:** `main_tidal_tensor_field_fastpm.slurm` — reutiliza `tidal_tensor/` (`--read_from_path`), borra y reescribe `tidal_anisotropy/`.
- **UNIT:** `main_tidal_tensor_field_unit.slurm` — pipeline completo (tensor + descriptores). Descomprime `hlist_1.00000.list.bz2` en `output/unit_files/` la primera vez (~100 GB descomprimido).
