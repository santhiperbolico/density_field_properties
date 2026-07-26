# Slurm

Plantillas `sbatch` para el cluster. Todas asumen `cd` al directorio del repo y activación del entorno conda `density_field_properties`.

## Estructura

| Carpeta | Contenido |
|---------|-----------|
| `density_field/` | Construcción del campo de densidad CIC (`main_density_field_cic.py`) |
| `tidal_tensor/` | Cálculo del tensor de marea y descriptores de entorno (`main_tidal_tensor_field.py`) |
| `pipelines/` | Pipeline completo: CIC + tensor de marea |

## Convenciones

- Logs `#SBATCH --error` / `--output` bajo `output/` (directorio existente), p. ej. `output/density_field_cic/density_field_cic.out`.
- Scripts Python en `scripts/`.
- Editar las variables de rutas de entrada al inicio de cada `.slurm` antes de enviar el job.
- `PYTHONPATH` apunta a `src/` para importar el paquete `density_field_properties`.

## Ejemplos

```bash
sbatch slurm/density_field/main_density_field_cic.slurm
sbatch slurm/tidal_tensor/main_tidal_tensor_field.slurm
sbatch slurm/pipelines/full_pipeline.slurm
```
