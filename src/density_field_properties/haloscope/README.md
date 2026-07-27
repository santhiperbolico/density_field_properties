# HALOSCOPE (código vendored)

[HALOSCOPE](https://github.com/computationalAstroUAM/haloscope) — Halo PropertieS having Covariance Preserved with Environment (Ramakrishnan et al., 2024).

## Código upstream

`haloscope.py` es una **copia literal** del módulo  
[haloscope.py](https://github.com/computationalAstroUAM/haloscope/blob/main/haloscope.py).  
No se reformatea con black/flake8/isort en este repo (ver `src/.pre-commit-config.yaml`).

Import en el proyecto:

```python
from density_field_properties.haloscope import ConditionalMultiVariateGaussian
```

## Contenido propio

`sim_to_fastpm/` es el pipeline UNIT → FastPM de este repositorio.

## Alternativa: dependencia pip

Para no versionar el fichero, añade en `environment.yml` bajo `pip:`:

```yaml
- haloscope @ git+https://github.com/computationalAstroUAM/haloscope.git@main
```

Elimina `haloscope.py` y usa `from haloscope import ConditionalMultiVariateGaussian` donde haga falta.
