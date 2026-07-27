# HALOSCOPE (vendored)

[HALOSCOPE](https://github.com/computationalAstroUAM/haloscope) — Halo PropertieS having Covariance Preserved with Environment (Ramakrishnan et al., 2024).

## Upstream code

`haloscope.py` is a **verbatim copy** of the upstream module  
[haloscope.py](https://github.com/computationalAstroUAM/haloscope/blob/main/haloscope.py).  
It is not reformatted with black/flake8/isort in this repo (see `src/.pre-commit-config.yaml`).

Import in this project:

```python
from density_field_properties.haloscope import ConditionalMultiVariateGaussian
```

## Repository-specific code

`sim_to_fastpm/` contains the UNIT → FastPM enrichment pipeline used here.

## Alternative: pip dependency

To avoid vendoring the file, add under `pip:` in `environment.yml`:

```yaml
- haloscope @ git+https://github.com/computationalAstroUAM/haloscope.git@main
```

Remove `haloscope.py` and use `from haloscope import ConditionalMultiVariateGaussian` where needed.
