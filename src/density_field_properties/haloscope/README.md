# HALOSCOPE (vendored)

[HALOSCOPE](https://github.com/computationalAstroUAM/haloscope) — Halo PropertieS having Covariance Preserved with Environment (Ramakrishnan et al., 2024).

## Upstream code

`model.py` is a **verbatim copy** of the upstream module
[haloscope.py](https://github.com/computationalAstroUAM/haloscope/blob/main/haloscope.py).
It is not reformatted with black/flake8/isort in this repo (see `src/.pre-commit-config.yaml`).

Import in this project:

```python
from density_field_properties.haloscope import ConditionalMultiVariateGaussian
from density_field_properties.haloscope import fit_models, predict_models
```

Canonical modules: `model.py`, `bins.py`, `training.py`, `predict.py`.

Pipeline orchestration lives in `density_field_properties.pipelines`; validation plots in
`density_field_properties.validation`.

## Alternative: pip dependency

To avoid vendoring the file, add under `pip:` in `environment.yml`:

```yaml
- haloscope @ git+https://github.com/computationalAstroUAM/haloscope.git@main
```

Remove `model.py` and use `from haloscope import ConditionalMultiVariateGaussian` where needed.
