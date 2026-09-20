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
`haloscope.py` and `sim_to_fastpm/training.py` are deprecated shims until S7.

## Repository-specific code

`sim_to_fastpm/` contains domain logic for the UNIT → FastPM enrichment pipeline.
Orchestration lives in `density_field_properties.pipelines` (repo root entrypoints
under `pipelines/`).

**Phase 0 decisions and full pipeline plan** (in `phd-agents-toolkit`):

- [`notes/planes-cursor/2026-09-17-haloscope-phase0-decisions.md`](../../../../phd-agents-toolkit/notes/planes-cursor/2026-09-17-haloscope-phase0-decisions.md)
- [`notes/planes-cursor/2026-09-17-haloscope-pipeline-implementacion.md`](../../../../phd-agents-toolkit/notes/planes-cursor/2026-09-17-haloscope-pipeline-implementacion.md)

**First production config:** [`config/haloscope_run.yaml`](../../../config/haloscope_run.yaml)

## Alternative: pip dependency

To avoid vendoring the file, add under `pip:` in `environment.yml`:

```yaml
- haloscope @ git+https://github.com/computationalAstroUAM/haloscope.git@main
```

Remove `haloscope.py` and use `from haloscope import ConditionalMultiVariateGaussian` where needed.
