#!/usr/bin/env python
"""
Run SIM-to-FastPM Haloscope on a subset (smoke) or full catalogs.

Example (from repository root, smoke ~few minutes):

    PYTHONPATH=src python scripts/run_sim_to_fastpm_haloscope.py \\
        --config config/haloscope_run_env_smoke.json

Full run (heavy; cluster recommended):

    PYTHONPATH=src python scripts/run_sim_to_fastpm_haloscope.py \\
        --config config/haloscope_run_env_production.json
"""

import argparse
import json
import logging
import sys
from pathlib import Path

from density_field_properties.pipelines.config import (
    DEFAULT_ENV_PRODUCTION_CONFIG,
    DEFAULT_ENV_SMOKE_CONFIG,
    load_haloscope_enrichment_config,
)
from density_field_properties.pipelines.haloscope_pipeline import run_haloscope_pipeline


def _parse_args(argv: list[str]) -> argparse.Namespace:
    """
    Parse CLI arguments for the Haloscope pipeline runner.

    Parameters
    ----------
    argv : list[str]
        Command-line arguments without the program name.

    Returns
    -------
    argparse.Namespace
        Parsed options.
    """
    parser = argparse.ArgumentParser(description="SIM (UNIT) to FastPM Haloscope enrichment")
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_ENV_SMOKE_CONFIG,
        help=(
            "JSON run configuration "
            f"(default: {DEFAULT_ENV_SMOKE_CONFIG}; production: {DEFAULT_ENV_PRODUCTION_CONFIG})."
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    """
    Execute the Haloscope pipeline and log the output path.

    Parameters
    ----------
    argv : list[str]
        Command-line arguments without the program name.

    Returns
    -------
    int
        Process exit code (0 on success).
    """
    args = _parse_args(argv)
    config_path = Path(args.config)
    config = load_haloscope_enrichment_config(config_path)

    with config_path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    run_name = payload.get("run_name", config_path.stem)

    logging.info(
        "Run settings: run_name=%s config=%s max_sim_halos=%s max_fastpm_halos=%s "
        "min_bin_size=%s output_dir=%s",
        run_name,
        config_path,
        config.max_sim_halos,
        config.max_fastpm_halos,
        config.min_bin_size,
        config.output_dir,
    )

    out_path = run_haloscope_pipeline(config)
    logging.info("Enriched catalog written to %s", out_path)
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    raise SystemExit(main(sys.argv[1:]))
