#!/usr/bin/env python
"""
Run Haloscope SIM-to-FastPM enrichment from a JSON configuration file.

Example (env-only smoke, from repository root):

    PYTHONPATH=src python pipelines/run_haloscope_enrichment.py \\
        --config config/haloscope_run_env_smoke.json

Tidal smoke preset:

    PYTHONPATH=src python pipelines/run_haloscope_enrichment.py \\
        --config config/haloscope_run_tidal_smoke.json

Tidal smoke with assembly-bias PDF:

    PYTHONPATH=src python pipelines/run_haloscope_enrichment.py \\
        --config config/haloscope_run_tidal_smoke_assembly_bias.json
"""

import argparse
import json
import logging
import sys
from pathlib import Path

from density_field_properties.pipelines.config import (
    ASSEMBLY_BIAS_TIDAL_PDF_NAME,
    DEFAULT_ENV_PRODUCTION_CONFIG,
    DEFAULT_ENV_SMOKE_CONFIG,
    load_haloscope_enrichment_config,
)
from density_field_properties.pipelines.haloscope_enrichment import (
    run_haloscope_enrichment_pipeline,
)


def _parse_args(argv: list[str]) -> argparse.Namespace:
    """
    Parse CLI arguments for the Haloscope enrichment pipeline.

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
    Execute the Haloscope enrichment pipeline and log the output path.

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
        "Run settings: run_name=%s config=%s sim_hlist=%s fastpm_list=%s "
        "max_sim_halos=%s max_fastpm_halos=%s max_descriptor_batches=%s "
        "min_bin_size=%s output_dir=%s input_features=%s assembly_bias=%s",
        run_name,
        config_path,
        config.sim_hlist_path,
        config.fastpm_list_path,
        config.max_sim_halos,
        config.max_fastpm_halos,
        config.max_descriptor_batch_files,
        config.min_bin_size,
        config.output_dir,
        config.input_features,
        config.run_assembly_bias_plot,
    )

    out_path = run_haloscope_enrichment_pipeline(config)
    logging.info("Enriched catalog written to %s", out_path)
    if config.run_assembly_bias_plot:
        logging.info(
            "Assembly bias panel written to %s",
            config.output_dir / ASSEMBLY_BIAS_TIDAL_PDF_NAME,
        )
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    raise SystemExit(main(sys.argv[1:]))
