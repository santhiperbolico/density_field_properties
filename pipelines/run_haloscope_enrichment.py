#!/usr/bin/env python3
"""
Run Haloscope SIM-to-FastPM enrichment from a JSON configuration file.

Example (from repository root):

    PYTHONPATH=src python3 pipelines/run_haloscope_enrichment.py \\
        --config config/pipeline_haloscope_template.json
"""

import argparse
import logging
import os
import sys
from pathlib import Path

from density_field_properties.pipelines.config import (
    DEFAULT_PIPELINE_CONFIG,
    load_haloscope_enrichment_config,
)
from density_field_properties.pipelines.haloscope_pipeline import run_haloscope_pipeline


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
        default=DEFAULT_PIPELINE_CONFIG,
        help=f"JSON run configuration (default: {DEFAULT_PIPELINE_CONFIG}).",
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

    pipeline = config.pipeline
    logging.info(
        "Run settings: run_name=%s config=%s sim_hlist=%s fastpm_list=%s "
        "output_dir=%s input_features=%s env_features=%s preprocessing=%s "
        "haloscope=%s validation=%s",
        config.run_name,
        config_path,
        config.sim_hlist_path,
        config.fastpm_list_path,
        config.output_dir,
        config.input_features,
        pipeline.environment_features.enabled,
        pipeline.preprocessing.enabled,
        pipeline.haloscope.enabled,
        pipeline.validation.enabled,
    )

    state = run_haloscope_pipeline(config)
    if state.preprocess_paths is not None:
        hr_path, lr_path = state.preprocess_paths
        logging.info("Preprocessed tables written to %s and %s", hr_path, lr_path)
    if state.enrichment_result is not None:
        logging.info("Enrichment output: %s", state.enrichment_result)
    return 0


if __name__ == "__main__":
    root = logging.getLogger()
    root.setLevel(os.environ.get("LOGLEVEL", "INFO"))
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] [%(asctime)s] %(message)s")
    raise SystemExit(main(sys.argv[1:]))
