#!/usr/bin/env python
"""
Run SIM-to-FastPM Haloscope on a subset (smoke) or full catalogs.

Example (from repository root, smoke ~few minutes):

    PYTHONPATH=src python scripts/run_sim_to_fastpm_haloscope.py \\
        --max-sim-halos 8000 --max-fastpm-halos 8000 --min-bin-size 5

Full run (heavy; cluster recommended):

    PYTHONPATH=src python scripts/run_sim_to_fastpm_haloscope.py \\
        --max-sim-halos 0 --max-fastpm-halos 0
"""

import argparse
import logging
import sys
from pathlib import Path

from density_field_properties.haloscope.sim_to_fastpm.config import (
    default_fastpm_list_path,
    default_sim_hlist_path,
)
from density_field_properties.haloscope.sim_to_fastpm.pipeline import (
    run_sim_to_fastpm_haloscope_pipeline,
)


def _parse_args(argv: list[str], default_sim: Path, default_fastpm: Path) -> argparse.Namespace:
    """
    Parse CLI arguments for the Haloscope pipeline runner.

    Parameters
    ----------
    argv : list[str]
        Command-line arguments without the program name.
    default_sim : Path
        Default SIM hlist path when ``--sim-hlist`` is omitted.
    default_fastpm : Path
        Default FastPM list path when ``--fastpm-list`` is omitted.

    Returns
    -------
    argparse.Namespace
        Parsed options.
    """
    parser = argparse.ArgumentParser(description="SIM (UNIT) to FastPM Haloscope enrichment")
    parser.add_argument(
        "--sim-hlist",
        type=Path,
        default=default_sim,
        help="Path to SIM (UNIT) hlist catalog.",
    )
    parser.add_argument(
        "--fastpm-list",
        type=Path,
        default=default_fastpm,
        help="Path to FastPM Rockstar out_*.list catalog.",
    )
    parser.add_argument(
        "--max-sim-halos",
        type=int,
        default=8000,
        help="Max UNIT data rows to read (0 = entire file). Default: 8000 smoke subset.",
    )
    parser.add_argument(
        "--max-fastpm-halos",
        type=int,
        default=8000,
        help="Max FastPM halos to read (0 = entire file). Default: 8000.",
    )
    parser.add_argument(
        "--min-bin-size",
        type=int,
        default=5,
        help="Minimum halos per mass bin for fit/validation (use 10 for production).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/sim_to_fastpm_haloscope"),
        help="Output directory for enriched Parquet.",
    )
    parser.add_argument(
        "--skip-holdout",
        action="store_true",
        help="Skip SIM hold-out validation step.",
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
    args = _parse_args(
        argv,
        default_sim=default_sim_hlist_path(),
        default_fastpm=default_fastpm_list_path(),
    )
    max_sim = None if args.max_sim_halos == 0 else args.max_sim_halos
    max_fastpm = None if args.max_fastpm_halos == 0 else args.max_fastpm_halos

    out_path = run_sim_to_fastpm_haloscope_pipeline(
        sim_hlist_path=args.sim_hlist,
        fastpm_list_path=args.fastpm_list,
        max_sim_halos=max_sim,
        max_fastpm_halos=max_fastpm,
        output_dir=args.output_dir,
        min_bin_size=args.min_bin_size,
        run_holdout_validation=not args.skip_holdout,
    )
    logging.info("Enriched catalog written to %s", out_path)
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    raise SystemExit(main(sys.argv[1:]))
