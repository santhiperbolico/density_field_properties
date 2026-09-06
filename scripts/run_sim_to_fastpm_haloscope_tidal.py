#!/usr/bin/env python
"""
Run SIM-to-FastPM Haloscope with T/|U| and tidal anisotropy inputs.

Quick run (same limits as notebook ``QUICK_RUN=True``):

    PYTHONPATH=src python scripts/run_sim_to_fastpm_haloscope_tidal.py --quick-run

Quick run with assembly-bias PDF:

    PYTHONPATH=src python scripts/run_sim_to_fastpm_haloscope_tidal.py \\
        --quick-run --assembly-bias

Full run with assembly-bias PDF (heavy; cluster recommended):

    PYTHONPATH=src python scripts/run_sim_to_fastpm_haloscope_tidal.py \\
        --max-sim-halos 0 --max-fastpm-halos 0 --max-descriptor-batches 0 \\
        --assembly-bias
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

from density_field_properties.haloscope.sim_to_fastpm.config import (
    OUTPUT_DIR_TIDAL,
    OUTPUT_DIR_TIDAL_SMOKE,
    default_fastpm_list_path,
    default_sim_hlist_path,
    max_descriptor_batch_files_for_run,
    max_fastpm_halos_for_run,
    max_sim_halos_for_run,
    min_bin_size_for_run,
)
from density_field_properties.haloscope.sim_to_fastpm.pipeline_tidal import (
    run_sim_to_fastpm_haloscope_tidal_pipeline,
)


def _parse_args(argv: list[str], default_sim: Path, default_fastpm: Path) -> argparse.Namespace:
    """
    Parse CLI arguments for the tidal Haloscope pipeline runner.

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
    parser = argparse.ArgumentParser(
        description="SIM (UNIT) to FastPM Haloscope enrichment with tidal inputs"
    )
    parser.add_argument(
        "--quick-run",
        action="store_true",
        help="Use SMOKE_* limits from config.py (notebook QUICK_RUN=True).",
    )
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
        "--repo-root",
        type=Path,
        default=Path.cwd(),
        help="Repository root for resolving descriptor directories.",
    )
    parser.add_argument(
        "--max-sim-halos",
        type=int,
        default=None,
        help="Max UNIT data rows to read (0 = entire file). Overrides --quick-run when set.",
    )
    parser.add_argument(
        "--max-fastpm-halos",
        type=int,
        default=None,
        help="Max FastPM halos to read (0 = entire file). Overrides --quick-run when set.",
    )
    parser.add_argument(
        "--max-descriptor-batches",
        type=int,
        default=None,
        help=(
            "Max tidal descriptor batch files per simulation "
            "(0 = all batches). Overrides --quick-run when set."
        ),
    )
    parser.add_argument(
        "--min-bin-size",
        type=int,
        default=None,
        help="Minimum halos per mass bin (10 for production). Overrides --quick-run when set.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for enriched Parquet and plots.",
    )
    parser.add_argument(
        "--skip-holdout",
        action="store_true",
        help="Skip SIM hold-out validation step.",
    )
    parser.add_argument(
        "--assembly-bias",
        action="store_true",
        help="Write assembly_bias_tidal_input.pdf after enrichment (slow; needs matter delta).",
    )
    parser.add_argument(
        "--assembly-bias-n-grid",
        type=int,
        default=128,
        help="Grid resolution for Paranjape assembly-bias diagnostic (default: 128).",
    )
    return parser.parse_args(argv)


def _cap_from_cli_or_config(
    cli_value: Optional[int],
    quick_run: bool,
    config_quick_value: Optional[int],
    config_full_value: Optional[int],
) -> Optional[int]:
    """
    Resolve a halo or batch cap from CLI flags or config defaults.

    Parameters
    ----------
    cli_value : Optional[int]
        Explicit CLI value; ``None`` means use config defaults.
    quick_run : bool
        Whether ``--quick-run`` was passed.
    config_quick_value : Optional[int]
        Smoke cap from ``config.py``.
    config_full_value : Optional[int]
        Production cap from ``config.py``.

    Returns
    -------
    Optional[int]
        Resolved cap, or ``None`` for unlimited when CLI passes ``0``.
    """
    if cli_value is not None:
        return None if cli_value == 0 else cli_value
    config_value = config_quick_value if quick_run else config_full_value
    return config_value


def _resolve_run_settings(
    args: argparse.Namespace,
) -> tuple[
    Optional[int],
    Optional[int],
    Optional[int],
    int,
    Path,
]:
    """
    Build effective run limits and output directory from parsed CLI args.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed CLI options.

    Returns
    -------
    tuple
        ``max_sim_halos``, ``max_fastpm_halos``, ``max_descriptor_batches``,
        ``min_bin_size``, ``output_dir``.
    """
    quick_run = args.quick_run
    max_sim = _cap_from_cli_or_config(
        args.max_sim_halos,
        quick_run,
        max_sim_halos_for_run(True),
        max_sim_halos_for_run(False),
    )
    max_fastpm = _cap_from_cli_or_config(
        args.max_fastpm_halos,
        quick_run,
        max_fastpm_halos_for_run(True),
        max_fastpm_halos_for_run(False),
    )
    max_descriptor_batches = _cap_from_cli_or_config(
        args.max_descriptor_batches,
        quick_run,
        max_descriptor_batch_files_for_run(True),
        max_descriptor_batch_files_for_run(False),
    )
    if args.min_bin_size is not None:
        min_bin_size = args.min_bin_size
    else:
        min_bin_size = min_bin_size_for_run(quick_run)

    if args.output_dir is not None:
        output_dir = args.output_dir
    elif quick_run:
        output_dir = OUTPUT_DIR_TIDAL_SMOKE
    else:
        output_dir = OUTPUT_DIR_TIDAL

    return max_sim, max_fastpm, max_descriptor_batches, min_bin_size, output_dir


def main(argv: list[str]) -> int:
    """
    Execute the tidal Haloscope pipeline and log the output path.

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
    max_sim, max_fastpm, max_descriptor_batches, min_bin_size, output_dir = _resolve_run_settings(
        args
    )

    logging.info(
        "Run settings: quick_run=%s max_sim=%s max_fastpm=%s "
        "max_descriptor_batches=%s min_bin_size=%s output_dir=%s",
        args.quick_run,
        max_sim,
        max_fastpm,
        max_descriptor_batches,
        min_bin_size,
        output_dir,
    )

    out_path = run_sim_to_fastpm_haloscope_tidal_pipeline(
        sim_hlist_path=args.sim_hlist,
        fastpm_list_path=args.fastpm_list,
        repo_root=args.repo_root,
        max_sim_halos=max_sim,
        max_fastpm_halos=max_fastpm,
        max_descriptor_batch_files=max_descriptor_batches,
        output_dir=output_dir,
        min_bin_size=min_bin_size,
        run_holdout_validation=not args.skip_holdout,
        run_assembly_bias_plot=args.assembly_bias,
        assembly_bias_n_grid=args.assembly_bias_n_grid,
    )
    logging.info("Enriched catalog written to %s", out_path)
    if args.assembly_bias:
        logging.info(
            "Assembly bias panel written to %s",
            output_dir / "assembly_bias_tidal_input.pdf",
        )
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    raise SystemExit(main(sys.argv[1:]))
