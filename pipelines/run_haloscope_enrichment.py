#!/usr/bin/env python
"""
Run Haloscope SIM-to-FastPM enrichment.

Example (env-only smoke, from repository root):

    PYTHONPATH=src python pipelines/run_haloscope_enrichment.py \\
        --max-sim-halos 8000 --max-fastpm-halos 8000 --min-bin-size 5

Tidal INPUT features:

    PYTHONPATH=src python pipelines/run_haloscope_enrichment.py --tidal-preset --quick-run

Tidal preset with assembly-bias PDF:

    PYTHONPATH=src python pipelines/run_haloscope_enrichment.py \\
        --tidal-preset --quick-run --assembly-bias
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

from density_field_properties.haloscope.sim_to_fastpm.config import (
    ENRICHED_PARQUET_NAME,
    ENRICHED_TIDAL_PARQUET_NAME,
    OUTPUT_DIR,
    OUTPUT_DIR_TIDAL,
    OUTPUT_DIR_TIDAL_SMOKE,
    TIDAL_INPUT_FEATURES,
    default_fastpm_list_path,
    default_sim_hlist_path,
    max_descriptor_batch_files_for_run,
    max_fastpm_halos_for_run,
    max_sim_halos_for_run,
    min_bin_size_for_run,
)
from density_field_properties.pipelines.haloscope_enrichment import (
    run_haloscope_enrichment_pipeline,
)


def _parse_args(argv: list[str], default_sim: Path, default_fastpm: Path) -> argparse.Namespace:
    """
    Parse CLI arguments for the Haloscope enrichment pipeline.

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
        "--quick-run",
        action="store_true",
        help="Use SMOKE_* limits from config.py (notebook QUICK_RUN=True).",
    )
    parser.add_argument(
        "--tidal-preset",
        action="store_true",
        help="Use TIDAL_INPUT_FEATURES and tidal output defaults.",
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
        "--min-bin-size",
        type=int,
        default=None,
        help="Minimum halos per mass bin for fit/validation (10 for production).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for enriched Parquet.",
    )
    parser.add_argument(
        "--skip-holdout",
        action="store_true",
        help="Skip SIM hold-out validation step.",
    )
    parser.add_argument(
        "--input-features",
        nargs="+",
        default=None,
        help="Haloscope INPUT columns to attach (default: env, or tidal preset).",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path.cwd(),
        help="Repository root for resolving relative tidal descriptor directories.",
    )
    parser.add_argument(
        "--unit-descriptors-dir",
        type=Path,
        default=None,
        help="UNIT tidal descriptor directory (required for tidal_anisotropy).",
    )
    parser.add_argument(
        "--fastpm-descriptors-dir",
        type=Path,
        default=None,
        help="FastPM tidal descriptor directory (required for tidal_anisotropy).",
    )
    parser.add_argument(
        "--max-descriptor-batches",
        type=int,
        default=None,
        help="Max tidal descriptor batch files per simulation (0 = all batches).",
    )
    parser.add_argument(
        "--tidal-n-grid",
        type=int,
        default=512,
        help="Grid resolution used for tidal descriptors.",
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
    tuple[str, ...],
    str,
]:
    """
    Build effective run limits and output settings from parsed CLI args.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed CLI options.

    Returns
    -------
    tuple
        ``max_sim_halos``, ``max_fastpm_halos``, ``max_descriptor_batches``,
        ``min_bin_size``, ``output_dir``, ``input_features``, ``parquet_name``.
    """
    quick_run = args.quick_run
    tidal_preset = args.tidal_preset
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
    elif quick_run or tidal_preset:
        min_bin_size = min_bin_size_for_run(quick_run)
    else:
        min_bin_size = 5

    if args.output_dir is not None:
        output_dir = args.output_dir
    elif tidal_preset and quick_run:
        output_dir = OUTPUT_DIR_TIDAL_SMOKE
    elif tidal_preset:
        output_dir = OUTPUT_DIR_TIDAL
    else:
        output_dir = OUTPUT_DIR

    if args.input_features is not None:
        input_features = tuple(args.input_features)
    elif tidal_preset:
        input_features = TIDAL_INPUT_FEATURES
    else:
        input_features = None

    parquet_name = ENRICHED_TIDAL_PARQUET_NAME if tidal_preset else ENRICHED_PARQUET_NAME
    return (
        max_sim,
        max_fastpm,
        max_descriptor_batches,
        min_bin_size,
        output_dir,
        input_features,
        parquet_name,
    )


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
    args = _parse_args(
        argv,
        default_sim=default_sim_hlist_path(),
        default_fastpm=default_fastpm_list_path(),
    )
    (
        max_sim,
        max_fastpm,
        max_descriptor_batches,
        min_bin_size,
        output_dir,
        input_features,
        parquet_name,
    ) = _resolve_run_settings(args)

    logging.info(
        "Run settings: quick_run=%s tidal_preset=%s max_sim=%s max_fastpm=%s "
        "max_descriptor_batches=%s min_bin_size=%s output_dir=%s input_features=%s",
        args.quick_run,
        args.tidal_preset,
        max_sim,
        max_fastpm,
        max_descriptor_batches,
        min_bin_size,
        output_dir,
        input_features,
    )

    out_path = run_haloscope_enrichment_pipeline(
        sim_hlist_path=args.sim_hlist,
        fastpm_list_path=args.fastpm_list,
        max_sim_halos=max_sim,
        max_fastpm_halos=max_fastpm,
        output_dir=output_dir,
        min_bin_size=min_bin_size,
        run_holdout_validation=not args.skip_holdout,
        input_features=input_features,
        repo_root=args.repo_root,
        unit_descriptors_dir=args.unit_descriptors_dir,
        fastpm_descriptors_dir=args.fastpm_descriptors_dir,
        tidal_n_grid=args.tidal_n_grid,
        max_descriptor_batch_files=max_descriptor_batches,
        enriched_parquet_name=parquet_name,
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
