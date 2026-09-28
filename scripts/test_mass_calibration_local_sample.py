#!/usr/bin/env python
"""
Compare the three LR mass-calibration methods on local_sample catalogs.

By default runs ``apply_mass_calibration`` on loaded HR/LR tables (no ``env``
KD-tree), which avoids periodic-box issues on non-recentered local_sample
coordinates. Use ``--with-haloscope-features`` to run the full feature-table
path (requires coordinates in ``[0, box_size)``).

Example (repository root):

    PYTHONPATH=src python scripts/test_mass_calibration_local_sample.py

    PYTHONPATH=src python scripts/test_mass_calibration_local_sample.py \\
        --config config/haloscope_run_local_sample_mass_calibration.json \\
        --with-haloscope-features
"""

import argparse
import json
import logging
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from density_field_properties.pipelines.config import (
    load_haloscope_enrichment_config,
    resolve_fastpm_boxsize_mpc_h,
)
from density_field_properties.pipelines.haloscope_enrichment import build_haloscope_feature_tables
from density_field_properties.preprocessing.catalog_loaders import (
    load_fastpm_target_catalog,
    load_unit_sim_training_catalog,
)
from density_field_properties.preprocessing.mass_calibration import apply_mass_calibration
from density_field_properties.preprocessing.mass_calibration_config import (
    MASS_CALIBRATION_METHOD_ABUNDANCE,
    MASS_CALIBRATION_METHOD_MATCHING_1TO1,
    MASS_CALIBRATION_METHOD_MATCHING_ML,
)

DEFAULT_CONFIG = Path("config/haloscope_run_local_sample_mass_calibration.json")
CALIBRATION_METHODS = (
    MASS_CALIBRATION_METHOD_ABUNDANCE,
    MASS_CALIBRATION_METHOD_MATCHING_1TO1,
    MASS_CALIBRATION_METHOD_MATCHING_ML,
)
SUMMARY_FILENAME = "mass_calibration_methods_summary.json"


def _parse_args(argv: list[str]) -> argparse.Namespace:
    """
    Parse CLI arguments for the local_sample mass-calibration comparison.

    Parameters
    ----------
    argv : list[str]
        Command-line arguments without the program name.

    Returns
    -------
    argparse.Namespace
        Parsed options.
    """
    parser = argparse.ArgumentParser(
        description="Run abundance_matching, matching_1to1, and matching_ml on local_sample."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help=f"Base Haloscope JSON config (default: {DEFAULT_CONFIG}).",
    )
    parser.add_argument(
        "--with-haloscope-features",
        action="store_true",
        help="Attach Haloscope INPUT features (e.g. env) before calibration.",
    )
    return parser.parse_args(argv)


def _summarize_lr_table(lr_table: pd.DataFrame, mass_column: str, method: str) -> dict:
    """
    Build summary statistics for one calibrated LR table.

    Parameters
    ----------
    lr_table : pd.DataFrame
        LR table after calibration.
    mass_column : str
        Mass column used for Haloscope bins.
    method : str
        Mass calibration method key.

    Returns
    -------
    dict
        JSON-serializable summary record.
    """
    calibrated = lr_table[mass_column].to_numpy(dtype=float)
    raw = lr_table["M200b"].to_numpy(dtype=float)
    finite_mask = np.isfinite(calibrated)
    n_lr = len(lr_table)
    n_finite = int(finite_mask.sum())
    summary = {
        "method": method,
        "n_lr": n_lr,
        "n_finite_calibrated_mass": n_finite,
        "fraction_finite_calibrated_mass": float(n_finite / n_lr) if n_lr else 0.0,
        "mass_column": mass_column,
    }
    if n_finite > 0:
        log_cal = np.log10(calibrated[finite_mask])
        log_raw = np.log10(raw[finite_mask])
        summary["log10_mass_cal_mean"] = float(log_cal.mean())
        summary["log10_mass_cal_std"] = float(log_cal.std())
        summary["mean_log10_mass_offset_cal_minus_raw"] = float((log_cal - log_raw).mean())
    if "mass_calib_matched" in lr_table.columns:
        summary["n_matched_flag_true"] = int(lr_table["mass_calib_matched"].sum())
    if "mass_calib_p_keep" in lr_table.columns:
        p_keep = lr_table["mass_calib_p_keep"].to_numpy(dtype=float)
        summary["p_keep_mean"] = float(np.nanmean(p_keep))
        summary["p_keep_median"] = float(np.nanmedian(p_keep))
    return summary


def _run_calibration_only(
    base_config,
    method: str,
) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    """
    Load catalogs and apply one mass-calibration method without INPUT attachers.

    Parameters
    ----------
    base_config
        Parsed Haloscope enrichment configuration.
    method : str
        Mass calibration method key.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame, str]
        HR table, calibrated LR table, and mass column name.
    """
    halos_sim = load_unit_sim_training_catalog(
        Path(base_config.sim_hlist_path),
        max_halos=base_config.max_sim_halos,
    )
    halos_fastpm = load_fastpm_target_catalog(
        Path(base_config.fastpm_list_path),
        max_halos=base_config.max_fastpm_halos,
    )
    method_config = replace(
        base_config,
        mass_calibration=replace(base_config.mass_calibration, method=method),
    )
    lr_table = halos_fastpm.copy()
    hr_table = halos_sim
    box_size = resolve_fastpm_boxsize_mpc_h(method_config)
    lr_table, mass_column = apply_mass_calibration(
        lr_table,
        hr_table,
        method_config.mass_calibration,
        box_size_mpc_h=box_size,
    )
    return hr_table, lr_table, mass_column


def main(argv: list[str]) -> int:
    """
    Run all mass-calibration methods and write a comparison summary.

    Parameters
    ----------
    argv : list[str]
        Command-line arguments without the program name.

    Returns
    -------
    int
        Process exit code (0 on success).
    """
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = _parse_args(argv)
    config_path = Path(args.config).resolve()
    base_config = load_haloscope_enrichment_config(config_path)
    output_dir = Path(base_config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logging.info("Config: %s", config_path)
    logging.info("SIM hlist: %s", base_config.sim_hlist_path)
    logging.info("FastPM list: %s", base_config.fastpm_list_path)
    logging.info(
        "Mode: %s",
        "full feature tables" if args.with_haloscope_features else "calibration only",
    )

    method_summaries = []
    for method in CALIBRATION_METHODS:
        logging.info("Running mass calibration method: %s", method)
        if args.with_haloscope_features:
            method_config = replace(
                base_config,
                mass_calibration=replace(base_config.mass_calibration, method=method),
            )
            hr_table, lr_table, mass_column = build_haloscope_feature_tables(method_config)
        else:
            hr_table, lr_table, mass_column = _run_calibration_only(base_config, method)

        summary = _summarize_lr_table(lr_table, mass_column, method)
        summary["n_hr"] = len(hr_table)
        method_summaries.append(summary)
        parquet_path = output_dir / f"lr_preprocessed_{method}.parquet"
        lr_table.to_parquet(parquet_path, index=False)
        logging.info(
            "Method %s: %s/%s finite %s -> %s",
            method,
            summary["n_finite_calibrated_mass"],
            summary["n_lr"],
            mass_column,
            parquet_path,
        )

    payload = {
        "config": str(config_path),
        "sim_hlist": str(base_config.sim_hlist_path),
        "fastpm_list": str(base_config.fastpm_list_path),
        "calibration_only": not args.with_haloscope_features,
        "methods": method_summaries,
    }
    summary_path = output_dir / SUMMARY_FILENAME
    summary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logging.info("Wrote summary: %s", summary_path)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
