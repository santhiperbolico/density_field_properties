#!/usr/bin/env python
"""
Compare mass-calibration methods across FastPM catalogs against UNIT (local_sample).

Rockstar FastPM lists use host halos only (``PID == -1``) when the catalog exposes
``PID``; FoF groups are unchanged. UNIT ``hlist`` is already host-only.

Runs ``abundance_matching`` and ``matching_ml`` (XGBoost by default) for each entry
in ``benchmark_catalogs``, including Rockstar lists and FoF BigFile groups.

Example (repository root):

    PYTHONPATH=src python scripts/compare_mass_calibration_catalogs.py \\
        --config config/local/mass_calibration_four_catalogs_local_sample.json

Taurus (full MN5 box; catalog paths match ``config/hmf_four_catalogs_taurus.json``):

    sbatch slurm/compare_halo_hmf/compare_mass_calibration_four_catalogs_taurus.slurm
"""

import argparse
import json
import logging
import sys
from dataclasses import replace
from pathlib import Path

from density_field_properties.pipelines.config import resolve_fastpm_boxsize_mpc_h
from density_field_properties.preprocessing.mass_calibration_config import (
    MASS_CALIBRATION_METHOD_MATCHING_ML,
)

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from mass_calibration_benchmark_helpers import (  # noqa: E402
    attach_matching_ml_input_features,
    build_hmf_comparison_curves,
    load_benchmark_hr_lr_tables,
    load_mass_calibration_benchmark_config,
    resolve_catalog_methods,
    resolve_catalog_train_subbox,
    run_mass_calibration_method,
    save_hmf_comparison_plot,
    skip_matching_ml_reason,
    summarize_lr_calibration,
)

DEFAULT_CONFIG = Path("config/local/mass_calibration_four_catalogs_local_sample.json")
SUMMARY_FILENAME = "mass_calibration_benchmark_summary.json"
HMF_FIGURE_TEMPLATE = "hmf_{catalog_key}_{method}.png"


def _parse_args(argv: list[str]) -> argparse.Namespace:
    """
    Parse CLI arguments for the multi-catalog benchmark.

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
        description="Benchmark mass calibration on multiple FastPM catalogs vs UNIT."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help=f"Benchmark JSON config (default: {DEFAULT_CONFIG}).",
    )
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    """
    Run configured calibration methods for each FastPM catalog.

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
    base_config, catalogs, methods, hmf_config = load_mass_calibration_benchmark_config(
        config_path
    )
    output_dir = Path(base_config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    box_size_mpc_h = resolve_fastpm_boxsize_mpc_h(base_config)

    logging.info("Config: %s", config_path)
    logging.info("UNIT hlist: %s", base_config.sim_hlist_path)
    logging.info("Output: %s", output_dir)
    logging.info("Methods: %s", ", ".join(methods))

    run_records: list[dict] = []
    figure_paths: dict[str, str] = {}

    for entry in catalogs:
        logging.info("Loading catalogs for %s (%s)", entry.key, entry.path)
        hr_table, lr_template = load_benchmark_hr_lr_tables(base_config, entry)
        logging.info("HR halos: %d, LR halos: %d", len(hr_table), len(lr_template))

        catalog_methods = resolve_catalog_methods(entry, methods)
        train_subbox = resolve_catalog_train_subbox(base_config, entry)
        hr_work, lr_work = hr_table, lr_template
        if MASS_CALIBRATION_METHOD_MATCHING_ML in catalog_methods:
            hr_work, lr_work = attach_matching_ml_input_features(
                hr_table, lr_template, base_config
            )

        for method in catalog_methods:
            skip_reason = None
            if method == MASS_CALIBRATION_METHOD_MATCHING_ML:
                skip_reason = skip_matching_ml_reason(hr_work, lr_work, base_config, entry)
            if skip_reason is not None:
                logging.warning("%s: %s", entry.key, skip_reason)
                run_records.append(
                    {
                        "catalog_key": entry.key,
                        "catalog_label": entry.label,
                        "method": method,
                        "status": "skipped",
                        "skip_reason": skip_reason,
                    }
                )
                continue

            logging.info("Calibrating %s with method %s", entry.key, method)
            method_config = replace(
                base_config,
                mass_calibration=replace(
                    base_config.mass_calibration,
                    method=method,
                    matching_ml=replace(
                        base_config.mass_calibration.matching_ml,
                        train_subbox=train_subbox,
                    ),
                ),
            )
            ml_settings = method_config.mass_calibration.matching_ml
            if (
                ml_settings.model_path is None
                and ml_settings.write_model
                and method == MASS_CALIBRATION_METHOD_MATCHING_ML
            ):
                model_dir = output_dir / "models"
                model_dir.mkdir(parents=True, exist_ok=True)
                model_path = model_dir / f"{entry.key}_matching_ml.joblib"
                method_config = replace(
                    method_config,
                    mass_calibration=replace(
                        method_config.mass_calibration,
                        matching_ml=replace(ml_settings, model_path=model_path),
                    ),
                )

            lr_table, mass_column = run_mass_calibration_method(
                method_config,
                method,
                hr_work,
                lr_work.copy(),
                box_size_mpc_h,
            )
            summary = summarize_lr_calibration(lr_table, mass_column, method, entry.key)
            summary["status"] = "ok"
            summary["catalog_label"] = entry.label
            summary["catalog_path"] = str(entry.path)
            summary["n_hr"] = len(hr_table)
            summary["regressor_type"] = (
                method_config.mass_calibration.matching_ml.regressor.regressor_type
                if method == MASS_CALIBRATION_METHOD_MATCHING_ML
                else None
            )

            curves, hmf_metrics = build_hmf_comparison_curves(
                hr_table,
                lr_table,
                mass_column,
                method,
                box_size_mpc_h,
                hmf_config=hmf_config,
            )
            summary.update(hmf_metrics)

            catalog_dir = output_dir / entry.key
            catalog_dir.mkdir(parents=True, exist_ok=True)
            parquet_path = catalog_dir / f"lr_calibrated_{method}.parquet"
            lr_table.to_parquet(parquet_path, index=False)
            summary["lr_parquet"] = str(parquet_path)

            figure_path = catalog_dir / HMF_FIGURE_TEMPLATE.format(
                catalog_key=entry.key,
                method=method,
            )
            save_hmf_comparison_plot(
                curves,
                figure_path,
                title=f"HMF: {entry.label} ({method})",
                hmf_config=hmf_config,
            )
            figure_paths[f"{entry.key}/{method}"] = str(figure_path)
            summary["hmf_figure"] = str(figure_path)
            run_records.append(summary)
            logging.info(
                "Finished %s/%s: %s/%s finite masses, RMSE cal=%.4f",
                entry.key,
                method,
                summary["n_finite_calibrated_mass"],
                summary["n_lr"],
                summary.get(f"hmf_rmse_log_nv_{method}_vs_unit", float("nan")),
            )

    payload = {
        "config": str(config_path),
        "sim_hlist": str(base_config.sim_hlist_path),
        "box_size_mpc_h": box_size_mpc_h,
        "benchmark_hmf": {
            "min_mass_msun_h": hmf_config.min_mass_msun_h,
            "max_mass_msun_h": hmf_config.max_mass_msun_h,
        },
        "methods": list(methods),
        "catalogs": [
            {
                "key": entry.key,
                "label": entry.label,
                "path": str(entry.path),
                "loader": entry.loader,
            }
            for entry in catalogs
        ],
        "hmf_figures": figure_paths,
        "runs": run_records,
    }
    summary_path = output_dir / SUMMARY_FILENAME
    summary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logging.info("Wrote summary: %s", summary_path)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
