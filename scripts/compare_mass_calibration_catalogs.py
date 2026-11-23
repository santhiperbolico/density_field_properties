#!/usr/bin/env python
"""
Compare halo mass functions (HMF): UNIT vs FastPM raw vs FastPM corrected.

Pipeline mode (``config/pipeline_haloscope_template.json``):
    Reads preprocessed HR/LR Parquet files from ``general.output_dir`` and plots
    UNIT ``M200b`` against FastPM raw and corrected masses.

Benchmark mode (legacy multi-catalog JSON with ``benchmark_catalogs``):
    Re-runs mass-correction methods per FastPM catalog vs UNIT.

Examples (repository root):

    PYTHONPATH=src python scripts/compare_mass_calibration_catalogs.py \\
        --config config/pipeline_haloscope_template.json

    PYTHONPATH=src python scripts/compare_mass_calibration_catalogs.py \\
        --config config/local/mass_calibration_four_catalogs_local_sample.json
"""

import argparse
import json
import logging
import sys
from dataclasses import replace
from pathlib import Path

from density_field_properties.pipelines.config import (
    load_haloscope_enrichment_config,
    resolve_fastpm_boxsize_mpc_h,
)
from density_field_properties.preprocessing.mass_correction.config import (
    MASS_CORRECTION_METHOD_MATCHING_ML,
)

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from mass_calibration_benchmark_helpers import (  # noqa: E402
    attach_matching_ml_input_features,
    benchmark_hmf_from_pipeline_sample,
    build_hmf_comparison_curves,
    is_pipeline_preprocessed_config,
    load_benchmark_hr_lr_tables,
    load_mass_calibration_benchmark_config,
    load_pipeline_preprocessed_parquets,
    resolve_catalog_methods,
    resolve_catalog_train_subbox,
    run_mass_calibration_method,
    save_hmf_comparison_plot,
    skip_matching_ml_reason,
    summarize_lr_calibration,
)

DEFAULT_CONFIG = Path("config/pipeline_haloscope_template.json")
BENCHMARK_SUMMARY_FILENAME = "mass_calibration_benchmark_summary.json"
HMF_FIGURE_TEMPLATE = "hmf_{catalog_key}_{method}.png"
PIPELINE_HMF_FIGURE = "hmf_unit_vs_fastpm_preprocessed.png"
PIPELINE_SUMMARY_FILENAME = "hmf_preprocessed_summary.json"


def _parse_args(argv: list[str]) -> argparse.Namespace:
    """
    Parse CLI arguments for HMF comparison runs.

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
        description="Compare UNIT and FastPM HMFs (raw vs mass-corrected)."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help=f"Pipeline or benchmark JSON (default: {DEFAULT_CONFIG}).",
    )
    return parser.parse_args(argv)


def _run_pipeline_preprocessed_hmf(config_path: Path) -> int:
    """
    Plot HMF from preprocessed Parquet tables referenced in a pipeline config.

    Parameters
    ----------
    config_path : Path
        Path to ``pipeline_haloscope_template.json`` or equivalent.

    Returns
    -------
    int
        Process exit code (0 on success).
    """
    config = load_haloscope_enrichment_config(config_path)
    hr_table, lr_table, mass_column, method = load_pipeline_preprocessed_parquets(config)
    box_size_mpc_h = resolve_fastpm_boxsize_mpc_h(config)
    hmf_config = benchmark_hmf_from_pipeline_sample(config)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logging.info("Config: %s", config_path)
    logging.info("HR Parquet rows: %d", len(hr_table))
    logging.info("LR Parquet rows: %d", len(lr_table))
    logging.info("LR correction column: %s (method=%s)", mass_column, method)
    logging.info("Box size (Mpc/h): %.4g", box_size_mpc_h)

    curves, metrics = build_hmf_comparison_curves(
        hr_table,
        lr_table,
        mass_column,
        method,
        box_size_mpc_h,
        hmf_config=hmf_config,
    )
    figure_path = output_dir / PIPELINE_HMF_FIGURE
    save_hmf_comparison_plot(
        curves,
        figure_path,
        title=f"UNIT vs FastPM HMF ({method})",
        hmf_config=hmf_config,
    )
    logging.info("Wrote HMF figure: %s", figure_path)
    logging.info(
        "RMSE log(n/V) raw vs UNIT: %.4f",
        metrics.get("hmf_rmse_log_nv_raw_vs_unit", float("nan")),
    )
    logging.info(
        "RMSE log(n/V) %s vs UNIT: %.4f",
        method,
        metrics.get(f"hmf_rmse_log_nv_{method}_vs_unit", float("nan")),
    )

    summary = {
        "config": str(config_path),
        "hr_parquet_rows": len(hr_table),
        "lr_parquet_rows": len(lr_table),
        "mass_correction_column": mass_column,
        "mass_correction_method": method,
        "box_size_mpc_h": box_size_mpc_h,
        "hmf_mass_range_msun_h": {
            "min": hmf_config.min_mass_msun_h,
            "max": hmf_config.max_mass_msun_h,
        },
        "hmf_figure": str(figure_path),
        **metrics,
    }
    summary_path = output_dir / PIPELINE_SUMMARY_FILENAME
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logging.info("Wrote summary: %s", summary_path)
    return 0


def _run_multi_catalog_benchmark(config_path: Path) -> int:
    """
    Run legacy multi-catalog mass-correction benchmark with per-method HMF plots.

    Parameters
    ----------
    config_path : Path
        Benchmark JSON with ``benchmark_catalogs``.

    Returns
    -------
    int
        Process exit code (0 on success).
    """
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
        if MASS_CORRECTION_METHOD_MATCHING_ML in catalog_methods:
            hr_work, lr_work = attach_matching_ml_input_features(
                hr_table, lr_template, base_config
            )

        for method in catalog_methods:
            skip_reason = None
            if method == MASS_CORRECTION_METHOD_MATCHING_ML:
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
                mass_correction=replace(
                    base_config.mass_correction,
                    method=method,
                    matching_ml=replace(
                        base_config.mass_correction.matching_ml,
                        train_subbox=train_subbox,
                    ),
                ),
            )
            ml_settings = method_config.mass_correction.matching_ml
            if (
                ml_settings.model_path is None
                and ml_settings.write_model
                and method == MASS_CORRECTION_METHOD_MATCHING_ML
            ):
                model_dir = output_dir / "models"
                model_dir.mkdir(parents=True, exist_ok=True)
                model_path = model_dir / f"{entry.key}_matching_ml.joblib"
                method_config = replace(
                    method_config,
                    mass_correction=replace(
                        method_config.mass_correction,
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
                method_config.mass_correction.matching_ml.regressor.regressor_type
                if method == MASS_CORRECTION_METHOD_MATCHING_ML
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
    summary_path = output_dir / BENCHMARK_SUMMARY_FILENAME
    summary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logging.info("Wrote summary: %s", summary_path)
    return 0


def main(argv: list[str]) -> int:
    """
    Run pipeline Parquet HMF mode or legacy multi-catalog benchmark.

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
    with config_path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Config must be a JSON object: {config_path}")

    if is_pipeline_preprocessed_config(payload) and "benchmark_catalogs" not in payload:
        return _run_pipeline_preprocessed_hmf(config_path)
    return _run_multi_catalog_benchmark(config_path)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
