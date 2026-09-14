#!/usr/bin/env python
"""
Compare halo mass functions for UNIT and FastPM catalogs at a=1.

Example (Taurus):

    PYTHONPATH=src python scripts/compare_halo_hmf.py \\
        --output-dir output/halo_hmf_a1
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from density_field_properties.halo_catalog.mass_function import (
    HaloMassFunctionResult,
    default_hmf_log_mass_bin_edges,
    halo_mass_function_from_catalog,
)
from density_field_properties.haloscope.sim_to_fastpm.config import (
    DM_MASS_PARTICLE_MSUN_H,
    default_unit_rockstar_list_path,
)

DEFAULT_OUTPUT_DIR = Path("output/halo_hmf_a1")
DEFAULT_FASTPM_FOF = Path(
    "/data21/users/mruiz/fastpm_MN5/fastpm_tfm/output_01/fof_1.0000"
)
DEFAULT_FASTPM_ROCKSTAR_PM = Path(
    "/data21/users/mruiz/fastpm_MN5/fastpm_tfm/rockstar_out_pm/out_8.list"
)
DEFAULT_FASTPM_ROCKSTAR_NBODY = Path(
    "/data21/users/mruiz/fastpm_MN5/fastpm_tfm/rockstar_out_nbody/out_8.list"
)
DEFAULT_UNIT = default_unit_rockstar_list_path()
DEFAULT_LOG_MASS_MIN = 10.0
DEFAULT_LOG_MASS_MAX = 14.5
DEFAULT_N_BINS = 20

def _parse_args(argv: list[str]) -> argparse.Namespace:
    """
    Parse CLI arguments for the halo mass function comparison script.

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
        description="Compare halo mass functions for UNIT and FastPM at a=1"
    )
    parser.add_argument(
        "--unit-list",
        type=Path,
        default=DEFAULT_UNIT,
        help="UNIT Rockstar catalog at a=1 (default: out_128p.list.bz2).",
    )
    parser.add_argument(
        "--fastpm-fof",
        type=Path,
        default=DEFAULT_FASTPM_FOF,
        help="FastPM native FOF BigFile directory at a=1.",
    )
    parser.add_argument(
        "--fastpm-rockstar-pm",
        type=Path,
        default=DEFAULT_FASTPM_ROCKSTAR_PM,
        help="FastPM Rockstar PM out_*.list catalog at a=1.",
    )
    parser.add_argument(
        "--fastpm-rockstar-nbody",
        type=Path,
        default=DEFAULT_FASTPM_ROCKSTAR_NBODY,
        help="FastPM Rockstar N-body out_*.list catalog at a=1.",
    )
    parser.add_argument(
        "--box-size",
        type=float,
        default=None,
        help="Override simulation box size in Mpc/h for all catalogs.",
    )
    parser.add_argument(
        "--min-m200b",
        type=float,
        default=0.0,
        help="Minimum halo M200b in Msun/h.",
    )
    parser.add_argument(
        "--min-m200b-times-mp",
        type=float,
        default=None,
        help=(
            "Minimum halo mass as a factor times DM_MASS_PARTICLE_MSUN_H "
            f"(currently {DM_MASS_PARTICLE_MSUN_H:.6g} Msun/h). "
            "Overrides --min-m200b when set."
        ),
    )
    parser.add_argument(
        "--max-halos",
        type=int,
        default=0,
        help="Maximum halos per catalog (0 = no limit).",
    )
    parser.add_argument(
        "--log-mass-min",
        type=float,
        default=DEFAULT_LOG_MASS_MIN,
        help="Lower log10(M200b) bin edge in Msun/h.",
    )
    parser.add_argument(
        "--log-mass-max",
        type=float,
        default=DEFAULT_LOG_MASS_MAX,
        help="Upper log10(M200b) bin edge in Msun/h.",
    )
    parser.add_argument(
        "--n-bins",
        type=int,
        default=DEFAULT_N_BINS,
        help="Number of log10 mass bins.",
    )
    parser.add_argument(
        "--include-subhalos",
        action="store_true",
        help="Include subhalos (do not require PID == -1 for Rockstar catalogs).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for CSV, PNG, and summary.json.",
    )
    return parser.parse_args(argv)


def _resolve_min_mass_msun_h(args: argparse.Namespace) -> float:
    """
    Resolve the minimum halo mass threshold from CLI options.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed CLI options.

    Returns
    -------
    float
        Minimum mass in Msun/h applied to every catalog.
    """
    if args.min_m200b_times_mp is not None:
        return float(args.min_m200b_times_mp * DM_MASS_PARTICLE_MSUN_H)
    return float(args.min_m200b)


def _resolve_max_halos(max_halos_cli: int) -> int | None:
    """
    Resolve the optional per-catalog halo cap from CLI input.

    Parameters
    ----------
    max_halos_cli : int
        CLI value where zero means no limit.

    Returns
    -------
    int or None
        Maximum halos to read, or ``None`` for the full catalog.
    """
    if max_halos_cli <= 0:
        return None
    return max_halos_cli


def _catalog_specs(args: argparse.Namespace) -> list[tuple[str, Path, str | None]]:
    """
    Build the list of catalogs to compare.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed CLI options.

    Returns
    -------
    list[tuple[str, Path, str | None]]
        Tuples of label, path, and optional explicit reader name.
    """
    return [
        ("unit", args.unit_list, "rockstar"),
        ("fastpm_fof", args.fastpm_fof, "fastpm"),
        ("fastpm_rockstar_pm", args.fastpm_rockstar_pm, "rockstar"),
        ("fastpm_rockstar_nbody", args.fastpm_rockstar_nbody, "rockstar"),
    ]


def _validate_catalog_paths(catalog_specs: list[tuple[str, Path, str | None]]) -> None:
    """
    Ensure all requested catalog paths exist.

    Parameters
    ----------
    catalog_specs : list[tuple[str, Path, str | None]]
        Catalog definitions to validate.

    Raises
    ------
    FileNotFoundError
        If any catalog path is missing.
    """
    for label, path, _ in catalog_specs:
        if not path.exists():
            raise FileNotFoundError(f"Catalog '{label}' not found: {path}")


def _hmf_to_dataframe(label: str, result: HaloMassFunctionResult) -> pd.DataFrame:
    """
    Convert one mass-function result into a long-form DataFrame.

    Parameters
    ----------
    label : str
        Catalog label.
    result : HaloMassFunctionResult
        Computed halo mass function.

    Returns
    -------
    pd.DataFrame
        One row per occupied mass bin.
    """
    return pd.DataFrame(
        {
            "catalog": label,
            "log10_m200b": result.log_mass_bin_centers,
            "count": result.counts,
            "dn_dlog10_m": result.dn_dlog10_m,
            "dn_dln_m": result.dn_dln_m,
        }
    )


def _save_hmf_plot(
    results: dict[str, HaloMassFunctionResult],
    figure_path: Path,
) -> None:
    """
    Save a comparison plot of dn/dln(M) versus log10(M200b).

    Parameters
    ----------
    results : dict[str, HaloMassFunctionResult]
        Mass functions keyed by catalog label.
    figure_path : Path
        Output PNG path.
    """
    figure, axis = plt.subplots(figsize=(8, 6))
    for label, result in results.items():
        valid = np.isfinite(result.log_mass_bin_centers) & np.isfinite(result.dn_dln_m)
        valid &= result.counts > 0
        axis.plot(
            result.log_mass_bin_centers[valid],
            result.dn_dln_m[valid],
            marker="o",
            linewidth=1.5,
            label=label,
        )

    axis.set_xlabel(r"$\log_{10}(M_{200b}\,/\,M_\odot h^{-1})$")
    axis.set_ylabel(r"$dn / d\ln M_{200b}$ [$h^3\,\mathrm{Mpc}^{-3}$]")
    axis.set_yscale("log")
    axis.legend(loc="best")
    axis.grid(True, alpha=0.3)
    figure.tight_layout()
    figure.savefig(figure_path, dpi=120)
    plt.close(figure)


def run_comparison(args: argparse.Namespace) -> dict[str, object]:
    """
    Compute and save halo mass functions for all configured catalogs.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed CLI options.

    Returns
    -------
    dict[str, object]
        Summary dictionary also written to ``summary.json``.
    """
    catalog_specs = _catalog_specs(args)
    _validate_catalog_paths(catalog_specs)

    max_halos = _resolve_max_halos(args.max_halos)
    min_mass_msun_h = _resolve_min_mass_msun_h(args)
    log_mass_bin_edges = default_hmf_log_mass_bin_edges(
        log_mass_min=args.log_mass_min,
        log_mass_max=args.log_mass_max,
        n_bins=args.n_bins,
    )
    central_only = not args.include_subhalos

    results: dict[str, HaloMassFunctionResult] = {}
    for label, path, reader_name in catalog_specs:
        logging.info("Computing HMF for %s: %s", label, path)
        results[label] = halo_mass_function_from_catalog(
            path,
            catalog_name=reader_name,
            box_size_mpc_h=args.box_size,
            central_only=central_only,
            min_mass_msun_h=min_mass_msun_h,
            max_halos=max_halos,
            log_mass_bin_edges=log_mass_bin_edges,
        )
        logging.info(
            "  %s: n_halos=%d, box=%.3f Mpc/h",
            label,
            results[label].n_halos,
            results[label].box_size_mpc_h,
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.concat(
        [_hmf_to_dataframe(label, result) for label, result in results.items()],
        ignore_index=True,
    )
    csv_path = args.output_dir / "hmf_comparison.csv"
    figure_path = args.output_dir / "hmf_comparison.png"
    summary_path = args.output_dir / "summary.json"

    frame.to_csv(csv_path, index=False)
    _save_hmf_plot(results, figure_path)

    summary = {
        "scale_factor": 1.0,
        "central_only": central_only,
        "min_m200b": min_mass_msun_h,
        "min_m200b_times_mp": args.min_m200b_times_mp,
        "dm_mass_particle_msun_h": DM_MASS_PARTICLE_MSUN_H,
        "max_halos": max_halos,
        "log_mass_bin_edges": log_mass_bin_edges.tolist(),
        "catalogs": {
            label: {
                "path": str(path),
                "reader": reader_name,
                "n_halos": results[label].n_halos,
                "box_size_mpc_h": results[label].box_size_mpc_h,
            }
            for (label, path, reader_name) in catalog_specs
        },
        "outputs": {
            "csv": str(csv_path),
            "figure": str(figure_path),
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logging.info("Wrote %s", csv_path)
    logging.info("Wrote %s", figure_path)
    logging.info("Wrote %s", summary_path)
    return summary


def main(argv: list[str]) -> int:
    """
    CLI entry point.

    Parameters
    ----------
    argv : list[str]
        Command-line arguments without the program name.

    Returns
    -------
    int
        Process exit code.
    """
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = _parse_args(argv)
    try:
        run_comparison(args)
    except (FileNotFoundError, ValueError) as error:
        logging.error("%s", error)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
