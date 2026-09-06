#!/usr/bin/env python
"""
Verify FastPM and UNIT initial-condition compatibility via halo cross-correlation r(k).

Example (Taurus, subset):

    PYTHONPATH=src python scripts/verify_fastpm_unit_ics.py \\
        --n-halos 100000 --n-grid 128 \\
        --output-dir output/fastpm_unit_seed_check

Example (full FastPM catalog, capped UNIT to the same count is not applied — see
``--all-halos`` warning in logs):

    PYTHONPATH=src python scripts/verify_fastpm_unit_ics.py \\
        --all-halos --n-grid 256 \\
        --output-dir output/fastpm_unit_seed_check_full
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from density_field_properties.density_field.power_spectrum import (
    median_r_k_below_k,
    spherical_power_spectra,
)
from density_field_properties.halo_catalog.rockstar import (
    read_rockstar_box_size_header,
    read_rockstar_cosmology_header,
)
from density_field_properties.haloscope.sim_to_fastpm.assembly_bias import (
    halo_overdensity_field_cic,
)
from density_field_properties.haloscope.sim_to_fastpm.config import (
    FASTPM_BOXSIZE_MPC_H,
    SIM_BOXSIZE_MPC_H,
    default_fastpm_list_path,
    default_unit_rockstar_list_path,
)
from density_field_properties.haloscope.sim_to_fastpm.load_catalogs import (
    _rockstar_pid_column_index,
    load_fastpm_central_target_catalog,
    load_unit_rockstar_target_catalog,
    load_unit_sim_training_catalog,
)

DEFAULT_OUTPUT_DIR = Path("output/fastpm_unit_seed_check")
DEFAULT_N_HALOS = 500_000
DEFAULT_N_GRID = 128
DEFAULT_N_K_BINS = 64
DEFAULT_K_LOW_THRESHOLD = 0.05
COMPATIBLE_MEDIAN_R = 0.9
INCOMPATIBLE_MEDIAN_R = 0.3


def _parse_args(argv: list[str]) -> argparse.Namespace:
    """
    Parse CLI arguments for the IC seed verification script.

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
        description="Compute halo cross-correlation r(k) for FastPM vs UNIT IC check"
    )
    parser.add_argument(
        "--fastpm-list",
        type=Path,
        default=default_fastpm_list_path(),
        help=(
            "Path to FastPM Rockstar out_*.list catalog "
            "(default: .../rockstar_out_pm/out_8.list)."
        ),
    )
    parser.add_argument(
        "--unit-list",
        type=Path,
        default=default_unit_rockstar_list_path(),
        help=(
            "Path to UNIT Rockstar catalog (default: "
            "/data21/UNITSIM/.../ROCKSTAR/out_128p.list.bz2, a=1)."
        ),
    )
    parser.add_argument(
        "--box-size",
        type=float,
        default=None,
        help="Box side length in Mpc/h; default: read from catalog headers.",
    )
    parser.add_argument(
        "--n-halos",
        type=int,
        default=DEFAULT_N_HALOS,
        help=(
            "Random uniform subsample size per catalog (reservoir sampling). "
            "Ignored when --all-halos is set."
        ),
    )
    parser.add_argument(
        "--all-halos",
        action="store_true",
        help=(
            "Read every halo that passes the central/mass filters (no count cap). "
            "Impractical for the full UNIT Rockstar catalog (~1e8 centrals)."
        ),
    )
    parser.add_argument(
        "--n-grid",
        type=int,
        default=DEFAULT_N_GRID,
        help="CIC grid resolution per dimension.",
    )
    parser.add_argument(
        "--min-m200b",
        type=float,
        default=0.0,
        help="Optional lower mass cut in Msun/h before CIC deposition.",
    )
    parser.add_argument(
        "--n-k-bins",
        type=int,
        default=DEFAULT_N_K_BINS,
        help="Number of spherical |k| bins for P(k).",
    )
    parser.add_argument(
        "--k-low-threshold",
        type=float,
        default=DEFAULT_K_LOW_THRESHOLD,
        help="k threshold in h/Mpc for the summary median r(k).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for r_k.csv, r_k.png, and summary.json.",
    )
    return parser.parse_args(argv)


def _resolve_n_halos(args: argparse.Namespace) -> int | None:
    """
    Resolve the halo count cap from parsed CLI options.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed CLI options.

    Returns
    -------
    int or None
        Maximum halos per catalog, or ``None`` for no cap.
    """
    if args.all_halos:
        return None
    return args.n_halos


def _is_unit_hlist_catalog(path: Path) -> bool:
    """
    Return whether ``path`` points to a consistent-trees UNIT hlist catalog.

    Parameters
    ----------
    path : Path
        Catalog file path.

    Returns
    -------
    bool
        True for consistent-trees ``hlist_*`` filenames.
    """
    return path.name.startswith("hlist_")


def _resolve_box_size(
    fastpm_list: Path,
    unit_list: Path,
    box_size_cli: float | None,
) -> float:
    """
    Resolve and validate the simulation box size in Mpc/h.

    Parameters
    ----------
    fastpm_list : Path
        FastPM Rockstar catalog path.
    unit_list : Path
        UNIT catalog path (hlist or Rockstar ``.list``).
    box_size_cli : float or None
        User-provided box size, if any.

    Returns
    -------
    float
        Box side length in Mpc/h.

    Raises
    ------
    ValueError
        If headers disagree or no box size can be determined.
    """
    if box_size_cli is not None:
        return float(box_size_cli)

    fastpm_box = read_rockstar_box_size_header(str(fastpm_list))
    unit_box = read_rockstar_box_size_header(str(unit_list))
    if unit_box is None:
        unit_box = SIM_BOXSIZE_MPC_H

    if fastpm_box is None:
        fastpm_box = FASTPM_BOXSIZE_MPC_H
    if not np.isclose(fastpm_box, unit_box, rtol=1e-3):
        raise ValueError(f"Box size mismatch: FastPM={fastpm_box} Mpc/h, UNIT={unit_box} Mpc/h")
    return float(fastpm_box)


def _apply_mass_cut(
    positions: np.ndarray,
    weights: np.ndarray,
    min_m200b: float,
    catalog_path: Path,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Optionally filter halos by a minimum ``M200b`` threshold.

    Parameters
    ----------
    positions : np.ndarray
        Halo positions with shape ``(N, 3)``.
    weights : np.ndarray
        CIC weights with shape ``(N,)``.
    min_m200b : float
        Minimum ``M200b`` in Msun/h.
    catalog_path : Path
        Source catalog path for error messages.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Filtered positions and weights.

    Raises
    ------
    ValueError
        If no halos remain after filtering.
    """
    if min_m200b > 0.0:
        mask = weights >= min_m200b
        positions = positions[mask]
        weights = weights[mask]
    if positions.shape[0] == 0:
        raise ValueError(f"No halos left after filtering catalog {catalog_path}")
    return positions, weights


def _resolve_halo_selection(fastpm_list: Path) -> tuple[bool, str]:
    """
    Choose a symmetric central-halo filter for both catalogs.

    Parameters
    ----------
    fastpm_list : Path
        FastPM Rockstar catalog path.

    Returns
    -------
    tuple[bool, str]
        ``(central_only, filter_label)`` for summary metadata.
    """
    if _rockstar_pid_column_index(fastpm_list) is not None:
        return True, "pid==-1"
    logging.warning(
        "FastPM catalog has no PID column; comparing all positive-mass halos on both sides."
    )
    return False, "none"


def _load_fastpm_halos(
    list_path: Path,
    n_halos: int | None,
    min_m200b: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Load FastPM halo positions and CIC weights from a Rockstar ``.list`` file.

    Parameters
    ----------
    list_path : Path
        FastPM Rockstar catalog path.
    n_halos : int or None
        Maximum number of halos to read; ``None`` loads the full catalog.
    min_m200b : float
        Minimum ``M200b`` in Msun/h.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Positions with shape ``(N, 3)`` and weights with shape ``(N,)``.
    """
    frame = load_fastpm_central_target_catalog(list_path, max_centrals=n_halos)
    positions = frame[["x", "y", "z"]].to_numpy(dtype=np.float64)
    weights = frame["M200b"].to_numpy(dtype=np.float64)
    return _apply_mass_cut(positions, weights, min_m200b, list_path)


def _load_unit_halos(
    list_path: Path,
    n_halos: int | None,
    min_m200b: float,
    central_only: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Load UNIT halo positions and CIC weights from an hlist or Rockstar catalog.

    Parameters
    ----------
    list_path : Path
        UNIT catalog path.
    n_halos : int or None
        Maximum number of halos to read; ``None`` loads the full catalog.
    min_m200b : float
        Minimum ``M200b`` in Msun/h.
    central_only : bool
        When True, keep UNIT host halos with ``PID == -1`` only.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Positions with shape ``(N, 3)`` and weights with shape ``(N,)``.
    """
    if _is_unit_hlist_catalog(list_path):
        frame = load_unit_sim_training_catalog(list_path, max_halos=n_halos)
        positions = frame[["x", "y", "z"]].to_numpy(dtype=np.float64)
        weights = frame["M200b"].to_numpy(dtype=np.float64)
        return _apply_mass_cut(positions, weights, min_m200b, list_path)

    frame = load_unit_rockstar_target_catalog(
        list_path,
        max_centrals=n_halos,
        central_only=central_only,
    )
    positions = frame[["x", "y", "z"]].to_numpy(dtype=np.float64)
    weights = frame["M200b"].to_numpy(dtype=np.float64)
    return _apply_mass_cut(positions, weights, min_m200b, list_path)


def _classify_verdict(median_r_k_low: float) -> str:
    """
    Map the low-k median r(k) to a coarse IC compatibility verdict.

    Parameters
    ----------
    median_r_k_low : float
        Median r(k) for k below the configured threshold.

    Returns
    -------
    str
        One of ``compatible``, ``incompatible``, or ``inconclusive``.
    """
    if not np.isfinite(median_r_k_low):
        return "inconclusive"
    if median_r_k_low >= COMPATIBLE_MEDIAN_R:
        return "compatible"
    if median_r_k_low <= INCOMPATIBLE_MEDIAN_R:
        return "incompatible"
    return "inconclusive"


def _save_r_k_plot(
    k: np.ndarray,
    r_k: np.ndarray,
    output_path: Path,
    median_r_k_low: float,
    k_low_threshold: float,
) -> None:
    """
    Save r(k) versus k to a PNG figure.

    Parameters
    ----------
    k : np.ndarray
        Bin-center wavenumbers.
    r_k : np.ndarray
        Correlation coefficient per bin.
    output_path : Path
        Destination PNG path.
    median_r_k_low : float
        Summary statistic shown in the legend.
    k_low_threshold : float
        Threshold used for the summary statistic.
    """
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.axhline(1.0, color="0.5", linestyle="--", linewidth=1.0, label="r(k)=1")
    axis.plot(k, r_k, marker="o", markersize=3, linewidth=1.2, label="r(k)")
    axis.set_xscale("log")
    axis.set_xlabel(r"$k$ [$h\,\mathrm{Mpc}^{-1}$]")
    axis.set_ylabel(r"$r(k)$")
    axis.set_title("FastPM vs UNIT halo cross-correlation")
    axis.legend(title=(f"median r(k) for k<{k_low_threshold:g} = " f"{median_r_k_low:.3f}"))
    axis.grid(True, alpha=0.3)
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def run_verification(args: argparse.Namespace) -> dict:
    """
    Execute the full r(k) verification workflow.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed CLI options.

    Returns
    -------
    dict
        Summary dictionary also written to ``summary.json``.
    """
    if not args.fastpm_list.is_file():
        raise FileNotFoundError(f"FastPM catalog not found: {args.fastpm_list}")
    if not args.unit_list.is_file():
        raise FileNotFoundError(f"UNIT catalog not found: {args.unit_list}")

    n_halos = _resolve_n_halos(args)
    if n_halos is None:
        logging.warning(
            "Loading all halos after filters (--all-halos). "
            "The UNIT Rockstar catalog can exceed memory on a login node; "
            "prefer Slurm and a subset for UNIT unless you know the size fits."
        )

    central_only, halo_filter = _resolve_halo_selection(args.fastpm_list)

    box_size = _resolve_box_size(args.fastpm_list, args.unit_list, args.box_size)
    fastpm_cosmo = read_rockstar_cosmology_header(str(args.fastpm_list))
    if _is_unit_hlist_catalog(args.unit_list):
        unit_cosmo = None
    else:
        unit_cosmo = read_rockstar_cosmology_header(str(args.unit_list))
    fastpm_pos, fastpm_weights = _load_fastpm_halos(
        args.fastpm_list,
        n_halos,
        args.min_m200b,
    )
    unit_pos, unit_weights = _load_unit_halos(
        args.unit_list,
        n_halos,
        args.min_m200b,
        central_only,
    )

    logging.info(
        "Building CIC halo fields: FastPM N=%d, UNIT N=%d, n_grid=%d, box=%.3f Mpc/h",
        fastpm_pos.shape[0],
        unit_pos.shape[0],
        args.n_grid,
        box_size,
    )
    delta_fastpm = halo_overdensity_field_cic(
        fastpm_pos,
        fastpm_weights,
        box_size,
        args.n_grid,
    )
    delta_unit = halo_overdensity_field_cic(
        unit_pos,
        unit_weights,
        box_size,
        args.n_grid,
    )

    spectra = spherical_power_spectra(
        delta_fastpm,
        delta_unit,
        box_size,
        n_k_bins=args.n_k_bins,
    )
    median_r_k_low = median_r_k_below_k(
        spectra["k"],
        spectra["r_k"],
        args.k_low_threshold,
    )
    verdict = _classify_verdict(median_r_k_low)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "r_k.csv"
    figure_path = args.output_dir / "r_k.png"
    summary_path = args.output_dir / "summary.json"

    frame = pd.DataFrame(
        {
            "k_h_mpc": spectra["k"],
            "pk_fastpm": spectra["pk_a"],
            "pk_unit": spectra["pk_b"],
            "cross_pk": spectra["cross_pk"],
            "r_k": spectra["r_k"],
        }
    )
    frame.to_csv(csv_path, index=False)
    _save_r_k_plot(
        spectra["k"],
        spectra["r_k"],
        figure_path,
        median_r_k_low,
        args.k_low_threshold,
    )

    summary = {
        "fastpm_list": str(args.fastpm_list),
        "unit_list": str(args.unit_list),
        "n_halos_requested": n_halos,
        "all_halos": n_halos is None,
        "halo_filter": halo_filter,
        "central_only": central_only,
        "n_halos_fastpm": int(fastpm_pos.shape[0]),
        "n_halos_unit": int(unit_pos.shape[0]),
        "min_m200b_msun_h": args.min_m200b,
        "box_size_mpc_h": box_size,
        "n_grid": args.n_grid,
        "n_k_bins": args.n_k_bins,
        "k_low_threshold_h_mpc": args.k_low_threshold,
        "median_r_k_low": median_r_k_low,
        "verdict": verdict,
        "fastpm_cosmology": (
            None
            if fastpm_cosmo is None
            else {
                "omega_matter": fastpm_cosmo.omega_matter,
                "omega_lambda": fastpm_cosmo.omega_lambda,
                "h0": fastpm_cosmo.h0,
            }
        ),
        "unit_cosmology": (
            None
            if unit_cosmo is None
            else {
                "omega_matter": unit_cosmo.omega_matter,
                "omega_lambda": unit_cosmo.omega_lambda,
                "h0": unit_cosmo.h0,
            }
        ),
        "outputs": {
            "r_k_csv": str(csv_path),
            "r_k_png": str(figure_path),
            "summary_json": str(summary_path),
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    logging.info("Median r(k) for k < %.3f h/Mpc: %.4f", args.k_low_threshold, median_r_k_low)
    logging.info("Verdict: %s", verdict)
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
    args = _parse_args(argv)
    run_verification(args)
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    raise SystemExit(main(sys.argv[1:]))
