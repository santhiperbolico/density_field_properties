"""Paths and hyperparameters for SIM-to-FastPM Haloscope."""

from pathlib import Path
from typing import Optional

SIM_ROCKSTAR_ROOT = Path("/data5/UNITSIM/fixedAmp_InvPhase_001/ROCKSTAR/outputs")
SIM_HLIST_RELATIVE = Path("hlists/hlist_1.00000.list.bz2")
SIM_BOXSIZE_MPC_H = 1000.0

FASTPM_ROCKSTAR_DIR = Path("/data21/users/mruiz/fastpm_MN5/fastpm_tfm/rockstar_out_pm")
FASTPM_LIST_NAME = "out_8.list"
FASTPM_BOXSIZE_MPC_H = 1000.0

ENV_RADIUS_MPC_H = 5.0
CALIBRATE_MASS = True

INPUT_FEATURES = ("env",)
OUTPUT_FEATURES = ("cv", "Spin", "ca", "ba")

MAX_SIM_HALOS = None
MAX_FASTPM_HALOS = None

QUICK_RUN = False

SMOKE_MAX_SIM_HALOS = 8000
SMOKE_MAX_FASTPM_HALOS = 8000
SMOKE_MIN_BIN_SIZE = 5
PRODUCTION_MIN_BIN_SIZE = 10

OUTPUT_DIR = Path("output/sim_to_fastpm_haloscope")
ENRICHED_PARQUET_NAME = "fastpm_out_8_haloscope_enriched.parquet"

UNIT_HLIST_COLUMNS = {
    "id": 1,
    "pid": 5,
    "Rvir": 11,
    "x": 17,
    "y": 18,
    "z": 19,
    "Spin": 26,
    "Rs_Klypin": 37,
    "M200b": 39,
    "ba": 46,
    "ca": 47,
}

ROCKSTAR_LIST_COLUMNS = {
    "halo_id": 0,
    "halo_x": 8,
    "halo_y": 9,
    "halo_z": 10,
    "halo_m200b": 20,
}


def max_sim_halos_for_run(quick_run: bool = QUICK_RUN) -> Optional[int]:
    """
    Row cap for the UNIT hlist according to ``quick_run``.

    Parameters
    ----------
    quick_run : bool, optional
        If True, use ``SMOKE_MAX_SIM_HALOS``; else ``MAX_SIM_HALOS``.

    Returns
    -------
    Optional[int]
        Maximum data rows to read, or ``None`` for the full catalog.
    """
    return SMOKE_MAX_SIM_HALOS if quick_run else MAX_SIM_HALOS


def max_fastpm_halos_for_run(quick_run: bool = QUICK_RUN) -> Optional[int]:
    """
    Halo cap for the FastPM ``.list`` according to ``quick_run``.

    Parameters
    ----------
    quick_run : bool, optional
        If True, use ``SMOKE_MAX_FASTPM_HALOS``; else ``MAX_FASTPM_HALOS``.

    Returns
    -------
    Optional[int]
        Maximum halos to read, or ``None`` for the full catalog.
    """
    return SMOKE_MAX_FASTPM_HALOS if quick_run else MAX_FASTPM_HALOS


def min_bin_size_for_run(quick_run: bool = QUICK_RUN) -> int:
    """
    Minimum halos per mass bin for Haloscope fits.

    Parameters
    ----------
    quick_run : bool, optional
        If True, use ``SMOKE_MIN_BIN_SIZE``; else ``PRODUCTION_MIN_BIN_SIZE``.

    Returns
    -------
    int
        Minimum bin population.
    """
    return SMOKE_MIN_BIN_SIZE if quick_run else PRODUCTION_MIN_BIN_SIZE


def default_sim_hlist_path() -> Path:
    """
    Default absolute path to the UNIT consistent-trees hlist at scale factor a=1 (z=0).

    Returns
    -------
    Path
        Path to ``hlist_1.00000.list.bz2``.
    """
    return SIM_ROCKSTAR_ROOT / SIM_HLIST_RELATIVE


def default_fastpm_list_path() -> Path:
    """
    Default absolute path to the FastPM Rockstar catalog file.

    Returns
    -------
    Path
        Path to ``FASTPM_LIST_NAME`` under ``FASTPM_ROCKSTAR_DIR``.
    """
    return FASTPM_ROCKSTAR_DIR / FASTPM_LIST_NAME
