"""Paths and hyperparameters for SIM-to-FastPM Haloscope."""

from pathlib import Path
from typing import Optional

SIM_ROCKSTAR_ROOT = Path("/data5/UNITSIM/fixedAmp_InvPhase_001/ROCKSTAR/outputs")
SIM_HLIST_RELATIVE = Path("hlists/hlist_1.00000.list.bz2")
SIM_BOXSIZE_MPC_H = 1000.0

FASTPM_ROCKSTAR_DIR = Path("/data21/users/mruiz/fastpm_MN5/fastpm_tfm/rockstar_out_pm")
FASTPM_LIST_NAME = "out_8.list"
FASTPM_BOXSIZE_MPC_H = 1000.0

UNIT_ROCKSTAR_DIR = Path("/data21/UNITSIM/fixedAmp_InvPhase_001/ROCKSTAR")
UNIT_ROCKSTAR_LIST_NAME = "out_128p.list.bz2"

ENV_RADIUS_MPC_H = 5.0
CALIBRATE_MASS = True

DM_MASS_PARTICLE_MSUN_H = 1.2e9
FASTPM_DM_PARTICLES_PATH = Path(
    "/data21/users/mruiz/fastpm_MN5/fastpm_tfm/output_01/snap_1.0000/1"
)
FASTPM_SAVED_CIC_DENSITY = Path("output/fast_pm_bigfile/snap_1.0000_density")
FASTPM_SAVED_CIC_DENSITY_INFO = Path("output/fast_pm_bigfile/snap_1.0000_density_info.txt")
SIM_SAVED_CIC_DENSITY = Path("output/unit_files/dm_particles_0.5_128_density")
SIM_SAVED_CIC_DENSITY_INFO = Path("output/unit_files/dm_particles_0.5_128_density_info.txt")
SIM_DM_PARTICLES_PATH = Path("output/unit_files/dm_particles_0.5_128")
ASSEMBLY_BIAS_DM_BATCH_SIZE: Optional[int] = None

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
    "desc_id": 1,
    "pid": 33,
    "halo_x": 8,
    "halo_y": 9,
    "halo_z": 10,
    "halo_m200b": 20,
}

EXTENDED_ROCKSTAR_MIN_COLUMNS = 55
ROCKSTAR_RESERVOIR_SEED = 42

UNIT_ROCKSTAR_LIST_COLUMNS = {
    "halo_id": 0,
    "pid": 33,
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


def default_sim_saved_cic_density_paths() -> tuple[Path, Path]:
    """
    Relative paths to the precomputed UNIT SIM CIC density under ``output/unit_files/``.

    Returns
    -------
    tuple[Path, Path]
        ``(density_binary, density_info_txt)`` from ``main_density_field_cic``.
    """
    return SIM_SAVED_CIC_DENSITY, SIM_SAVED_CIC_DENSITY_INFO


def default_fastpm_saved_cic_density_paths() -> tuple[Path, Path]:
    """
    Relative paths to the precomputed FastPM CIC density under the repo ``output/`` tree.

    Returns
    -------
    tuple[Path, Path]
        ``(density_binary, density_info_txt)`` as produced by ``main_density_field_cic``.
    """
    return FASTPM_SAVED_CIC_DENSITY, FASTPM_SAVED_CIC_DENSITY_INFO


def default_fastpm_dm_particles_path() -> Path:
    """
    Default FastPM BigFile block path for DM at ``snap_1.0000``.

    Returns
    -------
    Path
        Block directory under the FastPM ``output_*`` snapshot.
    """
    return FASTPM_DM_PARTICLES_PATH


def default_sim_dm_particles_path() -> Optional[Path]:
    """
    UNIT SIM DM particle file for live CIC (relative to repo ``output/unit_files/``).

    Returns
    -------
    Optional[Path]
        Path to DM positions when configured; ``None`` skips the DM fallback.
    """
    return SIM_DM_PARTICLES_PATH


def default_fastpm_list_path() -> Path:
    """
    Default absolute path to the FastPM Rockstar catalog file.

    Returns
    -------
    Path
        Path to ``FASTPM_LIST_NAME`` under ``FASTPM_ROCKSTAR_DIR``.
    """
    return FASTPM_ROCKSTAR_DIR / FASTPM_LIST_NAME


def default_unit_rockstar_list_path() -> Path:
    """
    Default absolute path to the UNIT Rockstar catalog at ``a = 1``.

    Returns
    -------
    Path
        Path to ``out_128p.list.bz2`` under ``UNIT_ROCKSTAR_DIR``.
    """
    return UNIT_ROCKSTAR_DIR / UNIT_ROCKSTAR_LIST_NAME
