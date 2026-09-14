"""Halo mass function utilities for Rockstar and FastPM catalogs."""

import bz2
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from density_field_properties.halo_catalog.fastpm import FASTPM_HALO_COLUMNS_POSITION
from density_field_properties.halo_catalog.rockstar import (
    ROCKSTAR_HALO_COLUMNS_POSITION,
    read_rockstar_box_size_header,
)
from density_field_properties.halo_catalog.utils import get_halo_catalog_reader
from density_field_properties.haloscope.sim_to_fastpm.config import (
    FASTPM_BOXSIZE_MPC_H,
    ROCKSTAR_LIST_COLUMNS,
    SIM_BOXSIZE_MPC_H,
    UNIT_HLIST_COLUMNS,
)
from density_field_properties.haloscope.sim_to_fastpm.load_catalogs import (
    _rockstar_data_column_count,
    _rockstar_pid_column_index,
)

LOG10 = np.log(10.0)
DEFAULT_LOG_MASS_MIN = 10.0
DEFAULT_LOG_MASS_MAX = 14.5
DEFAULT_N_BINS = 20
ROCKSTAR_OUT_MIN_COLUMNS = 34


@dataclass(frozen=True)
class HaloMassFunctionResult:
    """
    Binned halo mass function for one catalog.

    Attributes
    ----------
    log_mass_bin_edges : np.ndarray
        log10(M200b / Msun/h) bin edges, length ``n_bins + 1``.
    log_mass_bin_centers : np.ndarray
        Midpoints of ``log_mass_bin_edges`` in log10(Msun/h).
    counts : np.ndarray
        Halo counts per bin.
    dn_dlog10_m : np.ndarray
        Number density per dex in units h^3 Mpc^-3.
    dn_dln_m : np.ndarray
        Number density per natural logarithmic mass interval.
    box_size_mpc_h : float
        Simulation box side length in Mpc/h.
    volume_mpc_h3 : float
        Simulation volume in (Mpc/h)^3.
    n_halos : int
        Number of halos used after mass and host filters.
    """

    log_mass_bin_edges: np.ndarray
    log_mass_bin_centers: np.ndarray
    counts: np.ndarray
    dn_dlog10_m: np.ndarray
    dn_dln_m: np.ndarray
    box_size_mpc_h: float
    volume_mpc_h3: float
    n_halos: int


def default_hmf_log_mass_bin_edges(
    log_mass_min: float = DEFAULT_LOG_MASS_MIN,
    log_mass_max: float = DEFAULT_LOG_MASS_MAX,
    n_bins: int = DEFAULT_N_BINS,
) -> np.ndarray:
    """
    Build uniform log10(M200b) bin edges for a halo mass function.

    Parameters
    ----------
    log_mass_min : float, optional
        Lower edge in log10(Msun/h).
    log_mass_max : float, optional
        Upper edge in log10(Msun/h).
    n_bins : int, optional
        Number of bins between the edges.

    Returns
    -------
    np.ndarray
        Bin edges with length ``n_bins + 1``.
    """
    return np.linspace(log_mass_min, log_mass_max, n_bins + 1)


def compute_halo_mass_function(
    halo_masses_msun_h: np.ndarray,
    box_size_mpc_h: float,
    log_mass_bin_edges: Optional[np.ndarray] = None,
    min_mass_msun_h: float = 0.0,
) -> HaloMassFunctionResult:
    """
    Compute dn/dlog10(M) and dn/dln(M) from a sample of halo masses.

    Parameters
    ----------
    halo_masses_msun_h : np.ndarray
        Halo masses in Msun/h.
    box_size_mpc_h : float
        Simulation box side length in Mpc/h.
    log_mass_bin_edges : Optional[np.ndarray], optional
        log10(M200b) bin edges. Defaults to ``default_hmf_log_mass_bin_edges()``.
    min_mass_msun_h : float, optional
        Ignore halos with mass less than or equal to this threshold.

    Returns
    -------
    HaloMassFunctionResult
        Binned mass function and normalization metadata.

    Raises
    ------
    ValueError
        If no halos remain after filtering or the box size is not positive.
    """
    if box_size_mpc_h <= 0.0:
        raise ValueError("box_size_mpc_h must be positive")
    if log_mass_bin_edges is None:
        log_mass_bin_edges = default_hmf_log_mass_bin_edges()

    masses = np.asarray(halo_masses_msun_h, dtype=np.float64)
    masses = masses[np.isfinite(masses) & (masses > min_mass_msun_h)]
    if masses.size == 0:
        raise ValueError("No halos remain after applying the mass filter")

    log_mass = np.log10(masses)
    counts, _ = np.histogram(log_mass, bins=log_mass_bin_edges)
    counts = counts.astype(np.float64)
    log_mass_bin_centers = 0.5 * (log_mass_bin_edges[:-1] + log_mass_bin_edges[1:])

    delta_log10 = np.diff(log_mass_bin_edges)
    volume = box_size_mpc_h ** 3
    with np.errstate(divide="ignore", invalid="ignore"):
        dn_dlog10_m = counts / (delta_log10 * volume)
        dn_dln_m = dn_dlog10_m / LOG10

    return HaloMassFunctionResult(
        log_mass_bin_edges=log_mass_bin_edges,
        log_mass_bin_centers=log_mass_bin_centers,
        counts=counts,
        dn_dlog10_m=dn_dlog10_m,
        dn_dln_m=dn_dln_m,
        box_size_mpc_h=float(box_size_mpc_h),
        volume_mpc_h3=float(volume),
        n_halos=int(masses.size),
    )


def detect_catalog_reader_name(catalog_path: Path) -> str:
    """
    Infer the supported halo catalog reader from a path.

    Parameters
    ----------
    catalog_path : Path
        Rockstar ``.list`` file or FastPM BigFile halo directory.

    Returns
    -------
    str
        ``"rockstar"`` or ``"fastpm"``.

    Raises
    ------
    ValueError
        If the path type is not recognized.
    """
    path = Path(catalog_path)
    if path.is_file() and (path.suffix in {".list", ".bz2"} or path.name.endswith(".list.bz2")):
        return "rockstar"
    if path.is_dir():
        return "fastpm"
    raise ValueError(f"Unsupported halo catalog path: {catalog_path}")


def _rockstar_mass_and_pid_indices(list_path: Path) -> tuple[int, Optional[int]]:
    """
    Resolve Rockstar mass and optional PID column indices for one catalog.

    Parameters
    ----------
    list_path : Path
        Rockstar ``.list`` or ``.list.bz2`` path.

    Returns
    -------
    tuple[int, Optional[int]]
        Mass column index and optional PID column index.
    """
    if list_path.name.startswith("hlist_"):
        return UNIT_HLIST_COLUMNS["M200b"], UNIT_HLIST_COLUMNS["pid"]

    column_count = _rockstar_data_column_count(list_path)
    if column_count >= ROCKSTAR_OUT_MIN_COLUMNS:
        mass_index = ROCKSTAR_LIST_COLUMNS["halo_m200b"]
        pid_index = _rockstar_pid_column_index(list_path)
        if pid_index is not None and pid_index >= column_count:
            pid_index = ROCKSTAR_LIST_COLUMNS["pid"]
        if pid_index is not None and pid_index >= column_count:
            pid_index = None
        return mass_index, pid_index

    return ROCKSTAR_HALO_COLUMNS_POSITION["m200b"], None


def load_halo_masses_from_catalog(
    catalog_path: Path,
    catalog_name: Optional[str] = None,
    central_only: bool = True,
    min_mass_msun_h: float = 0.0,
    max_halos: Optional[int] = None,
) -> np.ndarray:
    """
    Load filtered halo masses from a Rockstar or FastPM catalog file.

    Parameters
    ----------
    catalog_path : Path
        Path to a Rockstar ``.list`` file or a FastPM halo BigFile directory.
    catalog_name : Optional[str], optional
        Explicit reader name (``"rockstar"`` or ``"fastpm"``). When omitted,
        the reader is inferred from ``catalog_path``.
    central_only : bool, optional
        Keep only host halos with ``PID == -1`` when a PID column exists.
    min_mass_msun_h : float, optional
        Ignore halos with mass less than or equal to this threshold.
    max_halos : Optional[int], optional
        Maximum number of accepted halos to read. ``None`` reads the full catalog.

    Returns
    -------
    np.ndarray
        Halo masses in Msun/h.

    Raises
    ------
    ValueError
        If no halos are found or the catalog type is unsupported.
    FileNotFoundError
        If ``catalog_path`` does not exist.
    """
    path = Path(catalog_path)
    if not path.exists():
        raise FileNotFoundError(f"Halo catalog not found: {path}")

    reader_name = catalog_name or detect_catalog_reader_name(path)
    if reader_name == "fastpm":
        reader = get_halo_catalog_reader("fastpm")
        catalog = reader.read_catalog(
            str(path),
            n_lines=max_halos,
            halo_id_block=FASTPM_HALO_COLUMNS_POSITION["halo_id"],
            halo_position_block=FASTPM_HALO_COLUMNS_POSITION["halo_position"],
            halo_length_block=FASTPM_HALO_COLUMNS_POSITION["halo_length"],
        )
        masses = catalog.halo_m200b
    elif reader_name == "rockstar":
        masses = _load_rockstar_halo_masses(
            path,
            central_only=central_only,
            min_mass_msun_h=min_mass_msun_h,
            max_halos=max_halos,
        )
        return masses
    else:
        raise ValueError(f"Unsupported catalog reader: {reader_name}")

    masses = np.asarray(masses, dtype=np.float64)
    masses = masses[np.isfinite(masses) & (masses > min_mass_msun_h)]
    if masses.size == 0:
        raise ValueError(f"No halos found in catalog {path}")
    return masses


def _load_rockstar_halo_masses(
    list_path: Path,
    central_only: bool,
    min_mass_msun_h: float,
    max_halos: Optional[int],
) -> np.ndarray:
    """
    Stream halo masses from a Rockstar text catalog.

    Parameters
    ----------
    list_path : Path
        Rockstar ``.list`` or ``.list.bz2`` path.
    central_only : bool
        Keep only rows with ``PID == -1`` when the column exists.
    min_mass_msun_h : float
        Ignore halos with mass less than or equal to this threshold.
    max_halos : Optional[int]
        Stop after this many accepted halos.

    Returns
    -------
    np.ndarray
        Accepted halo masses in Msun/h.
    """
    mass_index, pid_index = _rockstar_mass_and_pid_indices(list_path)
    if central_only and pid_index is None:
        logging.warning(
            "Rockstar catalog %s has no PID column; loading all halos with M200b > 0.",
            list_path,
        )

    opener = bz2.open if list_path.suffix == ".bz2" else open
    mode = "rt" if list_path.suffix == ".bz2" else "r"
    masses: list[float] = []
    with opener(list_path, mode) as handle:
        for line in handle:
            if line.startswith("#"):
                continue
            columns = line.split()
            if mass_index >= len(columns):
                continue
            if central_only and pid_index is not None:
                if pid_index >= len(columns):
                    continue
                if int(float(columns[pid_index])) != -1:
                    continue
            mass = float(columns[mass_index])
            if mass <= min_mass_msun_h:
                continue
            masses.append(mass)
            if max_halos is not None and len(masses) >= max_halos:
                break

    if not masses:
        raise ValueError(f"No halos found in catalog {list_path}")
    return np.asarray(masses, dtype=np.float64)


def resolve_catalog_box_size_mpc_h(
    catalog_path: Path,
    catalog_name: Optional[str] = None,
    box_size_mpc_h: Optional[float] = None,
    default_box_size_mpc_h: float = SIM_BOXSIZE_MPC_H,
) -> float:
    """
    Resolve the simulation box size for a halo catalog.

    Parameters
    ----------
    catalog_path : Path
        Catalog path used to read optional Rockstar headers.
    catalog_name : Optional[str], optional
        Explicit reader name. When omitted, it is inferred from ``catalog_path``.
    box_size_mpc_h : Optional[float], optional
        User-provided override in Mpc/h.
    default_box_size_mpc_h : float, optional
        Fallback box size when no header value is available.

    Returns
    -------
    float
        Box side length in Mpc/h.
    """
    if box_size_mpc_h is not None:
        return float(box_size_mpc_h)

    reader_name = catalog_name or detect_catalog_reader_name(catalog_path)
    if reader_name == "rockstar":
        header_box = read_rockstar_box_size_header(str(catalog_path))
        if header_box is not None:
            return float(header_box)
    if reader_name == "fastpm":
        return float(FASTPM_BOXSIZE_MPC_H)
    return float(default_box_size_mpc_h)


def halo_mass_function_from_catalog(
    catalog_path: Path,
    catalog_name: Optional[str] = None,
    box_size_mpc_h: Optional[float] = None,
    central_only: bool = True,
    min_mass_msun_h: float = 0.0,
    max_halos: Optional[int] = None,
    log_mass_bin_edges: Optional[np.ndarray] = None,
) -> HaloMassFunctionResult:
    """
    Compute the halo mass function for one catalog file.

    Parameters
    ----------
    catalog_path : Path
        Rockstar ``.list`` file or FastPM halo BigFile directory.
    catalog_name : Optional[str], optional
        Explicit reader name (``"rockstar"`` or ``"fastpm"``).
    box_size_mpc_h : Optional[float], optional
        Simulation box side length in Mpc/h. When omitted, headers or defaults
        are used.
    central_only : bool, optional
        Keep only host halos with ``PID == -1`` for Rockstar catalogs.
    min_mass_msun_h : float, optional
        Ignore halos with mass less than or equal to this threshold.
    max_halos : Optional[int], optional
        Maximum number of halos to read.
    log_mass_bin_edges : Optional[np.ndarray], optional
        log10(M200b) bin edges passed to ``compute_halo_mass_function``.

    Returns
    -------
    HaloMassFunctionResult
        Binned halo mass function for the requested catalog.
    """
    path = Path(catalog_path)
    reader_name = catalog_name or detect_catalog_reader_name(path)
    resolved_box_size = resolve_catalog_box_size_mpc_h(
        path,
        catalog_name=reader_name,
        box_size_mpc_h=box_size_mpc_h,
    )
    masses = load_halo_masses_from_catalog(
        path,
        catalog_name=reader_name,
        central_only=central_only,
        min_mass_msun_h=min_mass_msun_h,
        max_halos=max_halos,
    )
    return compute_halo_mass_function(
        masses,
        resolved_box_size,
        log_mass_bin_edges=log_mass_bin_edges,
        min_mass_msun_h=min_mass_msun_h,
    )
