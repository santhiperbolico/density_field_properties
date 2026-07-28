"""Assembly-bias diagnostics (HALOSCOPE paper style) for SIM-to-FastPM runs."""

from pathlib import Path
from typing import Optional, Sequence, Tuple, Union

import numpy as np
from numpy.fft import irfftn, rfftn
from scipy import stats

from density_field_properties.density_field.cic_deposit import (
    delta_field_from_dm_particles,
    delta_field_from_saved_cic,
    overdensity_from_cic_grid,
    weighted_field_cic,
)
from density_field_properties.density_field.fourrier_transformations import kgrid
from density_field_properties.density_field.particle_io import detect_dm_particle_format
from density_field_properties.haloscope.sim_to_fastpm.config import (
    OUTPUT_FEATURES,
    default_fastpm_saved_cic_density_paths,
    default_sim_dm_particles_path,
    default_sim_saved_cic_density_paths,
)


def solve_joint_percentile(
    properties: np.ndarray,
    target_fraction: float = 0.25,
    tail: str = "lower",
) -> float:
    """
    Find the per-property percentile that selects ``target_fraction`` of haloes jointly.

    Following Ramakrishnan et al. (2025), lower (upper) tails require every property
    to lie below (above) its ``p`` (``100 - p``) percentile.

    Parameters
    ----------
    properties : np.ndarray
        Shape ``(n_halos, n_properties)``.
    target_fraction : float, optional
        Desired fraction of the sample in the joint tail.
    tail : str, optional
        ``"lower"`` or ``"upper"``.

    Returns
    -------
    float
        Percentile ``p`` used for the joint tail (can exceed 50 when several properties are used).
    """
    if properties.ndim != 2:
        raise ValueError("properties must have shape (n_halos, n_properties)")
    if tail not in ("lower", "upper"):
        raise ValueError("tail must be 'lower' or 'upper'")
    if not 0.0 < target_fraction < 0.5:
        raise ValueError("target_fraction must be between 0 and 0.5")

    def selected_fraction(percentile: float) -> float:
        if tail == "lower":
            thresholds = np.percentile(properties, percentile, axis=0)
            mask = np.all(properties < thresholds, axis=1)
        else:
            thresholds = np.percentile(properties, 100.0 - percentile, axis=0)
            mask = np.all(properties > thresholds, axis=1)
        return float(mask.mean())

    low = 1e-6
    high = 99.99
    for _ in range(64):
        mid = 0.5 * (low + high)
        if selected_fraction(mid) < target_fraction:
            low = mid
        else:
            high = mid
    return 0.5 * (low + high)


def joint_assembly_masks(
    properties: np.ndarray,
    target_fraction: float = 0.25,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Boolean masks for lower and upper joint assembly-bias subsamples.

    Parameters
    ----------
    properties : np.ndarray
        Halo properties with shape ``(n_halos, n_properties)``.
    target_fraction : float, optional
        Target fraction for each tail (default 25%).

    Returns
    -------
    tuple[np.ndarray, np.ndarray, float]
        ``lower_mask``, ``upper_mask``, and the solved lower-tail percentile ``p``.
    """
    percentile = solve_joint_percentile(properties, target_fraction=target_fraction, tail="lower")
    lower_thresholds = np.percentile(properties, percentile, axis=0)
    upper_thresholds = np.percentile(properties, 100.0 - percentile, axis=0)
    lower_mask = np.all(properties < lower_thresholds, axis=1)
    upper_mask = np.all(properties > upper_thresholds, axis=1)
    return lower_mask, upper_mask, percentile


def halo_overdensity_field_cic(
    positions_mpc_h: np.ndarray,
    weights: np.ndarray,
    boxsize_mpc_h: float,
    n_grid: int,
) -> np.ndarray:
    """
    Build a mass-weighted overdensity field with internal CIC deposition.

    Parameters
    ----------
    positions_mpc_h : np.ndarray
        Halo positions with shape ``(N, 3)`` in Mpc/h.
    weights : np.ndarray
        Non-negative weights per halo (typically ``M200b``).
    boxsize_mpc_h : float
        Periodic box side length in Mpc/h.
    n_grid : int
        Cells per dimension.

    Returns
    -------
    np.ndarray
        Overdensity ``delta`` on a cubic grid of shape ``(n_grid, n_grid, n_grid)``.
    """
    cic_grid = weighted_field_cic(
        positions_mpc_h.astype(np.float64),
        weights.astype(np.float64),
        boxsize_mpc_h,
        n_grid,
    )
    return overdensity_from_cic_grid(cic_grid)


def halo_weighted_overdensity_field(
    positions_mpc_h: np.ndarray,
    weights: np.ndarray,
    boxsize_mpc_h: float,
    n_grid: int,
) -> np.ndarray:
    """
    Build a mass-weighted overdensity field on a uniform grid (CIC).

    Parameters
    ----------
    positions_mpc_h : np.ndarray
        Halo positions with shape ``(N, 3)`` in Mpc/h.
    weights : np.ndarray
        Non-negative weights per halo (typically ``M200b``).
    boxsize_mpc_h : float
        Periodic box side length in Mpc/h.
    n_grid : int
        Cells per dimension.

    Returns
    -------
    np.ndarray
        Overdensity ``delta`` on a cubic grid of shape ``(n_grid, n_grid, n_grid)``.
    """
    return halo_overdensity_field_cic(positions_mpc_h, weights, boxsize_mpc_h, n_grid)


def matter_overdensity_field_from_dm(
    dm_particles_path: Union[str, Path],
    boxsize_mpc_h: float,
    n_grid: int,
    mass_particle_msun_h: float,
    batch_size: Optional[int] = None,
) -> np.ndarray:
    """
    Matter overdensity ``delta`` from a DM particle snapshot (text or FastPM BigFile).

    Parameters
    ----------
    dm_particles_path : str or Path
        Particle file or FastPM block directory.
    boxsize_mpc_h : float
        Box side length in Mpc/h.
    n_grid : int
        Grid resolution.
    mass_particle_msun_h : float
        DM particle mass in Msun/h.
    batch_size : Optional[int], optional
        I/O batch size; ``None`` loads all particles in one batch.

    Returns
    -------
    np.ndarray
        Cubic overdensity grid.

    Raises
    ------
    ValueError
        If the path is missing or not a supported particle layout.
    """
    path = Path(dm_particles_path)
    if not path.exists():
        raise ValueError(f"DM particle path does not exist: {path}")
    detect_dm_particle_format(str(path))
    return delta_field_from_dm_particles(
        str(path),
        mass_particle=mass_particle_msun_h,
        box_size=boxsize_mpc_h,
        n_grid=n_grid,
        batch_size=batch_size,
    )


def matter_overdensity_from_saved_cic(
    density_binary_path: Union[str, Path],
    density_info_path: Union[str, Path],
    expected_boxsize_mpc_h: Optional[float] = None,
) -> np.ndarray:
    """
    Load matter ``delta`` from a precomputed CIC density binary and info file.

    Parameters
    ----------
    density_binary_path : str or Path
        Path to ``*_density`` binary grid.
    density_info_path : str or Path
        Path to ``*_density_info.txt``.
    expected_boxsize_mpc_h : Optional[float], optional
        If set, validate against metadata ``box_size``.

    Returns
    -------
    np.ndarray
        Matter overdensity field.
    """
    return delta_field_from_saved_cic(
        str(density_binary_path),
        str(density_info_path),
        expected_box_size=expected_boxsize_mpc_h,
    )


def load_matter_overdensity(
    repo_root: Union[str, Path],
    boxsize_mpc_h: float,
    n_grid: int,
    dm_mass_particle_msun_h: float,
    dm_particles_path: Optional[Union[str, Path]],
    dm_batch_size: Optional[int] = None,
    saved_density_path: Optional[Union[str, Path]] = None,
    saved_density_info_path: Optional[Union[str, Path]] = None,
) -> Tuple[Optional[np.ndarray], str]:
    """
    Resolve matter ``delta``: saved CIC grid, else DM snapshot, else unavailable.

    Parameters
    ----------
    repo_root : str or Path
        Repository root used to resolve relative paths.
    boxsize_mpc_h : float
        Expected box size in Mpc/h.
    n_grid : int
        Grid size when building from DM particles.
    dm_mass_particle_msun_h : float
        DM particle mass for the live CIC build.
    dm_particles_path : Optional[str or Path]
        Fallback DM particle file or BigFile block; ``None`` skips DM.
    dm_batch_size : Optional[int], optional
        DM I/O batch size.
    saved_density_path : Optional[str or Path], optional
        Binary density file (relative to ``repo_root`` if not absolute).
    saved_density_info_path : Optional[str or Path], optional
        Density metadata file.

    Returns
    -------
    tuple[Optional[np.ndarray], str]
        ``(delta_field or None, mode_label)`` — ``saved CIC``, ``DM+CIC``, or ``unavailable``.
    """
    root = Path(repo_root)
    if saved_density_path is not None and saved_density_info_path is not None:
        density_file = Path(saved_density_path)
        info_file = Path(saved_density_info_path)
        if not density_file.is_absolute():
            density_file = root / density_file
        if not info_file.is_absolute():
            info_file = root / info_file

        if density_file.is_file() and info_file.is_file():
            return (
                matter_overdensity_from_saved_cic(
                    density_file,
                    info_file,
                    expected_boxsize_mpc_h=boxsize_mpc_h,
                ),
                "saved CIC",
            )

    if dm_particles_path is not None:
        dm_path = Path(dm_particles_path)
        if not dm_path.is_absolute():
            dm_path = root / dm_path
        if dm_path.exists():
            return (
                matter_overdensity_field_from_dm(
                    dm_path,
                    boxsize_mpc_h,
                    n_grid,
                    dm_mass_particle_msun_h,
                    batch_size=dm_batch_size,
                ),
                "DM+CIC",
            )

    return None, "unavailable"


def load_sim_matter_overdensity(
    repo_root: Union[str, Path],
    boxsize_mpc_h: float,
    n_grid: int,
    dm_mass_particle_msun_h: float,
    dm_batch_size: Optional[int] = None,
    dm_particles_path: Optional[Union[str, Path]] = None,
    saved_density_path: Optional[Union[str, Path]] = None,
    saved_density_info_path: Optional[Union[str, Path]] = None,
) -> Tuple[Optional[np.ndarray], str]:
    """
    UNIT SIM matter ``delta`` (saved CIC under ``output/unit_files/``, then DM file).

    Parameters
    ----------
    repo_root : str or Path
        Repository root.
    boxsize_mpc_h : float
        Box size in Mpc/h.
    n_grid : int
        Grid size for live DM CIC.
    dm_mass_particle_msun_h : float
        DM particle mass.
    dm_batch_size : Optional[int], optional
        DM batch size.
    dm_particles_path : Optional[str or Path], optional
        Override DM path; defaults to ``config.SIM_DM_PARTICLES_PATH``.
    saved_density_path : Optional[str or Path], optional
        Override saved density binary.
    saved_density_info_path : Optional[str or Path], optional
        Override saved density info file.

    Returns
    -------
    tuple[Optional[np.ndarray], str]
        Matter delta and mode label.
    """
    if saved_density_path is None or saved_density_info_path is None:
        saved_density_path, saved_density_info_path = default_sim_saved_cic_density_paths()
    if dm_particles_path is None:
        dm_particles_path = default_sim_dm_particles_path()
    return load_matter_overdensity(
        repo_root,
        boxsize_mpc_h,
        n_grid,
        dm_mass_particle_msun_h,
        dm_particles_path,
        dm_batch_size=dm_batch_size,
        saved_density_path=saved_density_path,
        saved_density_info_path=saved_density_info_path,
    )


def load_fastpm_matter_overdensity(
    repo_root: Union[str, Path],
    boxsize_mpc_h: float,
    n_grid: int,
    dm_mass_particle_msun_h: float,
    dm_particles_path: Union[str, Path],
    dm_batch_size: Optional[int] = None,
    saved_density_path: Optional[Union[str, Path]] = None,
    saved_density_info_path: Optional[Union[str, Path]] = None,
) -> Tuple[Optional[np.ndarray], str]:
    """
    FastPM matter ``delta`` (saved CIC under ``output/fast_pm_bigfile/``, then BigFile DM).

    Parameters
    ----------
    repo_root : str or Path
        Repository root.
    boxsize_mpc_h : float
        Box size in Mpc/h.
    n_grid : int
        Grid size for live DM CIC.
    dm_mass_particle_msun_h : float
        DM particle mass.
    dm_particles_path : str or Path
        Fallback FastPM BigFile block path.
    dm_batch_size : Optional[int], optional
        DM batch size.
    saved_density_path : Optional[str or Path], optional
        Override saved density binary.
    saved_density_info_path : Optional[str or Path], optional
        Override saved density info file.

    Returns
    -------
    tuple[Optional[np.ndarray], str]
        Matter delta and mode label.
    """
    if saved_density_path is None or saved_density_info_path is None:
        saved_density_path, saved_density_info_path = default_fastpm_saved_cic_density_paths()
    return load_matter_overdensity(
        repo_root,
        boxsize_mpc_h,
        n_grid,
        dm_mass_particle_msun_h,
        dm_particles_path,
        dm_batch_size=dm_batch_size,
        saved_density_path=saved_density_path,
        saved_density_info_path=saved_density_info_path,
    )


def paranjape_halo_by_halo_bias(
    positions_mpc_h: np.ndarray,
    delta_field: np.ndarray,
    boxsize_mpc_h: float,
    k_max_h_mpc: float = 0.1,
) -> np.ndarray:
    """
    Low-k halo-by-halo linear bias (Paranjape et al. 2018; Eq. 3 in HALOSCOPE paper).

    Parameters
    ----------
    positions_mpc_h : np.ndarray
        Halo positions with shape ``(N, 3)`` in Mpc/h.
    delta_field : np.ndarray
        Real-space matter overdensity on a cubic grid.
    boxsize_mpc_h : float
        Box side length in Mpc/h.
    k_max_h_mpc : float, optional
        Maximum wavenumber in ``h Mpc^-1`` (default 0.1).

    Returns
    -------
    np.ndarray
        Per-halo linear bias estimates, shape ``(N,)``.
    """
    n_grid = delta_field.shape[0]
    if delta_field.shape != (n_grid, n_grid, n_grid):
        raise ValueError("delta_field must be a cubic 3D array")

    volume = boxsize_mpc_h**3
    delta_k = rfftn(delta_field) / n_grid**3
    kx, ky, kz = kgrid(n_grid, boxsize_mpc_h)
    k_mag = np.sqrt(kx**2 + ky**2 + kz**2)
    low_k = (k_mag > 0.0) & (k_mag <= k_max_h_mpc)
    if not np.any(low_k):
        raise ValueError("No Fourier modes below k_max; increase n_grid or k_max.")

    power = np.abs(delta_k) ** 2 * volume
    safe_power = np.maximum(power, 1e-30)
    weighted_delta_k = np.zeros_like(delta_k, dtype=np.complex128)
    weighted_delta_k[low_k] = delta_k[low_k] / safe_power[low_k]

    bias_field = irfftn(weighted_delta_k, s=(n_grid, n_grid, n_grid), axes=(0, 1, 2)).real * n_grid**3
    return _trilinear_sample_field(positions_mpc_h, bias_field, boxsize_mpc_h)


def _trilinear_sample_field(
    positions_mpc_h: np.ndarray,
    field: np.ndarray,
    boxsize_mpc_h: float,
) -> np.ndarray:
    """
    Trilinear interpolation of a periodic 3D field at halo positions.

    Parameters
    ----------
    positions_mpc_h : np.ndarray
        Positions with shape ``(N, 3)``.
    field : np.ndarray
        Cubic grid values.
    boxsize_mpc_h : float
        Box side length.

    Returns
    -------
    np.ndarray
        Sampled values, shape ``(N,)``.
    """
    n_grid = field.shape[0]
    cell = boxsize_mpc_h / n_grid
    wrapped = np.mod(positions_mpc_h, boxsize_mpc_h)
    grid_coord = wrapped / cell
    i0 = np.floor(grid_coord).astype(int) % n_grid
    frac = grid_coord - np.floor(grid_coord)

    samples = np.zeros(len(positions_mpc_h), dtype=np.float64)
    for di in (0, 1):
        for dj in (0, 1):
            for dk in (0, 1):
                weight = (
                    (frac[:, 0] if di else 1.0 - frac[:, 0])
                    * (frac[:, 1] if dj else 1.0 - frac[:, 1])
                    * (frac[:, 2] if dk else 1.0 - frac[:, 2])
                )
                index_x = (i0[:, 0] + di) % n_grid
                index_y = (i0[:, 1] + dj) % n_grid
                index_z = (i0[:, 2] + dk) % n_grid
                samples += weight * field[index_x, index_y, index_z]
    return samples


def central_68_scatter(values: np.ndarray) -> float:
    """
    Half-width of the central 68.3% interval (1-sigma for a Gaussian).

    Parameters
    ----------
    values : np.ndarray
        Sample in one mass bin.

    Returns
    -------
    float
        ``(P84.15 - P15.85) / 2``, or ``nan`` for empty input.
    """
    if values.size == 0:
        return float("nan")
    upper = np.percentile(values, 84.15)
    lower = np.percentile(values, 15.85)
    return 0.5 * (upper - lower)


def binned_mean_bias_with_scatter(
    mass: np.ndarray,
    bias: np.ndarray,
    log_mass_bin_edges: np.ndarray,
    mask: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Mean halo bias and 68% scatter per log10-mass bin.

    Parameters
    ----------
    mass : np.ndarray
        Halo masses (linear ``M200b`` in Msun/h).
    bias : np.ndarray
        Per-halo linear bias.
    log_mass_bin_edges : np.ndarray
        Bin edges in ``log10(M200b)``.
    mask : np.ndarray
        Boolean mask selecting haloes for this subsample.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        Mass bin centers, mean ``b_1``, and central 68% half-width per bin.
    """
    log_mass = np.log10(mass)
    selected_mass = log_mass[mask]
    selected_bias = bias[mask]
    mass_centers = stats.binned_statistic(
        selected_mass,
        selected_mass,
        statistic="mean",
        bins=log_mass_bin_edges,
    )[0]
    mean_bias = stats.binned_statistic(
        selected_mass,
        selected_bias,
        statistic="mean",
        bins=log_mass_bin_edges,
    )[0]
    scatter = stats.binned_statistic(
        selected_mass,
        selected_bias,
        statistic=central_68_scatter,
        bins=log_mass_bin_edges,
    )[0]
    geometric_mass = 10**mass_centers
    valid = np.isfinite(geometric_mass) & np.isfinite(mean_bias)
    return geometric_mass[valid], mean_bias[valid], scatter[valid]


def assembly_bias_curves_for_catalog(
    mass: np.ndarray,
    bias: np.ndarray,
    property_matrix: np.ndarray,
    log_mass_bin_edges: np.ndarray,
    target_fraction: float = 0.25,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Upper and lower assembly-bias ``b_1(M)`` curves for one halo catalog.

    Parameters
    ----------
    mass : np.ndarray
        Linear halo masses.
    bias : np.ndarray
        Per-halo linear bias.
    property_matrix : np.ndarray
        Properties used for joint ranking, shape ``(n_halos, n_properties)``.
    log_mass_bin_edges : np.ndarray
        Bin edges in ``log10(M200b)``.
    target_fraction : float, optional
        Joint tail fraction (default 25%).

    Returns
    -------
    tuple
        ``lower_mass``, ``lower_bias``, ``lower_scatter``, ``upper_mass``, ``upper_bias``,
        ``upper_scatter``.
    """
    lower_mask, upper_mask, _ = joint_assembly_masks(property_matrix, target_fraction)
    lower_mass, lower_bias, lower_scatter = binned_mean_bias_with_scatter(
        mass, bias, log_mass_bin_edges, lower_mask
    )
    upper_mass, upper_bias, upper_scatter = binned_mean_bias_with_scatter(
        mass, bias, log_mass_bin_edges, upper_mask
    )
    return lower_mass, lower_bias, lower_scatter, upper_mass, upper_bias, upper_scatter


def attach_paranjape_bias(
    frame,
    boxsize_mpc_h: float,
    n_grid: int = 128,
    mass_column: str = "M200b",
    k_max_h_mpc: float = 0.1,
    position_columns: Sequence[str] = ("x", "y", "z"),
    dm_particles_path: Optional[Union[str, Path]] = None,
    dm_mass_particle_msun_h: Optional[float] = None,
    dm_batch_size: Optional[int] = None,
    matter_delta_field: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Compute per-halo linear bias with the Paranjape estimator.

    Uses a matter ``delta`` field from DM particles when ``dm_particles_path`` or
    ``matter_delta_field`` is provided; otherwise falls back to halo CIC with ``mass_column``
    weights.

    Parameters
    ----------
    frame : pandas.DataFrame
        Halo catalog with positions and masses.
    boxsize_mpc_h : float
        Periodic box side length in Mpc/h.
    n_grid : int, optional
        Grid resolution for the overdensity field.
    mass_column : str, optional
        Mass column used as CIC weights in the halo-only fallback.
    k_max_h_mpc : float, optional
        Maximum wavenumber for the Paranjape estimator.
    position_columns : Sequence[str], optional
        Column names for halo positions.
    dm_particles_path : Optional[str or Path], optional
        DM snapshot (text or FastPM BigFile block). Ignored if ``matter_delta_field`` is set.
    dm_mass_particle_msun_h : Optional[float], optional
        DM particle mass; required when using ``dm_particles_path``.
    dm_batch_size : Optional[int], optional
        Particle batch size for DM I/O.
    matter_delta_field : Optional[np.ndarray], optional
        Precomputed matter overdensity grid (e.g. cached from an earlier DM run).

    Returns
    -------
    np.ndarray
        Per-halo bias values aligned with ``frame`` rows.
    """
    positions = frame[list(position_columns)].to_numpy(dtype=np.float64)
    if matter_delta_field is not None:
        delta_field = matter_delta_field
    elif dm_particles_path is not None:
        if dm_mass_particle_msun_h is None:
            raise ValueError("dm_mass_particle_msun_h is required when dm_particles_path is set")
        delta_field = matter_overdensity_field_from_dm(
            dm_particles_path,
            boxsize_mpc_h,
            n_grid,
            dm_mass_particle_msun_h,
            batch_size=dm_batch_size,
        )
    else:
        weights = frame[mass_column].to_numpy(dtype=np.float64)
        delta_field = halo_overdensity_field_cic(positions, weights, boxsize_mpc_h, n_grid)
    return paranjape_halo_by_halo_bias(
        positions,
        delta_field,
        boxsize_mpc_h,
        k_max_h_mpc=k_max_h_mpc,
    )


def property_matrix_from_frame(frame, columns: Sequence[str] = OUTPUT_FEATURES) -> np.ndarray:
    """
    Stack halo property columns into a 2D array for assembly-bias splits.

    Parameters
    ----------
    frame : pandas.DataFrame
        Halo table containing ``columns``.
    columns : Sequence[str], optional
        Property names (defaults to Haloscope outputs).

    Returns
    -------
    np.ndarray
        Array with shape ``(len(frame), len(columns))``.
    """
    return frame[list(columns)].to_numpy(dtype=np.float64)
