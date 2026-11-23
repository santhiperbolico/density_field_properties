"""Assembly-bias diagnostics (HALOSCOPE paper style)."""

from pathlib import Path
from typing import Optional, Sequence, Tuple, Union

import numpy as np

from density_field_properties.environment_properties.cic.cic_deposit import (
    delta_field_from_dm_particles,
    delta_field_from_saved_cic,
)
from density_field_properties.read_data.particles import detect_dm_particle_format
from density_field_properties.utils.stats import standard_error_mean
from density_field_properties.validation.haloscope_halo_bias import (
    haloscope_halo_by_halo_bias as paranjape_halo_by_halo_bias,
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


def joint_assembly_masks_from_reference(
    properties: np.ndarray,
    reference_properties: np.ndarray,
    target_fraction: float = 0.25,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Joint tail masks using percentile thresholds from a reference property sample.

    Parameters
    ----------
    properties : np.ndarray
        Halo properties with shape ``(n_halos, n_properties)`` to classify.
    reference_properties : np.ndarray
        Reference sample used to define tail thresholds in each property.
    target_fraction : float, optional
        Target fraction for each tail (default 25%).

    Returns
    -------
    tuple[np.ndarray, np.ndarray, float]
        ``lower_mask``, ``upper_mask``, and the solved lower-tail percentile ``p``.
    """
    percentile = solve_joint_percentile(
        reference_properties,
        target_fraction=target_fraction,
        tail="lower",
    )
    lower_thresholds = np.percentile(reference_properties, percentile, axis=0)
    upper_thresholds = np.percentile(reference_properties, 100.0 - percentile, axis=0)
    lower_mask = np.all(properties < lower_thresholds, axis=1)
    upper_mask = np.all(properties > upper_thresholds, axis=1)
    return lower_mask, upper_mask, percentile


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


def _geometric_mass_bin_center(log_mass_low: float, log_mass_high: float) -> float:
    """
    Geometric center of a log10-mass bin.

    Parameters
    ----------
    log_mass_low : float
        Lower log10-mass edge.
    log_mass_high : float
        Upper log10-mass edge.

    Returns
    -------
    float
        ``M`` at the midpoint of the bin in linear mass.
    """
    return 10 ** (0.5 * (log_mass_low + log_mass_high))


def assembly_bias_curves_for_catalog(
    mass: np.ndarray,
    bias: np.ndarray,
    property_matrix: np.ndarray,
    log_mass_bin_edges: np.ndarray,
    target_fraction: float = 0.25,
    threshold_mass: Optional[np.ndarray] = None,
    threshold_property_matrix: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Upper and lower assembly-bias ``b_1(M)`` curves for one halo catalog.

    Following Ramakrishnan et al. (2025), joint upper and lower tails are
    selected independently within each equal-mass bin before averaging ``b_1``.
    The returned error bands are the standard error of the mean in each bin.

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
        Joint tail fraction within each mass bin (default 25%).
    threshold_mass : Optional[np.ndarray], optional
        Linear masses for ``threshold_property_matrix`` when tail cuts are defined
        from a reference catalog (e.g. HR truth for LR+HALOSCOPE curves).
    threshold_property_matrix : Optional[np.ndarray], optional
        Reference properties aligned with ``threshold_mass``. When set together with
        ``threshold_mass``, tail thresholds in each bin come from this reference sample
        and are applied to ``property_matrix``.

    Returns
    -------
    tuple
        ``lower_mass``, ``lower_bias``, ``lower_sem``, ``upper_mass``, ``upper_bias``,
        ``upper_sem``.
    """
    use_reference_thresholds = threshold_mass is not None and threshold_property_matrix is not None
    if use_reference_thresholds and len(threshold_mass) != len(threshold_property_matrix):
        raise ValueError("threshold_mass and threshold_property_matrix must have equal length")
    log_mass = np.log10(mass)
    threshold_log_mass = np.log10(threshold_mass) if use_reference_thresholds else None
    lower_mass = []
    lower_bias = []
    lower_sem = []
    upper_mass = []
    upper_bias = []
    upper_sem = []

    for bin_index in range(len(log_mass_bin_edges) - 1):
        log_mass_low = log_mass_bin_edges[bin_index]
        log_mass_high = log_mass_bin_edges[bin_index + 1]
        in_bin = (log_mass >= log_mass_low) & (log_mass < log_mass_high)
        if not np.any(in_bin):
            continue

        bin_bias = bias[in_bin]
        bin_properties = property_matrix[in_bin]
        if use_reference_thresholds:
            reference_in_bin = (threshold_log_mass >= log_mass_low) & (
                threshold_log_mass < log_mass_high
            )
            if not np.any(reference_in_bin):
                continue
            reference_properties = threshold_property_matrix[reference_in_bin]
            lower_mask, upper_mask, _ = joint_assembly_masks_from_reference(
                bin_properties,
                reference_properties,
                target_fraction=target_fraction,
            )
        else:
            lower_mask, upper_mask, _ = joint_assembly_masks(
                bin_properties,
                target_fraction=target_fraction,
            )
        lower_values = bin_bias[lower_mask]
        upper_values = bin_bias[upper_mask]
        if lower_values.size == 0 or upper_values.size == 0:
            continue

        mass_center = _geometric_mass_bin_center(log_mass_low, log_mass_high)
        lower_mass.append(mass_center)
        lower_bias.append(float(np.mean(lower_values)))
        lower_sem.append(standard_error_mean(lower_values))
        upper_mass.append(mass_center)
        upper_bias.append(float(np.mean(upper_values)))
        upper_sem.append(standard_error_mean(upper_values))

    return (
        np.asarray(lower_mass, dtype=np.float64),
        np.asarray(lower_bias, dtype=np.float64),
        np.asarray(lower_sem, dtype=np.float64),
        np.asarray(upper_mass, dtype=np.float64),
        np.asarray(upper_bias, dtype=np.float64),
        np.asarray(upper_sem, dtype=np.float64),
    )


def attach_paranjape_bias(
    frame,
    boxsize_mpc_h: float,
    n_grid: int = 128,
    k_max_h_mpc: float = 0.1,
    position_columns: Sequence[str] = ("x", "y", "z"),
    dm_particles_path: Optional[Union[str, Path]] = None,
    dm_mass_particle_msun_h: Optional[float] = None,
    dm_batch_size: Optional[int] = None,
    matter_delta_field: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Compute per-halo linear bias with the HALOSCOPE halo-by-halo estimator.

    Requires a matter ``delta`` field (precomputed grid or DM snapshot path).

    Parameters
    ----------
    frame : pandas.DataFrame
        Halo catalog with positions and masses.
    boxsize_mpc_h : float
        Periodic box side length in Mpc/h.
    n_grid : int, optional
        Grid resolution when building ``delta`` from DM particles.
    k_max_h_mpc : float, optional
        Maximum wavenumber in h Mpc^-1.
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
        raise ValueError(
            "Halo-by-halo bias requires matter_delta_field or dm_particles_path; "
            "halo CIC is not a valid matter field for this estimator."
        )
    return paranjape_halo_by_halo_bias(
        positions,
        delta_field,
        boxsize_mpc_h,
        k_max_h_mpc=k_max_h_mpc,
    )


def property_matrix_from_frame(frame, columns: Sequence[str]) -> np.ndarray:
    """
    Stack halo property columns into a 2D array for assembly-bias splits.

    Parameters
    ----------
    frame : pandas.DataFrame
        Halo table containing ``columns``.
    columns : Sequence[str]
        Property names to stack.

    Returns
    -------
    np.ndarray
        Array with shape ``(len(frame), len(columns))``.
    """
    return frame[list(columns)].to_numpy(dtype=np.float64)
