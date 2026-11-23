"""Runtime context for Haloscope preprocessing."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from density_field_properties.preprocessing.mass_correction.config import MassCorrectionSettings
from density_field_properties.preprocessing.mass_filter_config import (
    HaloMassFilterConfig,
    default_halo_mass_filter_config,
)


@dataclass
class SimulationRunContext:
    """
    Per-simulation parameters for attaching Haloscope input features.

    Parameters
    ----------
    boxsize_mpc_h : float
        Periodic box side length in Mpc/h.
    env_radius_mpc_h : float
        Neighbor search radius for the ``env`` feature in Mpc/h.
    tidal_descriptors_dir : Optional[Path]
        Directory with tidal anisotropy descriptor batches.
    tidal_n_grid : int
        CIC grid resolution used when tidal descriptors were computed.
    max_descriptor_batch_files : Optional[int]
        Smoke cap on tidal descriptor batch files.
    """

    boxsize_mpc_h: float
    env_radius_mpc_h: float = 5.0
    tidal_descriptors_dir: Optional[Path] = None
    tidal_n_grid: int = 512
    max_descriptor_batch_files: Optional[int] = None


@dataclass
class PreprocessingContext:
    """
    HR/LR preprocessing settings shared across feature attachers.

    Parameters
    ----------
    sim : SimulationRunContext
        Context for the high-resolution training simulation.
    fastpm : SimulationRunContext
        Context for the low-resolution target simulation.
    mass_correction : MassCorrectionSettings
        LR mass correction method and parameters.
    mass_filter : HaloMassFilterConfig
        Optional HR/LR ``M200b`` range filter applied after feature attachment.
    """

    sim: SimulationRunContext
    fastpm: SimulationRunContext
    mass_correction: MassCorrectionSettings
    mass_filter: HaloMassFilterConfig = field(default_factory=default_halo_mass_filter_config)

    @property
    def apply_mass_correction(self) -> bool:
        """
        Whether LR mass correction is enabled.

        Returns
        -------
        bool
            ``mass_correction.enabled``.
        """
        return self.mass_correction.enabled


def build_preprocessing_context(
    repo_root: Path,
    mass_correction: MassCorrectionSettings,
    sim_boxsize_mpc_h: float,
    fastpm_boxsize_mpc_h: float,
    env_radius_mpc_h: float,
    unit_descriptors_dir: Optional[Path] = None,
    fastpm_descriptors_dir: Optional[Path] = None,
    tidal_n_grid: int = 512,
    max_descriptor_batch_files: Optional[int] = None,
    mass_filter: Optional[HaloMassFilterConfig] = None,
) -> PreprocessingContext:
    """
    Build HR/LR preprocessing contexts with optional tidal descriptor paths.

    Parameters
    ----------
    repo_root : Path
        Repository root used to resolve relative descriptor directories.
    mass_correction : MassCorrectionSettings
        LR mass correction settings for Haloscope preprocessing.
    sim_boxsize_mpc_h : float
        SIM periodic box side in Mpc/h.
    fastpm_boxsize_mpc_h : float
        FastPM periodic box side in Mpc/h.
    env_radius_mpc_h : float
        Environment search radius in Mpc/h.
    unit_descriptors_dir : Optional[Path]
        UNIT tidal descriptor directory.
    fastpm_descriptors_dir : Optional[Path]
        FastPM tidal descriptor directory.
    tidal_n_grid : int
        Grid resolution used for tidal descriptors.
    max_descriptor_batch_files : Optional[int]
        Smoke cap on tidal descriptor batch files per simulation.
    mass_filter : Optional[HaloMassFilterConfig]
        Optional ``M200b`` range filter for HR and LR tables.

    Returns
    -------
    PreprocessingContext
        Context for ``build_feature_tables``.
    """
    unit_path = _resolve_descriptor_dir(repo_root, unit_descriptors_dir)
    fastpm_path = _resolve_descriptor_dir(repo_root, fastpm_descriptors_dir)
    return PreprocessingContext(
        sim=SimulationRunContext(
            boxsize_mpc_h=sim_boxsize_mpc_h,
            env_radius_mpc_h=env_radius_mpc_h,
            tidal_descriptors_dir=unit_path,
            tidal_n_grid=tidal_n_grid,
            max_descriptor_batch_files=max_descriptor_batch_files,
        ),
        fastpm=SimulationRunContext(
            boxsize_mpc_h=fastpm_boxsize_mpc_h,
            env_radius_mpc_h=env_radius_mpc_h,
            tidal_descriptors_dir=fastpm_path,
            tidal_n_grid=tidal_n_grid,
            max_descriptor_batch_files=max_descriptor_batch_files,
        ),
        mass_correction=mass_correction,
        mass_filter=(
            mass_filter if mass_filter is not None else default_halo_mass_filter_config()
        ),
    )


def _resolve_descriptor_dir(
    repo_root: Path,
    descriptor_dir: Optional[Path],
) -> Optional[Path]:
    """
    Resolve a tidal descriptor directory relative to the repository root.

    Parameters
    ----------
    repo_root : Path
        Repository root path.
    descriptor_dir : Optional[Path]
        Absolute or repository-relative descriptor directory.

    Returns
    -------
    Optional[Path]
        Resolved path, or ``None`` when no directory was provided.
    """
    if descriptor_dir is None:
        return None
    path = Path(descriptor_dir)
    if path.is_absolute():
        return path
    return repo_root / path
