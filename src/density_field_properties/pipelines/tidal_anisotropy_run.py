"""Compute CIC density, tidal tensor, and tidal-anisotropy descriptors for one simulation."""

import gc
import logging
import os
from pathlib import Path

import numpy as np

from density_field_properties.environment_properties.cic.cic_deposit import (
    density_field_cic_main,
    get_delta_density,
    load_density_field_cic,
    save_density_field_cic,
)
from density_field_properties.environment_properties.tidal_anisotropy import (
    tidal_anisotropy_and_overdensity_from_halo_calaog,
)
from density_field_properties.environment_properties.tidal_tensor.tensor import TidalTensorArray
from density_field_properties.pipelines.catalog_columns import rockstar_catalog_column_kwargs
from density_field_properties.read_data.halos.registry import get_halo_catalog_reader

TIDAL_ANISOTROPY_DIRNAME = "tidal_anisotropy"
TIDAL_R_MIN_MPC_H = 1000 / 4096
TIDAL_R_MAX_MPC_H = 4.0
TIDAL_R_BINS = 20
DEFAULT_DESCRIPTOR_BATCH_SIZE = 100_000


def density_field_stem(dm_particles_file: Path) -> str:
    """
    Resolve the CIC density artifact stem for a DM particle source path.

    Parameters
    ----------
    dm_particles_file : Path
        Text particle file or FastPM BigFile block directory.

    Returns
    -------
    str
        Base name used for ``{stem}_density`` artifacts.
    """
    normalized = os.path.normpath(str(dm_particles_file))
    base = os.path.basename(normalized)
    if os.path.isdir(normalized):
        if base.isdigit():
            return os.path.basename(os.path.dirname(normalized))
        return base
    stem, _ = os.path.splitext(base)
    return stem


def density_field_paths(work_dir: Path, dm_particles_file: Path) -> tuple[Path, Path]:
    """
    Return expected CIC density and metadata paths under a work directory.

    Parameters
    ----------
    work_dir : Path
        Directory where CIC artifacts are stored.
    dm_particles_file : Path
        DM particle source used to build the density field.

    Returns
    -------
    tuple[Path, Path]
        ``(density_binary, density_info_txt)`` paths.
    """
    stem = density_field_stem(dm_particles_file)
    density_file = work_dir / f"{stem}_density"
    density_info = work_dir / f"{stem}_density_info.txt"
    return density_file, density_info


def tidal_anisotropy_output_dir(work_dir: Path) -> Path:
    """
    Return the tidal-anisotropy descriptor directory for a simulation work dir.

    Parameters
    ----------
    work_dir : Path
        Simulation-specific pipeline working directory.

    Returns
    -------
    Path
        Directory containing ``*_halo_environment_descriptors.txt`` batches.
    """
    return work_dir / TIDAL_ANISOTROPY_DIRNAME


def run_density_field_cic(
    dm_particles_file: Path,
    work_dir: Path,
    box_size: float,
    n_grid: int,
    mass_particle: float,
    cic_batch_size: int | None = None,
) -> tuple[Path, Path]:
    """
    Build and persist a CIC density field for one simulation.

    Parameters
    ----------
    dm_particles_file : Path
        DM particle source path.
    work_dir : Path
        Output directory for density artifacts.
    box_size : float
        Periodic box side length in Mpc/h.
    n_grid : int
        Grid resolution per axis.
    mass_particle : float
        DM particle mass in Msun/h.
    cic_batch_size : int or None, optional
        Particle batch size for streaming CIC deposition.

    Returns
    -------
    tuple[Path, Path]
        ``(density_binary, density_info_txt)`` paths.
    """
    work_dir.mkdir(parents=True, exist_ok=True)
    density, density_info = density_field_cic_main(
        dm_particles_file=str(dm_particles_file),
        mass_particle=mass_particle,
        box_size=box_size,
        n_grid=n_grid,
        batch_size=cic_batch_size,
    )
    density_path = Path(
        save_density_field_cic(
            density=density,
            path=str(work_dir),
            dm_particles_file=str(dm_particles_file),
            density_info=density_info,
        )
    )
    del density
    gc.collect()
    return (
        density_path,
        density_path.parent / f"{density_field_stem(dm_particles_file)}_density_info.txt",
    )


def run_tidal_tensor_from_density(
    density_file: Path,
    density_info_file: Path,
    work_dir: Path,
    read_tensor_from_disk: bool = False,
) -> None:
    """
    Compute tidal-tensor HDF5 batches from a saved CIC density field.

    Parameters
    ----------
    density_file : Path
        Binary CIC density field path.
    density_info_file : Path
        Density metadata text file.
    work_dir : Path
        Directory that will contain ``tidal_tensor/``.
    read_tensor_from_disk : bool, optional
        If True, skip tensor computation and reuse existing artifacts.
    """
    if read_tensor_from_disk:
        logging.info("Skipping tidal tensor computation; reusing %s", work_dir / "tidal_tensor")
        return

    density_data, density_info = load_density_field_cic(
        str(density_file),
        str(density_info_file),
    )
    if density_info is None:
        raise ValueError(f"Density metadata is required: {density_info_file}")

    delta_density = get_delta_density(
        density_data,
        density_info.n_particles,
        density_info.mass_particle,
        density_info.box_size,
    )
    del density_data
    gc.collect()

    gaussian_scales = np.logspace(
        np.log10(TIDAL_R_MIN_MPC_H),
        np.log10(TIDAL_R_MAX_MPC_H),
        TIDAL_R_BINS,
    )[1:]
    TidalTensorArray.from_delta(
        delta=delta_density,
        box_size=density_info.box_size,
        path=str(work_dir),
        gaussian_scale_list=gaussian_scales.tolist(),
    )
    del delta_density
    gc.collect()


def run_tidal_anisotropy_descriptors(
    work_dir: Path,
    halo_catalog_path: Path,
    catalog_layout: str,
    max_halos: int | None = None,
    descriptor_batch_size: int | None = DEFAULT_DESCRIPTOR_BATCH_SIZE,
) -> Path:
    """
    Attach tidal-anisotropy descriptors for halos in a Rockstar-format catalog.

    Parameters
    ----------
    work_dir : Path
        Directory containing ``tidal_tensor/`` and receiving ``tidal_anisotropy/``.
    halo_catalog_path : Path
        Halo catalog used to sample descriptors.
    catalog_layout : str
        ``hlist`` or ``rockstar_list`` column layout.
    max_halos : int or None, optional
        Optional cap on halos processed from the catalog.
    descriptor_batch_size : int or None, optional
        Batch size for streaming catalog reads.

    Returns
    -------
    Path
        Directory with descriptor batch files.
    """
    density_file, density_info_file = _resolve_existing_density_paths(work_dir)
    _, density_info = load_density_field_cic(str(density_file), str(density_info_file))
    if density_info is None:
        raise ValueError(f"Density metadata is required: {density_info_file}")

    halo_reader = get_halo_catalog_reader("rockstar")
    column_kwargs = rockstar_catalog_column_kwargs(catalog_layout)
    output_path = tidal_anisotropy_and_overdensity_from_halo_calaog(
        path=str(work_dir),
        halo_catalog=halo_reader,
        halo_catalog_path=str(halo_catalog_path),
        n_grid=density_info.n_grid,
        box_size=int(density_info.box_size),
        n_lines=max_halos,
        batch_size=descriptor_batch_size,
        catalog_column_kwargs=column_kwargs,
    )
    return Path(output_path)


def run_tidal_anisotropy_pipeline_for_target(
    target_name: str,
    dm_particles_file: Path,
    halo_catalog_path: Path,
    work_dir: Path,
    box_size: float,
    n_grid: int,
    mass_particle: float,
    catalog_layout: str,
    cic_batch_size: int | None = None,
    descriptor_batch_size: int | None = DEFAULT_DESCRIPTOR_BATCH_SIZE,
    max_halos: int | None = None,
    read_tensor_from_disk: bool = False,
    skip_cic_if_exists: bool = True,
) -> Path:
    """
    Run CIC, tidal tensor, and tidal-anisotropy descriptor steps for one simulation.

    Parameters
    ----------
    target_name : str
        Simulation label used in log messages (``unit`` or ``fastpm``).
    dm_particles_file : Path
        DM particle source path.
    halo_catalog_path : Path
        Halo catalog path for descriptor attachment.
    work_dir : Path
        Simulation-specific working directory under the run output tree.
    box_size : float
        Periodic box side length in Mpc/h.
    n_grid : int
        CIC grid resolution per axis.
    mass_particle : float
        DM particle mass in Msun/h.
    catalog_layout : str
        Rockstar reader column layout for ``halo_catalog_path``.
    cic_batch_size : int or None, optional
        Particle batch size for CIC deposition.
    descriptor_batch_size : int or None, optional
        Halo batch size for descriptor generation.
    max_halos : int or None, optional
        Optional cap on halos processed from the catalog.
    read_tensor_from_disk : bool, optional
        If True, skip tidal-tensor recomputation and reuse ``tidal_tensor/``.
    skip_cic_if_exists : bool, optional
        If True, reuse an existing density field when both artifacts are present.

    Returns
    -------
    Path
        Directory containing tidal-anisotropy descriptor batches.
    """
    logging.info(
        "Starting tidal-anisotropy pipeline for target=%s work_dir=%s", target_name, work_dir
    )
    work_dir.mkdir(parents=True, exist_ok=True)
    density_file, density_info_file = density_field_paths(work_dir, dm_particles_file)

    if skip_cic_if_exists and density_file.is_file() and density_info_file.is_file():
        logging.info("Reusing existing CIC density field at %s", density_file)
    else:
        density_file, density_info_file = run_density_field_cic(
            dm_particles_file=dm_particles_file,
            work_dir=work_dir,
            box_size=box_size,
            n_grid=n_grid,
            mass_particle=mass_particle,
            cic_batch_size=cic_batch_size,
        )

    run_tidal_tensor_from_density(
        density_file=density_file,
        density_info_file=density_info_file,
        work_dir=work_dir,
        read_tensor_from_disk=read_tensor_from_disk,
    )
    descriptors_dir = run_tidal_anisotropy_descriptors(
        work_dir=work_dir,
        halo_catalog_path=halo_catalog_path,
        catalog_layout=catalog_layout,
        max_halos=max_halos,
        descriptor_batch_size=descriptor_batch_size,
    )
    logging.info(
        "Finished tidal-anisotropy pipeline for target=%s descriptors_dir=%s",
        target_name,
        descriptors_dir,
    )
    return descriptors_dir


def _resolve_existing_density_paths(work_dir: Path) -> tuple[Path, Path]:
    """
    Locate a single CIC density artifact pair inside a work directory.

    Parameters
    ----------
    work_dir : Path
        Simulation-specific working directory.

    Returns
    -------
    tuple[Path, Path]
        ``(density_binary, density_info_txt)`` paths.

    Raises
    ------
    FileNotFoundError
        If no unique density artifact pair is found.
    """
    density_files = sorted(work_dir.glob("*_density"))
    info_files = sorted(work_dir.glob("*_density_info.txt"))
    if len(density_files) != 1 or len(info_files) != 1:
        raise FileNotFoundError(
            f"Expected one density artifact pair in {work_dir}, "
            f"found {len(density_files)} density files and {len(info_files)} info files."
        )
    return density_files[0], info_files[0]
