"""Deprecated shim — use density_field_properties.validation.assembly_bias."""

import warnings

from density_field_properties.haloscope.sim_to_fastpm.config import (
    OUTPUT_FEATURES,
    default_fastpm_saved_cic_density_paths,
    default_sim_dm_particles_path,
    default_sim_saved_cic_density_paths,
)
from density_field_properties.utils.stats import central_68_scatter
from density_field_properties.validation.assembly_bias import (
    assembly_bias_curves_for_catalog,
    attach_paranjape_bias,
    halo_overdensity_field_cic,
    halo_weighted_overdensity_field,
    joint_assembly_masks,
)
from density_field_properties.validation.assembly_bias import (
    load_fastpm_matter_overdensity as _load_fastpm_matter_overdensity,
)
from density_field_properties.validation.assembly_bias import (
    load_matter_overdensity,
)
from density_field_properties.validation.assembly_bias import (
    load_sim_matter_overdensity as _load_sim_matter_overdensity,
)
from density_field_properties.validation.assembly_bias import (
    matter_overdensity_field_from_dm,
    matter_overdensity_from_saved_cic,
    paranjape_halo_by_halo_bias,
)
from density_field_properties.validation.assembly_bias import (
    property_matrix_from_frame as _property_matrix_from_frame,
)
from density_field_properties.validation.assembly_bias import (
    solve_joint_percentile,
)

warnings.warn(
    "haloscope.sim_to_fastpm.assembly_bias is deprecated; "
    "use density_field_properties.validation.assembly_bias",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = [
    "assembly_bias_curves_for_catalog",
    "attach_paranjape_bias",
    "central_68_scatter",
    "halo_overdensity_field_cic",
    "halo_weighted_overdensity_field",
    "joint_assembly_masks",
    "load_fastpm_matter_overdensity",
    "load_matter_overdensity",
    "load_sim_matter_overdensity",
    "matter_overdensity_field_from_dm",
    "matter_overdensity_from_saved_cic",
    "paranjape_halo_by_halo_bias",
    "property_matrix_from_frame",
    "solve_joint_percentile",
]


def property_matrix_from_frame(frame, columns=None):
    """
    Deprecated wrapper defaulting property columns from sim_to_fastpm config.
    """
    if columns is None:
        columns = OUTPUT_FEATURES
    return _property_matrix_from_frame(frame, columns)


def load_sim_matter_overdensity(
    repo_root,
    boxsize_mpc_h,
    n_grid,
    dm_mass_particle_msun_h,
    dm_batch_size=None,
    dm_particles_path=None,
    saved_density_path=None,
    saved_density_info_path=None,
):
    """
    Deprecated wrapper applying sim_to_fastpm config path defaults.
    """
    if saved_density_path is None or saved_density_info_path is None:
        saved_density_path, saved_density_info_path = default_sim_saved_cic_density_paths()
    if dm_particles_path is None:
        dm_particles_path = default_sim_dm_particles_path()
    return _load_sim_matter_overdensity(
        repo_root,
        boxsize_mpc_h,
        n_grid,
        dm_mass_particle_msun_h,
        dm_batch_size=dm_batch_size,
        dm_particles_path=dm_particles_path,
        saved_density_path=saved_density_path,
        saved_density_info_path=saved_density_info_path,
    )


def load_fastpm_matter_overdensity(
    repo_root,
    boxsize_mpc_h,
    n_grid,
    dm_mass_particle_msun_h,
    dm_particles_path,
    dm_batch_size=None,
    saved_density_path=None,
    saved_density_info_path=None,
):
    """
    Deprecated wrapper applying sim_to_fastpm config path defaults.
    """
    if saved_density_path is None or saved_density_info_path is None:
        saved_density_path, saved_density_info_path = default_fastpm_saved_cic_density_paths()
    return _load_fastpm_matter_overdensity(
        repo_root,
        boxsize_mpc_h,
        n_grid,
        dm_mass_particle_msun_h,
        dm_particles_path,
        dm_batch_size=dm_batch_size,
        saved_density_path=saved_density_path,
        saved_density_info_path=saved_density_info_path,
    )
