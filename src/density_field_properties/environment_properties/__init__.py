"""Environment field properties derived from DM particles and halo positions."""

__all__ = [
    "ANISOTROPY_PATH",
    "DensityFieldInfo",
    "TIDAL_TENSOR_PATH",
    "TidalTensorArray",
    "density_field_cic_main",
    "get_grid_cell",
    "kgrid",
    "save_density_field_cic",
    "tidal_anisotropy_and_overdensity_from_halo_calaog",
]


def __getattr__(name):
    """
    Lazily import subpackages so tidal-only imports avoid optional CIC dependencies.
    """
    if name in {
        "DensityFieldInfo",
        "get_grid_cell",
        "density_field_cic_main",
        "save_density_field_cic",
    }:
        from density_field_properties.environment_properties import cic

        return getattr(cic, name)
    if name == "kgrid":
        from density_field_properties.environment_properties.fourier import kgrid

        return kgrid
    if name in {"TIDAL_TENSOR_PATH", "TidalTensorArray"}:
        from density_field_properties.environment_properties import tidal_tensor

        return getattr(tidal_tensor, name)
    if name in {"ANISOTROPY_PATH", "tidal_anisotropy_and_overdensity_from_halo_calaog"}:
        from density_field_properties.environment_properties import tidal_anisotropy

        return getattr(tidal_anisotropy, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
