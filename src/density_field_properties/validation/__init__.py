"""Scientific validation metrics and diagnostic plots for Haloscope runs."""

__all__ = [
    "write_tidal_assembly_bias_panel",
]


def __getattr__(name: str):
    """
    Lazily import validation helpers to avoid heavy optional dependencies.

    Parameters
    ----------
    name : str
        Requested attribute name.

    Returns
    -------
    object
        Exported validation helper.

    Raises
    ------
    AttributeError
        If ``name`` is not part of the public API.
    """
    if name == "write_tidal_assembly_bias_panel":
        from density_field_properties.validation.assembly_bias_panel import (
            write_tidal_assembly_bias_panel,
        )

        return write_tidal_assembly_bias_panel
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
