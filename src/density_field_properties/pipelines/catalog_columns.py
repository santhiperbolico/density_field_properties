"""Rockstar catalog column layouts for pipeline tidal-anisotropy runs."""

from typing import Any

from density_field_properties.read_data.halos.column_maps import (
    ROCKSTAR_LIST_COLUMNS,
    UNIT_HLIST_COLUMNS,
)
from density_field_properties.read_data.halos.rockstar import ROCKSTAR_HALO_COLUMNS_POSITION

HLIST_CATALOG_LAYOUT = "hlist"
ROCKSTAR_LIST_CATALOG_LAYOUT = "rockstar_list"

_CATALOG_LAYOUTS = frozenset({HLIST_CATALOG_LAYOUT, ROCKSTAR_LIST_CATALOG_LAYOUT})


def rockstar_catalog_column_kwargs(catalog_layout: str) -> dict[str, int]:
    """
    Return ``RockstarCatalogReader`` column index kwargs for a catalog layout.

    Parameters
    ----------
    catalog_layout : str
        Either ``hlist`` (consistent-trees) or ``rockstar_list`` (Rockstar ``.list``).

    Returns
    -------
    dict[str, int]
        Keyword arguments for ``read_catalog`` and ``read_catalog_batch_generator``.

    Raises
    ------
    ValueError
        If ``catalog_layout`` is not supported.
    """
    if catalog_layout not in _CATALOG_LAYOUTS:
        supported = ", ".join(sorted(_CATALOG_LAYOUTS))
        raise ValueError(f"Unsupported catalog layout '{catalog_layout}'; expected: {supported}")

    if catalog_layout == HLIST_CATALOG_LAYOUT:
        return {
            "halo_id_position": UNIT_HLIST_COLUMNS["id"],
            "halo_x_position": UNIT_HLIST_COLUMNS["x"],
            "halo_y_position": UNIT_HLIST_COLUMNS["y"],
            "halo_z_position": UNIT_HLIST_COLUMNS["z"],
            "halo_m200b_position": UNIT_HLIST_COLUMNS["M200b"],
        }

    return {
        "halo_id_position": ROCKSTAR_LIST_COLUMNS["halo_id"],
        "halo_x_position": ROCKSTAR_LIST_COLUMNS["halo_x"],
        "halo_y_position": ROCKSTAR_LIST_COLUMNS["halo_y"],
        "halo_z_position": ROCKSTAR_LIST_COLUMNS["halo_z"],
        "halo_m200b_position": ROCKSTAR_LIST_COLUMNS["halo_m200b"],
    }


def default_hlist_catalog_column_kwargs() -> dict[str, int]:
    """
    Return the default consistent-trees column mapping used by legacy scripts.

    Returns
    -------
    dict[str, int]
        Keyword arguments aligned with ``ROCKSTAR_HALO_COLUMNS_POSITION``.
    """
    return {
        "halo_id_position": ROCKSTAR_HALO_COLUMNS_POSITION["halo_id"],
        "halo_x_position": ROCKSTAR_HALO_COLUMNS_POSITION["halo_x"],
        "halo_y_position": ROCKSTAR_HALO_COLUMNS_POSITION["halo_y"],
        "halo_z_position": ROCKSTAR_HALO_COLUMNS_POSITION["halo_z"],
        "halo_m200b_position": ROCKSTAR_HALO_COLUMNS_POSITION["m200b"],
    }


def merge_catalog_column_kwargs(
    catalog_column_kwargs: dict[str, Any] | None,
) -> dict[str, int]:
    """
    Resolve optional catalog column overrides for Rockstar readers.

    Parameters
    ----------
    catalog_column_kwargs : dict[str, Any] or None
        Explicit column index mapping, or ``None`` for legacy defaults.

    Returns
    -------
    dict[str, int]
        Column index kwargs for Rockstar catalog readers.
    """
    if catalog_column_kwargs is None:
        return default_hlist_catalog_column_kwargs()
    return {key: int(value) for key, value in catalog_column_kwargs.items()}
