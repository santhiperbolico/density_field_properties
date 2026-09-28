"""Tests for Rockstar catalog column layout helpers."""

import pytest

from density_field_properties.pipelines.catalog_columns import (
    HLIST_CATALOG_LAYOUT,
    ROCKSTAR_LIST_CATALOG_LAYOUT,
    rockstar_catalog_column_kwargs,
)
from density_field_properties.read_data.halos.column_maps import (
    ROCKSTAR_LIST_COLUMNS,
    UNIT_HLIST_COLUMNS,
)


def test_hlist_catalog_column_layout():
    """
    Consistent-trees hlists should use UNIT column indices.
    """
    columns = rockstar_catalog_column_kwargs(HLIST_CATALOG_LAYOUT)
    assert columns["halo_id_position"] == UNIT_HLIST_COLUMNS["id"]
    assert columns["halo_x_position"] == UNIT_HLIST_COLUMNS["x"]
    assert columns["halo_m200b_position"] == UNIT_HLIST_COLUMNS["M200b"]


def test_rockstar_list_catalog_column_layout():
    """
    FastPM Rockstar ``.list`` files should use ROCKSTAR_LIST column indices.
    """
    columns = rockstar_catalog_column_kwargs(ROCKSTAR_LIST_CATALOG_LAYOUT)
    assert columns["halo_id_position"] == ROCKSTAR_LIST_COLUMNS["halo_id"]
    assert columns["halo_x_position"] == ROCKSTAR_LIST_COLUMNS["halo_x"]
    assert columns["halo_m200b_position"] == ROCKSTAR_LIST_COLUMNS["halo_m200b"]


def test_unknown_catalog_layout_raises():
    """
    Unsupported layouts should fail fast.
    """
    with pytest.raises(ValueError, match="Unsupported catalog layout"):
        rockstar_catalog_column_kwargs("unknown")
