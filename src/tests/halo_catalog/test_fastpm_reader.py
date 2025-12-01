from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from density_field_properties.halo_catalog.fastpm import MSUN_G, FastPMCatalogReader
from density_field_properties.halo_catalog.halo_catalog import HaloCatalogData


@pytest.fixture
def mock_bigfile():
    bfile = MagicMock()

    bfile.attrs = {
        "OmegaM": np.array([0.3]),
        "OmegaLambda": np.array([0.7]),
        "HubbleParam": np.array([0.7]),
        "UnitMass_in_g": np.array([MSUN_G]),
    }

    halo_id = np.array([1, 2, 3])
    halo_length = np.array([10, 20, 30])
    halo_pos = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])

    def open_side_effect(name):
        if name.endswith("/ID"):
            return halo_id
        if name.endswith("/Length"):
            return halo_length
        if name.endswith("/Position"):
            return halo_pos
        raise KeyError(f"Unexpected block name: {name}")

    bfile.open.side_effect = open_side_effect
    return bfile


def test_fastpm_reader_basic(mock_bigfile):
    with patch("density_field_properties.halo_catalog.fastpm.BigFile", return_value=mock_bigfile):
        hcat = FastPMCatalogReader.read_catalog("path/to/fastpm/000")

    assert isinstance(hcat, HaloCatalogData)
    assert hcat.n_halos == 3
    assert np.all(hcat.halo_id == np.array([1, 2, 3]))
    assert np.all(hcat.halo_x == np.array([1.0, 4.0, 7.0]))
    assert np.all(hcat.halo_y == np.array([2.0, 5.0, 8.0]))
    assert np.all(hcat.halo_z == np.array([3.0, 6.0, 9.0]))


def test_fastpm_reader_batch_generator(mock_bigfile):
    with patch("density_field_properties.halo_catalog.fastpm.BigFile", return_value=mock_bigfile):
        gen = FastPMCatalogReader.read_catalog_batch_generator("path/to/fastpm/000", batch_size=1)
        batch, start, end = next(gen)

    assert batch.n_halos == 1
    assert batch.halo_id[0] == 1
    assert isinstance(start, int)
    assert isinstance(end, int)
