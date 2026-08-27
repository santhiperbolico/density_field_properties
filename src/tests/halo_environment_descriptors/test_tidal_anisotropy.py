import os
from unittest.mock import MagicMock

import numpy as np
import pytest

from density_field_properties.halo_catalog.halo_catalog import HaloCatalogData
from density_field_properties.halo_environment_descriptors.tidal_anisotropy import (
    ANISOTROPY_PATH,
    _tidal_anisotropy_and_overdensity_from_halo_calaog_batches,
    _tidal_anisotropy_and_overdensity_from_halo_calaog_complete,
    format_halo_catalog,
    tidal_anisotropy_and_overdensity_from_halo_calaog,
)
from density_field_properties.tidal_tensor import TidalTensorArray

TIDAL_DESCRIPTOR_N_COLS = 8


@pytest.fixture
def mock_halo_data():
    halo_data = MagicMock()
    halo_data.data = np.array(
        [
            [1, 1, 1, 0.8],
            [2, 2, 2, 1.2],
            [5, 5, 5, 4.0],
        ],
        dtype=float,
    )

    halo_data.x_position = 0
    halo_data.y_position = 1
    halo_data.z_position = 2
    halo_data.rg_position = 3

    halo_data.halo_x = halo_data.data[:, 0]
    halo_data.halo_y = halo_data.data[:, 1]
    halo_data.halo_z = halo_data.data[:, 2]
    halo_data.halo_rg = halo_data.data[:, 3]

    halo_data.save_properties = MagicMock()
    halo_data.n_halos = len(halo_data.data)
    return halo_data


@pytest.fixture
def mock_halo_catalog(mock_halo_data):
    reader = MagicMock()

    reader.read_catalog.return_value = mock_halo_data

    def gen(path, batch_size, start_offset):
        yield mock_halo_data, 0, 3
        yield mock_halo_data, 3, 6

    reader.read_catalog_batch_generator = MagicMock(side_effect=gen)
    return reader


@pytest.fixture
def mock_tidal_tensor_array():
    tta = MagicMock()
    tta.gaussian_scale_list = [0.5, 1.0, 2.0]

    def fake_compute(x, y, z, rg):
        n = len(np.array(x))
        return np.ones(n), np.zeros(n)

    tta.get_tidal_anisotropy_and_overdensity = MagicMock(side_effect=fake_compute)
    return tta


def test_format_halo_catalog_filters_and_maps_positions(mock_halo_data):
    box_size = 10
    n_grid = 5
    r_min, r_max = 0.5, 2.0

    formatted = format_halo_catalog(
        mock_halo_data,
        box_size=box_size,
        n_grid=n_grid,
        r_min=r_min,
        r_max=r_max,
    )

    assert len(formatted.data) == 2

    assert np.all(formatted.data[:, 0] < n_grid)
    assert np.all(formatted.data[:, 1] < n_grid)
    assert np.all(formatted.data[:, 2] < n_grid)


def test_tidal_anisotropy_complete(tmp_path, mock_halo_catalog, mock_tidal_tensor_array):
    out = _tidal_anisotropy_and_overdensity_from_halo_calaog_complete(
        output_path=str(tmp_path) + "/",
        tidal_tensor_array=mock_tidal_tensor_array,
        halo_catalog=mock_halo_catalog,
        halo_catalog_path="catalog.txt",
        n_grid=10,
        box_size=50,
    )

    assert os.path.exists(out)
    mock_halo_catalog.read_catalog.assert_called_once()
    mock_tidal_tensor_array.get_tidal_anisotropy_and_overdensity.assert_called_once()
    mock_halo_catalog.read_catalog.return_value.save_properties.assert_called_once()


def test_tidal_anisotropy_batches(tmp_path, mock_halo_catalog, mock_tidal_tensor_array):
    out = _tidal_anisotropy_and_overdensity_from_halo_calaog_batches(
        output_path=str(tmp_path) + "/",
        tidal_tensor_array=mock_tidal_tensor_array,
        halo_catalog=mock_halo_catalog,
        halo_catalog_path="catalog.txt",
        n_grid=10,
        box_size=50,
        batch_size=3,
        n_lines=10,
    )

    assert os.path.exists(out)

    assert mock_halo_catalog.read_catalog_batch_generator.call_count == 1
    assert mock_tidal_tensor_array.get_tidal_anisotropy_and_overdensity.call_count == 2


def test_tidal_anisotropy_top_level(
    tmp_path, mock_halo_catalog, mock_tidal_tensor_array, monkeypatch
):
    path = str(tmp_path)
    tidal_tensor_path = os.path.join(path, "tidal_tensor")
    os.makedirs(tidal_tensor_path)

    monkeypatch.setattr(TidalTensorArray, "from_folder", lambda path: mock_tidal_tensor_array)

    out = tidal_anisotropy_and_overdensity_from_halo_calaog(
        path=path,
        halo_catalog=mock_halo_catalog,
        halo_catalog_path="catalog.txt",
        n_grid=10,
        box_size=50,
    )

    assert os.path.exists(out)
    assert out.endswith(ANISOTROPY_PATH + "/")


def test_tidal_anisotropy_missing_tensor_path(tmp_path, mock_halo_catalog):
    with pytest.raises(ValueError):
        tidal_anisotropy_and_overdensity_from_halo_calaog(
            path=str(tmp_path),
            halo_catalog=mock_halo_catalog,
            halo_catalog_path="catalog.txt",
            n_grid=10,
            box_size=50,
        )


@pytest.fixture
def integration_halo_catalog():
    reader = MagicMock()

    def read_catalog(path, n_lines=None):
        data = np.array(
            [
                [100, 2.0, 4.0, 6.0, 1e12, 1.0],
                [200, 5.0, 5.0, 5.0, 2e12, 1.5],
                [300, 1.0, 1.0, 1.0, 3e12, 4.0],
            ]
        )
        return HaloCatalogData(data, 0, 1, 2, 3, 4, 5)

    reader.read_catalog = MagicMock(side_effect=read_catalog)
    return reader


@pytest.fixture
def small_tidal_tensor_array(tmp_path):
    n_grid = 8
    box_size = 10
    delta = np.random.default_rng(0).normal(0, 0.01, (n_grid, n_grid, n_grid))
    tidal_tensor_array = TidalTensorArray.from_delta(
        delta=delta,
        box_size=box_size,
        path=str(tmp_path),
        gaussian_scale_list=[0.5, 1.0, 2.0],
    )
    return tidal_tensor_array, box_size, n_grid


def test_tidal_anisotropy_complete_writes_rows_per_halo(
    tmp_path, integration_halo_catalog, small_tidal_tensor_array
):
    tidal_tensor_array, box_size, n_grid = small_tidal_tensor_array
    n_halos_in_range = 2

    _tidal_anisotropy_and_overdensity_from_halo_calaog_complete(
        output_path=str(tmp_path) + "/",
        tidal_tensor_array=tidal_tensor_array,
        halo_catalog=integration_halo_catalog,
        halo_catalog_path="catalog.txt",
        n_grid=n_grid,
        box_size=box_size,
    )

    output_file = os.path.join(tmp_path, "halo_environment_descriptors.txt")
    assert os.path.exists(output_file)

    saved = np.loadtxt(output_file, comments="#")
    if saved.ndim == 1:
        saved = saved.reshape(1, -1)

    assert saved.shape == (n_halos_in_range, TIDAL_DESCRIPTOR_N_COLS)
    assert not np.isnan(saved).any()
