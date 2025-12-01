import numpy as np
import pytest

from density_field_properties.halo_catalog.halo_catalog import HaloCatalogData


def test_halo_catalog_data_basic_columns():
    data = np.array(
        [
            [10, 1.0, 2.0, 3.0, 100.0, 5.0],
            [20, 4.0, 5.0, 6.0, 200.0, 6.0],
        ]
    )

    hcat = HaloCatalogData(
        data,
        halo_id_position=0,
        halo_x_position=1,
        halo_y_position=2,
        halo_z_position=3,
        halo_m200b_position=4,
        halo_rg_position=5,
    )

    assert hcat.n_halos == 2
    assert np.all(hcat.halo_id == np.array([10, 20]))
    assert np.all(hcat.halo_x == np.array([1.0, 4.0]))
    assert np.all(hcat.halo_m200b == np.array([100.0, 200.0]))
    assert np.all(hcat.halo_rg == np.array([5.0, 6.0]))


def test_halo_catalog_data_missing_m200b_raises():
    data = np.zeros((3, 4))
    hcat = HaloCatalogData(data, 0, 1, 2, 3)

    with pytest.raises(ValueError):
        _ = hcat.halo_m200b


def test_halo_catalog_data_save_properties(tmp_path):
    data = np.array(
        [
            [1, 1.0, 2.0, 3.0, 10.0, 5.0],
        ]
    )
    hcat = HaloCatalogData(data, 0, 1, 2, 3, 4, 5)

    out_file = tmp_path / "saved.txt"
    props = np.array([42.0])

    hcat.save_properties(
        path=str(out_file),
        properties_data=props,
        properties_header="TestProperty",
    )

    assert out_file.exists()
    txt = out_file.read_text()
    assert "Halo ID" in txt
    assert "TestProperty" in txt
