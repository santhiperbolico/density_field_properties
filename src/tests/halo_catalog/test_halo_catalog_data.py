import numpy as np
import pytest

from density_field_properties.halo_catalog.halo_catalog import HaloCatalogData

TIDAL_DESCRIPTOR_N_COLS = 8


def _read_data_rows(path):
    return np.loadtxt(path, comments="#")


def _make_catalog(data):
    return HaloCatalogData(data, 0, 1, 2, 3, 4, 5)


def test_halo_catalog_data_basic_columns():
    data = np.array(
        [
            [10, 1.0, 2.0, 3.0, 100.0, 5.0],
            [20, 4.0, 5.0, 6.0, 200.0, 6.0],
        ]
    )

    hcat = _make_catalog(data)

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


@pytest.mark.parametrize(
    "data, properties_data, properties_header, expected",
    [
        (
            np.array([[1, 1.0, 2.0, 3.0, 10.0, 5.0]]),
            np.array([42.0]),
            "TestProperty",
            np.array([[1.0, 1.0, 2.0, 3.0, 10.0, 5.0, 42.0]]),
        ),
        (
            np.array(
                [
                    [1, 1.0, 2.0, 3.0, 10.0, 5.0],
                    [2, 4.0, 5.0, 6.0, 20.0, 6.0],
                ]
            ),
            (np.array([0.1, 0.2]), np.array([0.3, 0.4])),
            "Tidal Anisotropy, Overdensity",
            np.array(
                [
                    [1.0, 1.0, 2.0, 3.0, 10.0, 5.0, 0.1, 0.3],
                    [2.0, 4.0, 5.0, 6.0, 20.0, 6.0, 0.2, 0.4],
                ]
            ),
        ),
        (
            np.array(
                [
                    [1, 1.0, 2.0, 3.0, 10.0, 5.0],
                    [2, 4.0, 5.0, 6.0, 20.0, 6.0],
                ]
            ),
            np.array([[0.1, 0.3], [0.2, 0.4]]),
            "Tidal Anisotropy, Overdensity",
            np.array(
                [
                    [1.0, 1.0, 2.0, 3.0, 10.0, 5.0, 0.1, 0.3],
                    [2.0, 4.0, 5.0, 6.0, 20.0, 6.0, 0.2, 0.4],
                ]
            ),
        ),
    ],
)
def test_halo_catalog_data_save_properties_layout(
    tmp_path, data, properties_data, properties_header, expected
):
    hcat = _make_catalog(data)
    out_file = tmp_path / "saved.txt"

    hcat.save_properties(
        path=str(out_file),
        properties_data=properties_data,
        properties_header=properties_header,
    )

    assert out_file.exists()
    txt = out_file.read_text()
    assert "Halo ID" in txt
    assert properties_header.split(",")[0].strip() in txt

    saved = _read_data_rows(out_file)
    if saved.ndim == 1:
        saved = saved.reshape(1, -1)
    assert saved.shape == expected.shape
    np.testing.assert_allclose(saved, expected)


def test_halo_catalog_data_save_properties_wrong_length_raises(tmp_path):
    data = np.array(
        [
            [1, 1.0, 2.0, 3.0, 10.0, 5.0],
            [2, 4.0, 5.0, 6.0, 20.0, 6.0],
        ]
    )
    hcat = _make_catalog(data)
    out_file = tmp_path / "saved.txt"

    with pytest.raises(ValueError, match="number of halos"):
        hcat.save_properties(
            path=str(out_file),
            properties_data=np.array([0.1, 0.2, 0.3]),
            properties_header="TestProperty",
        )
