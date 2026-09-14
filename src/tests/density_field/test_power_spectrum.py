import numpy as np
import pytest

from density_field_properties.density_field.power_spectrum import (
    median_r_k_below_k,
    spherical_power_spectra,
)
from density_field_properties.halo_catalog.rockstar import (
    RockstarCatalogReader,
    read_rockstar_box_size_header,
)


def test_spherical_power_spectra_identical_fields_give_unit_correlation():
    rng = np.random.default_rng(42)
    n_grid = 16
    delta = rng.normal(size=(n_grid, n_grid, n_grid))
    delta -= delta.mean()

    result = spherical_power_spectra(delta, delta, box_size=100.0, n_k_bins=8)
    finite = np.isfinite(result["r_k"])
    assert np.any(finite)
    assert np.allclose(result["r_k"][finite], 1.0, atol=1e-10)


def test_spherical_power_spectra_independent_fields_have_low_correlation():
    rng = np.random.default_rng(7)
    n_grid = 16
    delta_a = rng.normal(size=(n_grid, n_grid, n_grid))
    delta_b = rng.normal(size=(n_grid, n_grid, n_grid))
    delta_a -= delta_a.mean()
    delta_b -= delta_b.mean()

    result = spherical_power_spectra(delta_a, delta_b, box_size=100.0, n_k_bins=8)
    finite = np.isfinite(result["r_k"])
    assert np.any(finite)
    assert np.all(np.abs(result["r_k"][finite]) < 0.5)


def test_median_r_k_below_k():
    k = np.array([0.01, 0.03, 0.06, 0.10])
    r_k = np.array([0.95, 0.92, 0.40, 0.10])
    median = median_r_k_below_k(k, r_k, k_threshold=0.05)
    assert median == pytest.approx(0.935)


def test_read_rockstar_box_size_header():
    txt = """#Box size: 1000.000000 Mpc/h ; h = 0.6774
1 2 3
"""
    import os
    from tempfile import TemporaryDirectory

    with TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "out_0.list")
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(txt)
        box_size = read_rockstar_box_size_header(path)

    assert box_size == pytest.approx(1000.0)


def test_rockstar_reader_respects_n_lines():
    import os
    from tempfile import TemporaryDirectory

    from density_field_properties.halo_catalog.rockstar import ROCKSTAR_HALO_COLUMNS_POSITION

    def row(halo_id, x, y, z, mass):
        cols = ["0"] * 40
        cols[ROCKSTAR_HALO_COLUMNS_POSITION["halo_id"]] = str(halo_id)
        cols[ROCKSTAR_HALO_COLUMNS_POSITION["halo_x"]] = str(x)
        cols[ROCKSTAR_HALO_COLUMNS_POSITION["halo_y"]] = str(y)
        cols[ROCKSTAR_HALO_COLUMNS_POSITION["halo_z"]] = str(z)
        cols[ROCKSTAR_HALO_COLUMNS_POSITION["m200b"]] = str(mass)
        return " ".join(cols)

    txt = "\n".join(
        [
            "# header",
            row(1, 1.0, 2.0, 3.0, 100.0),
            row(2, 4.0, 5.0, 6.0, 200.0),
            row(3, 7.0, 8.0, 9.0, 300.0),
        ]
    )
    with TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "out_0.list")
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(txt)
        catalog = RockstarCatalogReader.read_catalog(path, n_lines=2)

    assert catalog.n_halos == 2
    assert catalog.halo_id[0] == 1
    assert catalog.halo_id[1] == 2
