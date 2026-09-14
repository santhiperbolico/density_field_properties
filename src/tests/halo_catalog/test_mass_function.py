from tempfile import TemporaryDirectory

import numpy as np
import pytest

from density_field_properties.halo_catalog.mass_function import (
    compute_halo_mass_function,
    default_hmf_log_mass_bin_edges,
    detect_catalog_reader_name,
    halo_mass_function_from_catalog,
)


def test_default_hmf_log_mass_bin_edges():
    edges = default_hmf_log_mass_bin_edges(log_mass_min=10.0, log_mass_max=12.0, n_bins=4)
    assert edges.shape == (5,)
    assert edges[0] == pytest.approx(10.0)
    assert edges[-1] == pytest.approx(12.0)


@pytest.mark.parametrize(
    "path_text,expected",
    [
        ("out_8.list", "rockstar"),
        ("out_128p.list.bz2", "rockstar"),
    ],
)
def test_detect_catalog_reader_name_rockstar(path_text, expected, tmp_path):
    catalog_path = tmp_path / path_text
    catalog_path.write_text("# header\n", encoding="utf-8")
    assert detect_catalog_reader_name(catalog_path) == expected


def test_detect_catalog_reader_name_fastpm(tmp_path):
    catalog_path = tmp_path / "fof_1.000"
    catalog_path.mkdir()
    assert detect_catalog_reader_name(catalog_path) == "fastpm"


def test_compute_halo_mass_function_normalization():
    box_size = 1000.0
    masses = np.full(100, 5.0e10)
    edges = np.array([10.0, 11.0, 12.0])
    result = compute_halo_mass_function(
        masses,
        box_size_mpc_h=box_size,
        log_mass_bin_edges=edges,
    )

    volume = box_size**3
    assert result.n_halos == 100
    assert result.counts[0] == pytest.approx(100.0)
    assert result.log_mass_bin_centers[0] == pytest.approx(10.5)
    assert result.dn_dlog10_m[0] == pytest.approx(100.0 / volume)
    assert result.dn_dln_m[0] == pytest.approx(100.0 / (np.log(10.0) * volume))


def test_compute_halo_mass_function_empty_raises():
    with pytest.raises(ValueError, match="No halos remain"):
        compute_halo_mass_function(np.array([]), box_size_mpc_h=1000.0)


def test_halo_mass_function_from_catalog_rockstar():
    catalog_text = """#Box size: 1000.000000 Mpc/h ; h = 0.6774
#ID desc_id ... columns ...
0 1 0 0 0 0 0 0 1.0 2.0 3.0 0 0 0 0 0 0 0 0 0 1.0e11 0 0 0 0 0 0 0 0 0 0 0 0 -1
1 2 0 0 0 0 0 0 4.0 5.0 6.0 0 0 0 0 0 0 0 0 0 5.0e11 0 0 0 0 0 0 0 0 0 0 0 0 10
"""
    with TemporaryDirectory() as tmpdir:
        catalog_path = f"{tmpdir}/out_8.list"
        with open(catalog_path, "w", encoding="utf-8") as handle:
            handle.write(catalog_text)

        result = halo_mass_function_from_catalog(
            catalog_path,
            catalog_name="rockstar",
            central_only=False,
            log_mass_bin_edges=np.array([10.0, 11.0, 12.0, 13.0]),
        )

    assert result.n_halos == 2
    assert result.counts[1] == pytest.approx(2.0)
    assert result.box_size_mpc_h == pytest.approx(1000.0)
