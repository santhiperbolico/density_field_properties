import os
from tempfile import TemporaryDirectory

import numpy as np
import pytest

from density_field_properties.cosmology import Cosmology
from density_field_properties.halo_catalog.rockstar import (
    ROCKSTAR_HALO_COLUMNS_POSITION,
    RockstarCatalogReader,
    read_rockstar_cosmology_header,
)

FASTPM_COSMO_OM = 0.308900
FASTPM_COSMO_OL = 0.691100
FASTPM_COSMO_H = 0.677400


def test_read_rockstar_cosmology_header():
    txt = """#Omega_M = 0.3 ; Omega_L = 0.7; h0 = 0.6
            # Other header
            1 2 3
            """
    with TemporaryDirectory() as tmpdir:
        tmpfile_name = os.path.join(tmpdir, "rockstar.txt")
        with open(tmpfile_name, "w") as tmpfile:
            tmpfile.write(txt)
        cosmo = read_rockstar_cosmology_header(tmpfile_name)

    assert isinstance(cosmo, Cosmology)
    assert cosmo.omega_matter == 0.3
    assert cosmo.omega_lambda == 0.7
    assert cosmo.h0 == 0.6


def test_rockstar_reader_basic(tmp_path):
    txt = """# Comment
0 10 0 0 0  1 2 3 4 5  39  1.0 2.0 3.0  100.0
1 20 0 0 0  1 2 3 4 5  39  4.0 5.0 6.0  200.0
"""
    with TemporaryDirectory() as tmpdir:
        tmpfile_name = os.path.join(tmpdir, "rockstar.txt")
        with open(tmpfile_name, "w") as tmpfile:
            tmpfile.write(txt)

        hcat = RockstarCatalogReader.read_catalog(
            tmpfile_name,
            halo_x_position=11,
            halo_y_position=12,
            halo_z_position=13,
            halo_m200b_position=14,
        )

    assert hcat.n_halos == 2
    assert np.all(hcat.halo_id == np.array([10, 20]))
    assert np.all(hcat.halo_m200b == np.array([100.0, 200.0]))


def test_rockstar_batch_generator(tmp_path):
    txt = """# Header
#Omega_M = 0.3 ; Omega_L = 0.7 ; h0 = 0.6
0 10 0 0 0  1 2 3 4 5  39  1.0 2.0 3.0  100.0
1 20 0 0 0  1 2 3 4 5  39  4.0 5.0 6.0  200.0
"""
    with TemporaryDirectory() as tmpdir:
        tmpfile_name = os.path.join(tmpdir, "rockstar.txt")
        with open(tmpfile_name, "w") as tmpfile:
            tmpfile.write(txt)

        gen = RockstarCatalogReader.read_catalog_batch_generator(
            tmpfile_name,
            batch_size=1,
            halo_x_position=11,
            halo_y_position=12,
            halo_z_position=13,
            halo_m200b_position=14,
        )
        batch, start, end = next(gen)

    assert batch.n_halos == 1
    assert batch.halo_id[0] == 10
    assert isinstance(start, int)
    assert isinstance(end, int)


@pytest.mark.parametrize(
    "cosmo_line",
    [
        "#Om = 0.308900; Ol = 0.691100; h = 0.677400",
        "#Om=0.308900;Ol=0.691100;h=0.677400",
        "#OM = 0.308900; OL = 0.691100; H = 0.677400",
    ],
)
def test_read_rockstar_cosmology_header_fastpm_om_ol_h(cosmo_line):
    txt = f"""{cosmo_line}
#ID X Y Z ...
1 2 3
"""
    with TemporaryDirectory() as tmpdir:
        tmpfile_name = os.path.join(tmpdir, "rockstar.txt")
        with open(tmpfile_name, "w") as tmpfile:
            tmpfile.write(txt)
        cosmo = read_rockstar_cosmology_header(tmpfile_name)

    assert isinstance(cosmo, Cosmology)
    assert cosmo.omega_matter == pytest.approx(FASTPM_COSMO_OM)
    assert cosmo.omega_lambda == pytest.approx(FASTPM_COSMO_OL)
    assert cosmo.h0 == pytest.approx(FASTPM_COSMO_H)


def _rockstar_data_row(halo_id, x, y, z, m200b):
    cols = ["0"] * 40
    cols[ROCKSTAR_HALO_COLUMNS_POSITION["halo_id"]] = str(halo_id)
    cols[ROCKSTAR_HALO_COLUMNS_POSITION["halo_x"]] = str(x)
    cols[ROCKSTAR_HALO_COLUMNS_POSITION["halo_y"]] = str(y)
    cols[ROCKSTAR_HALO_COLUMNS_POSITION["halo_z"]] = str(z)
    cols[ROCKSTAR_HALO_COLUMNS_POSITION["m200b"]] = str(m200b)
    return " ".join(cols)


def test_rockstar_reader_with_fastpm_cosmology_header():
    m200b = 1.0e12
    cosmo = Cosmology(FASTPM_COSMO_OM, FASTPM_COSMO_OL, FASTPM_COSMO_H)
    expected_rg = 4.0 * cosmo.convert_m200b_to_r200b(m200b) / np.sqrt(5.0)
    txt = f"""#Om = {FASTPM_COSMO_OM}; Ol = {FASTPM_COSMO_OL}; h = {FASTPM_COSMO_H}
#ID X Y Z M200b_all
{_rockstar_data_row(10, 1.0, 2.0, 3.0, m200b)}
"""
    with TemporaryDirectory() as tmpdir:
        tmpfile_name = os.path.join(tmpdir, "out_0.list")
        with open(tmpfile_name, "w") as tmpfile:
            tmpfile.write(txt)
        hcat = RockstarCatalogReader.read_catalog(tmpfile_name)

    assert hcat.n_halos == 1
    assert hcat.rg_position is not None
    assert hcat.halo_rg[0] == pytest.approx(expected_rg)


def test_rockstar_reader_fastpm_list_fixture():
    txt = f"""#Om = {FASTPM_COSMO_OM}; Ol = {FASTPM_COSMO_OL}; h = {FASTPM_COSMO_H}
#ID X Y Z (Rockstar-style header)
{_rockstar_data_row(100, 10.5, 20.5, 30.5, 5.0e11)}
{_rockstar_data_row(101, 11.5, 21.5, 31.5, 6.0e11)}
"""
    with TemporaryDirectory() as tmpdir:
        tmpfile_name = os.path.join(tmpdir, "out_0.list")
        with open(tmpfile_name, "w") as tmpfile:
            tmpfile.write(txt)
        hcat = RockstarCatalogReader.read_catalog(tmpfile_name)

    assert hcat.n_halos == 2
    assert np.all(hcat.halo_id == np.array([100, 101]))
    assert hcat.halo_m200b[0] == pytest.approx(5.0e11)
    assert hcat.halo_rg is not None
    assert len(hcat.halo_rg) == 2


def test_rockstar_batch_generator_fastpm_header():
    txt = f"""#Om = {FASTPM_COSMO_OM}; Ol = {FASTPM_COSMO_OL}; h = {FASTPM_COSMO_H}
#ID X Y Z
{_rockstar_data_row(10, 1.0, 2.0, 3.0, 100.0)}
{_rockstar_data_row(20, 4.0, 5.0, 6.0, 200.0)}
"""
    with TemporaryDirectory() as tmpdir:
        tmpfile_name = os.path.join(tmpdir, "out_0.list")
        with open(tmpfile_name, "w") as tmpfile:
            tmpfile.write(txt)

        gen = RockstarCatalogReader.read_catalog_batch_generator(
            tmpfile_name,
            batch_size=1,
        )
        batch1, start1, end1 = next(gen)
        batch2, start2, end2 = next(gen)

    assert batch1.n_halos == 1
    assert batch1.halo_id[0] == 10
    assert batch1.rg_position is not None
    assert start2 == end1
    assert batch2.halo_id[0] == 20
