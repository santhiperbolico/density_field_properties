import os
from tempfile import TemporaryDirectory

import numpy as np
import pytest

from density_field_properties.cosmology import Cosmology
from density_field_properties.read_data.halos.rockstar import (
    ROCKSTAR_HALO_COLUMNS_POSITION,
    RockstarCatalogReader,
    read_rockstar_cosmology_header,
    read_rockstar_id_header_column_names,
    rockstar_pid_column_index,
)

FASTPM_COSMO_OM = 0.308900
FASTPM_COSMO_OL = 0.691100
FASTPM_COSMO_H = 0.677400

FASTPM_HEADER = (
    "#ID DescID Mvir Vmax Vrms Rvir Rs Np X Y Z VX VY VZ JX JY JZ Spin "
    "rs_klypin Mvir_all M200b M200c M500c M2500c Xoff Voff spin_bullock "
    "b_to_a c_to_a A[x] A[y] A[z] b_to_a(500c) c_to_a(500c) A[x](500c) "
    "A[y](500c) A[z](500c) T/|U| M_pe_Behroozi M_pe_Diemer Halfmass_Radius "
    "rvmax NFW_chi2 Ixx Iyy Izz Ixy Iyz Izx Ixx(500c) Iyy(500c) Izz(500c) "
    "Ixy(500c) Iyz(500c) Izx(500c)"
)

UNIT_HEADER_WITH_PID = (
    "#ID DescID Mvir Vmax Vrms Rvir Rs Np X Y Z VX VY VZ JX JY JZ Spin "
    "rs_klypin Mvir_all M200b M200c M500c M2500c Xoff Voff spin_bullock "
    "b_to_a c_to_a A[x] A[y] A[z] M_pe_Behroozi PID"
)

HLIST_SCHEMA_HEADER = (
    "#scale(0) id(1) desc_scale(2) desc_id(3) num_prog(4) pid(5) upid(6) " "M200b(39)"
)


def _fastpm_row(halo_id, x, y, z, mass):
    columns = ["0"] * 55
    columns[0] = str(halo_id)
    columns[8] = str(x)
    columns[9] = str(y)
    columns[10] = str(z)
    columns[20] = str(mass)
    return " ".join(columns)


def _unit_row_with_pid(halo_id, pid, x, y, z, mass):
    columns = ["0"] * 34
    columns[0] = str(halo_id)
    columns[8] = str(x)
    columns[9] = str(y)
    columns[10] = str(z)
    columns[20] = str(mass)
    columns[33] = str(pid)
    return " ".join(columns)


def test_rockstar_header_column_names_reads_id_line(tmp_path):
    path = tmp_path / "out_8.list"
    path.write_text(
        FASTPM_HEADER + "\n" + _fastpm_row(1, 10.0, 20.0, 30.0, 100.0) + "\n",
        encoding="utf-8",
    )
    header = read_rockstar_id_header_column_names(str(path))
    assert header is not None
    assert header[0] == "ID"
    assert "PID" not in header


def test_rockstar_header_column_names_reads_hlist_schema_line(tmp_path):
    path = tmp_path / "hlist_1.00000.list"
    path.write_text(
        HLIST_SCHEMA_HEADER + "\n#Omega_M = 0.308900; Omega_L = 0.691100; h0 = 0.677400\n",
        encoding="utf-8",
    )
    header = read_rockstar_id_header_column_names(str(path))
    assert header is not None
    assert header[0] == "scale(0)"
    assert "pid(5)" in header


def test_rockstar_pid_column_index_reads_hlist_pid_field(tmp_path):
    path = tmp_path / "hlist_1.00000.list"
    path.write_text(HLIST_SCHEMA_HEADER + "\n", encoding="utf-8")
    assert rockstar_pid_column_index(str(path)) == 5


def test_rockstar_pid_column_index_short_find_parents_output(tmp_path):
    header = FASTPM_HEADER + " PID"
    columns = ["0"] * 41
    columns[0] = "1"
    columns[20] = "1e12"
    columns[40] = "-1"
    path = tmp_path / "out_8_with_pid.list"
    path.write_text(header + "\n" + " ".join(columns) + "\n", encoding="utf-8")
    assert rockstar_pid_column_index(str(path)) == 40


def test_rockstar_pid_column_index_uses_header(tmp_path):
    fastpm_path = tmp_path / "out_8.list"
    unit_path = tmp_path / "out_128p.list"
    fastpm_path.write_text(
        FASTPM_HEADER + "\n" + _fastpm_row(1, 10.0, 20.0, 30.0, 100.0) + "\n",
        encoding="utf-8",
    )
    unit_path.write_text(
        UNIT_HEADER_WITH_PID + "\n" + _unit_row_with_pid(1, -1, 10.0, 20.0, 30.0, 100.0) + "\n",
        encoding="utf-8",
    )
    assert rockstar_pid_column_index(str(fastpm_path)) is None
    assert rockstar_pid_column_index(str(unit_path)) == 33


def test_rockstar_reader_fastpm_list_without_pid_raises_when_filtering(tmp_path):
    path = tmp_path / "out_8.list"
    path.write_text(
        FASTPM_HEADER + "\n" + _fastpm_row(1, 10.0, 20.0, 30.0, 100.0) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="no PID column"):
        RockstarCatalogReader.read_catalog(
            str(path),
            filter_by_pid=True,
            halo_id_position=0,
            halo_x_position=8,
            halo_y_position=9,
            halo_z_position=10,
            halo_m200b_position=20,
            halo_pid_position=40,
        )


def test_rockstar_reader_fastpm_list_without_pid_loads_all_halos_when_disabled(tmp_path):
    path = tmp_path / "out_8.list"
    rows = [
        _fastpm_row(1, 10.0, 20.0, 30.0, 100.0),
        _fastpm_row(2, 40.0, 50.0, 60.0, 200.0),
    ]
    path.write_text(FASTPM_HEADER + "\n" + "\n".join(rows) + "\n", encoding="utf-8")

    hcat = RockstarCatalogReader.read_catalog(
        str(path),
        filter_by_pid=False,
        halo_id_position=0,
        halo_x_position=8,
        halo_y_position=9,
        halo_z_position=10,
        halo_m200b_position=20,
        halo_pid_position=40,
    )

    assert hcat.n_halos == 2


@pytest.mark.parametrize("pid_position", [33, 40, 70])
def test_rockstar_reader_unit_pid_header_filters_subhalos(tmp_path, pid_position):
    path = tmp_path / "out_128p.list"
    path.write_text(
        UNIT_HEADER_WITH_PID
        + "\n"
        + _unit_row_with_pid(1, -1, 1.0, 2.0, 3.0, 100.0)
        + "\n"
        + _unit_row_with_pid(2, 5, 4.0, 5.0, 6.0, 200.0)
        + "\n"
        + _unit_row_with_pid(3, -1, 7.0, 8.0, 9.0, 300.0)
        + "\n",
        encoding="utf-8",
    )
    hcat = RockstarCatalogReader.read_catalog(
        str(path),
        filter_by_pid=True,
        halo_id_position=0,
        halo_x_position=8,
        halo_y_position=9,
        halo_z_position=10,
        halo_m200b_position=20,
        halo_pid_position=pid_position,
    )
    assert hcat.n_halos == 2
    assert np.all(hcat.halo_id == np.array([1, 3]))


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
            filter_by_pid=False,
        )

    assert hcat.n_halos == 2
    assert np.all(hcat.halo_id == np.array([10, 20]))
    assert np.all(hcat.halo_m200b == np.array([100.0, 200.0]))


def test_rockstar_reader_respects_n_lines(tmp_path):
    """
    read_catalog should stop after the requested number of data rows.
    """
    txt = """# Comment
0 10 0 0 0  1 2 3 4 5  39  1.0 2.0 3.0  100.0
1 20 0 0 0  1 2 3 4 5  39  4.0 5.0 6.0  200.0
2 30 0 0 0  1 2 3 4 5  39  7.0 8.0 9.0  300.0
"""
    catalog_path = tmp_path / "rockstar.txt"
    catalog_path.write_text(txt, encoding="utf-8")

    hcat = RockstarCatalogReader.read_catalog(
        str(catalog_path),
        n_lines=2,
        halo_x_position=11,
        halo_y_position=12,
        halo_z_position=13,
        halo_m200b_position=14,
        filter_by_pid=False,
    )

    assert hcat.n_halos == 2
    assert np.all(hcat.halo_id == np.array([10, 20]))


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
            filter_by_pid=False,
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
        hcat = RockstarCatalogReader.read_catalog(tmpfile_name, filter_by_pid=False)

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
        hcat = RockstarCatalogReader.read_catalog(tmpfile_name, filter_by_pid=False)

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
            filter_by_pid=False,
        )
        batch1, start1, end1 = next(gen)
        batch2, start2, end2 = next(gen)

    assert batch1.n_halos == 1
    assert batch1.halo_id[0] == 10
    assert batch1.rg_position is not None
    assert start2 == end1
    assert batch2.halo_id[0] == 20
