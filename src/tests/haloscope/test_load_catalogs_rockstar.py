import logging
from tempfile import TemporaryDirectory

import pytest

from density_field_properties.haloscope.sim_to_fastpm.load_catalogs import (
    _rockstar_data_column_count,
    _rockstar_pid_column_index,
    load_fastpm_central_target_catalog,
    load_unit_rockstar_target_catalog,
)


def _compact_row(halo_id, x, y, z, mass):
    columns = ["0"] * 34
    columns[0] = str(halo_id)
    columns[1] = "-1"
    columns[8] = str(x)
    columns[9] = str(y)
    columns[10] = str(z)
    columns[20] = str(mass)
    return " ".join(columns)


def _extended_row(halo_id, pid, x, y, z, mass):
    columns = ["0"] * 55
    columns[0] = str(halo_id)
    columns[1] = "-1"
    columns[8] = str(x)
    columns[9] = str(y)
    columns[10] = str(z)
    columns[20] = str(mass)
    columns[33] = str(pid)
    return " ".join(columns)


def test_rockstar_pid_column_index_requires_extended_layout():
    with TemporaryDirectory() as tmpdir:
        compact_path = f"{tmpdir}/compact.list"
        extended_path = f"{tmpdir}/extended.list"
        with open(compact_path, "w", encoding="utf-8") as handle:
            handle.write("# header\n")
            handle.write(_compact_row(1, 10.0, 20.0, 30.0, 100.0) + "\n")
        with open(extended_path, "w", encoding="utf-8") as handle:
            handle.write("# header\n")
            handle.write(_extended_row(1, -1, 10.0, 20.0, 30.0, 100.0) + "\n")

        assert _rockstar_data_column_count(compact_path) == 34
        assert _rockstar_pid_column_index(compact_path, 33) is None
        assert _rockstar_data_column_count(extended_path) == 55
        assert _rockstar_pid_column_index(extended_path, 33) == 33


def test_load_fastpm_central_target_catalog_uses_all_halos_without_pid(caplog):
    with TemporaryDirectory() as tmpdir:
        path = f"{tmpdir}/out_8.list"
        rows = [
            _compact_row(1, 10.0, 20.0, 30.0, 100.0),
            _compact_row(2, 40.0, 50.0, 60.0, 200.0),
        ]
        with open(path, "w", encoding="utf-8") as handle:
            handle.write("# header\n")
            handle.write("\n".join(rows) + "\n")

        with caplog.at_level(logging.WARNING):
            frame = load_fastpm_central_target_catalog(path, max_centrals=10)

    assert len(frame) == 2
    assert "no PID column" in caplog.text


@pytest.mark.parametrize(
    "pid_values, expected_count",
    [
        ([-1, 5, -1], 2),
        ([5, 6, 7], 0),
    ],
)
def test_load_unit_rockstar_target_catalog_filters_pid(pid_values, expected_count):
    with TemporaryDirectory() as tmpdir:
        path = f"{tmpdir}/out_128p.list"
        rows = [
            _extended_row(index + 1, pid, float(index), float(index), float(index), 100.0)
            for index, pid in enumerate(pid_values)
        ]
        with open(path, "w", encoding="utf-8") as handle:
            handle.write("# header\n")
            handle.write("\n".join(rows) + "\n")

        if expected_count == 0:
            with pytest.raises(ValueError, match="No halos found"):
                load_unit_rockstar_target_catalog(path, max_centrals=10)
        else:
            frame = load_unit_rockstar_target_catalog(path, max_centrals=10)
            assert len(frame) == expected_count


def test_load_unit_rockstar_target_catalog_can_include_subhalos():
    with TemporaryDirectory() as tmpdir:
        path = f"{tmpdir}/out_128p.list"
        rows = [
            _extended_row(1, -1, 1.0, 2.0, 3.0, 100.0),
            _extended_row(2, 1, 4.0, 5.0, 6.0, 200.0),
        ]
        with open(path, "w", encoding="utf-8") as handle:
            handle.write("# header\n")
            handle.write("\n".join(rows) + "\n")

        frame = load_unit_rockstar_target_catalog(path, max_centrals=10, central_only=False)
        assert len(frame) == 2


def test_collect_rockstar_halos_uses_reservoir_sampling_when_capped():
    with TemporaryDirectory() as tmpdir:
        path = f"{tmpdir}/out_8.list"
        rows = [_compact_row(index + 1, float(index), 0.0, 0.0, 100.0) for index in range(100)]
        with open(path, "w", encoding="utf-8") as handle:
            handle.write("# header\n")
            handle.write("\n".join(rows) + "\n")

        frame = load_fastpm_central_target_catalog(path, max_centrals=10)
        assert len(frame) == 10
        assert frame["x"].mean() > 20.0


def test_load_fastpm_central_target_catalog_reads_all_rows_when_uncapped():
    with TemporaryDirectory() as tmpdir:
        path = f"{tmpdir}/out_8.list"
        rows = [
            _compact_row(1, 1.0, 2.0, 3.0, 100.0),
            _compact_row(2, 4.0, 5.0, 6.0, 200.0),
            _compact_row(3, 7.0, 8.0, 9.0, 300.0),
        ]
        with open(path, "w", encoding="utf-8") as handle:
            handle.write("# header\n")
            handle.write("\n".join(rows) + "\n")

        frame = load_fastpm_central_target_catalog(path, max_centrals=None)
        assert len(frame) == 3
