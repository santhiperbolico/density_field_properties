import bz2
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from density_field_properties.density_field.particle_io import (
    detect_dm_particle_format,
    iter_dm_particle_batches,
)

PARTICLE_POSITIONS = np.array([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0], [3.0, 3.0, 3.0], [4.0, 4.0, 4.0]])


class _MockBigFilePositionBlock:
    """Mimics BigFile Position dataset: ``size`` is particle count, not flattened length."""

    def __init__(self, positions: np.ndarray) -> None:
        self._positions = positions
        self.size = positions.shape[0]

    def __getitem__(self, key):
        return self._positions[key]


@pytest.fixture
def text_particles_file():
    with TemporaryDirectory() as tmpdirname:
        path = f"{tmpdirname}/particles.txt"
        np.savetxt(path, PARTICLE_POSITIONS)
        yield path


@pytest.fixture
def mock_bigfile_positions():
    bfile = MagicMock()
    header = MagicMock()
    header.attrs = {}
    bfile.__getitem__.side_effect = lambda key: header if key == "Header" else MagicMock()

    positions = PARTICLE_POSITIONS.copy()

    def open_side_effect(name):
        if name.endswith("/Position"):
            return _MockBigFilePositionBlock(positions)
        raise KeyError(f"Unexpected block name: {name}")

    bfile.open.side_effect = open_side_effect
    return bfile, positions


@pytest.fixture
def bz2_particles_file():
    with TemporaryDirectory() as tmpdirname:
        path = f"{tmpdirname}/particles.txt.bz2"
        with bz2.open(path, mode="wt", encoding="utf-8") as handle:
            np.savetxt(handle, PARTICLE_POSITIONS)
        yield path


def test_detect_bz2_text_file(bz2_particles_file):
    assert detect_dm_particle_format(bz2_particles_file) == "text"


def test_bz2_batches_single_batch(bz2_particles_file):
    batches = list(iter_dm_particle_batches(bz2_particles_file, batch_size=None))
    assert len(batches) == 1
    data, start, end = batches[0]
    assert start == 0
    assert end == 4
    assert np.allclose(data, PARTICLE_POSITIONS)


def test_bz2_batches_chunked(bz2_particles_file):
    batches = list(iter_dm_particle_batches(bz2_particles_file, batch_size=2))
    assert len(batches) == 2
    assert np.allclose(batches[0][0], PARTICLE_POSITIONS[:2])
    assert np.allclose(batches[1][0], PARTICLE_POSITIONS[2:])


def test_detect_text_file(text_particles_file):
    assert detect_dm_particle_format(text_particles_file) == "text"


def test_detect_missing_path_raises():
    with pytest.raises(ValueError, match="does not exist"):
        detect_dm_particle_format("/nonexistent/path/particles.txt")


def test_detect_fastpm_bigfile_directory(mock_bigfile_positions):
    bfile, _ = mock_bigfile_positions
    block_path = "/data/snap_1.0000/1"
    with (
        patch("os.path.isdir", return_value=True),
        patch("os.path.isfile", return_value=False),
        patch("os.path.exists", return_value=True),
        patch("density_field_properties.density_field.particle_io.BigFile", return_value=bfile),
    ):
        assert detect_dm_particle_format(block_path) == "fastpm_bigfile"


def test_detect_directory_without_bigfile_raises():
    block_path = "/data/not_bigfile/1"
    with (
        patch("os.path.normpath", side_effect=lambda p: p),
        patch("os.path.exists", return_value=True),
        patch("os.path.isfile", return_value=False),
        patch("os.path.isdir", return_value=True),
        patch(
            "density_field_properties.density_field.particle_io.BigFile",
            side_effect=OSError("not a bigfile"),
        ),
    ):
        with pytest.raises(ValueError, match="FastPM BigFile"):
            detect_dm_particle_format(block_path)


def test_text_batches_single_batch(text_particles_file):
    batches = list(iter_dm_particle_batches(text_particles_file, batch_size=None))
    assert len(batches) == 1
    data, start, end = batches[0]
    assert start == 0
    assert end == 4
    assert np.allclose(data, PARTICLE_POSITIONS)


def test_text_batches_chunked(text_particles_file):
    batches = list(iter_dm_particle_batches(text_particles_file, batch_size=2))
    assert len(batches) == 2
    assert batches[0][1] == 0 and batches[0][2] == 2
    assert batches[1][1] == 2 and batches[1][2] == 4
    assert np.allclose(batches[0][0], PARTICLE_POSITIONS[:2])
    assert np.allclose(batches[1][0], PARTICLE_POSITIONS[2:])


def test_text_batches_larger_than_n(text_particles_file):
    batches = list(iter_dm_particle_batches(text_particles_file, batch_size=100))
    assert len(batches) == 1
    assert batches[0][2] - batches[0][1] == 4


def test_bigfile_batches(mock_bigfile_positions):
    bfile, _ = mock_bigfile_positions
    block_path = "/data/snap_1.0000/1"
    with (
        patch("os.path.normpath", side_effect=lambda p: p),
        patch("os.path.exists", return_value=True),
        patch("os.path.isfile", return_value=False),
        patch("os.path.isdir", return_value=True),
        patch("density_field_properties.density_field.particle_io.BigFile", return_value=bfile),
    ):
        batches = list(iter_dm_particle_batches(block_path, batch_size=2))
    assert len(batches) == 2
    assert batches[0][1] == 0 and batches[0][2] == 2
    assert batches[1][1] == 2 and batches[1][2] == 4


def test_bigfile_batches_single_when_batch_size_none(mock_bigfile_positions):
    bfile, _ = mock_bigfile_positions
    block_path = "/data/snap_1.0000/1"
    with (
        patch("os.path.normpath", side_effect=lambda p: p),
        patch("os.path.exists", return_value=True),
        patch("os.path.isfile", return_value=False),
        patch("os.path.isdir", return_value=True),
        patch("density_field_properties.density_field.particle_io.BigFile", return_value=bfile),
    ):
        batches = list(iter_dm_particle_batches(block_path, batch_size=None))
    assert len(batches) == 1
    assert batches[0][2] == 4
