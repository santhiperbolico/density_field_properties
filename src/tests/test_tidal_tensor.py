import os
from tempfile import TemporaryDirectory

import h5py
import numpy as np
import pytest

from density_field_properties.tidal_tensor import (
    GAUSSIAN_SCALE_DEFAULT,
    NAME_HD5,
    TidalTensor,
    TidalTensorArray,
    interpolate_array_generator,
    tidal_tensor_component_calculation,
)

MASS_PARTICLE = 1.2e9
BOX_SIZE = 3


@pytest.fixture
def dm_particles() -> np.ndarray:
    data = np.array([[1, 1, 1], [2.9, 1, 0], [0, 0, 2.9], [1, 2, 0], [0, 2.9, 0], [2.9, 0, 0]])
    return data


@pytest.fixture
def density_field() -> np.ndarray:
    expected = np.zeros((3, 3, 3))

    expected[1, 1, 1] += MASS_PARTICLE

    expected[2, 1, 0] += 0.1 * MASS_PARTICLE
    expected[0, 1, 0] += 0.9 * MASS_PARTICLE

    expected[0, 0, 2] += 0.1 * MASS_PARTICLE
    expected[0, 0, 0] += 0.9 * MASS_PARTICLE

    expected[1, 2, 0] += MASS_PARTICLE

    expected[0, 2, 0] += 0.1 * MASS_PARTICLE
    expected[0, 0, 0] += 0.9 * MASS_PARTICLE

    expected[2, 0, 0] += 0.1 * MASS_PARTICLE
    expected[0, 0, 0] += 0.9 * MASS_PARTICLE
    return expected


@pytest.fixture
def tidal_tensor_array(density_field, tmp_path) -> TidalTensorArray:
    gaussian_scales = [2.0, 0.5, 1.0]
    tta = TidalTensorArray.from_delta(
        delta=density_field,
        box_size=BOX_SIZE,
        path=str(tmp_path),
        gaussian_scale_list=gaussian_scales,
    )
    return tta


@pytest.mark.parametrize("gaussian_scale", [None, 1.0])
def test_tidal_tensor_from_delta(gaussian_scale, density_field):
    with TemporaryDirectory() as tmpath:
        tidal_tensor = TidalTensor.from_delta(
            delta=density_field, box_size=BOX_SIZE, path=tmpath, gaussian_scale=gaussian_scale
        )
    assert isinstance(tidal_tensor, TidalTensor)
    assert isinstance(tidal_tensor.tidal_tensor, dict)
    assert len(tidal_tensor.tidal_tensor) == 9


@pytest.mark.parametrize(
    "cell_x, cell_y, cell_z, expected_size",
    [(1, 1, 1, 1), (np.array([1, 2]), np.array([1, 2]), np.array([1, 2]), 2)],
)
def test_tidal_tensor_get_tidal_tensor(cell_x, cell_y, cell_z, expected_size, density_field):
    with TemporaryDirectory() as tmpath:
        tidal_tensor = TidalTensor.from_delta(
            delta=density_field, box_size=BOX_SIZE, path=tmpath, gaussian_scale=1.0
        )
        t_comp = tidal_tensor.get_tidal_tensor(cell_x, cell_y, cell_z)
    assert t_comp.shape == (expected_size, 3, 3)


@pytest.mark.parametrize("gaussian_scale", [None, 1.0])
def test_tidal_tensor_from_folder(gaussian_scale):
    components = [0, 3, 4, 6, 7, 8]
    gaussian_scale_name = gaussian_scale if gaussian_scale is not None else GAUSSIAN_SCALE_DEFAULT
    with TemporaryDirectory() as tmpath:
        default_values = np.zeros((3, 3, 3))
        for component in components:
            file_name = os.path.join(tmpath, f"TidalTensor_{component}_{gaussian_scale_name}.h5")
            with h5py.File(file_name, "w") as f:
                f.create_dataset(NAME_HD5, data=default_values, dtype="float32")
        tidal_tensor = TidalTensor.from_folder(path=tmpath, gaussian_scale=gaussian_scale)
    assert isinstance(tidal_tensor, TidalTensor)
    assert isinstance(tidal_tensor.tidal_tensor, dict)
    assert len(tidal_tensor.tidal_tensor) == 9


def test_tidal_tensor_component_calculation_creates_file_and_dataset(density_field, tmp_path):
    component = (0, 1)
    gaussian_scale = 1.0

    out_file = tidal_tensor_component_calculation(
        delta=density_field,
        box_size=BOX_SIZE,
        path=str(tmp_path),
        component=component,
        gaussian_scale=gaussian_scale,
    )

    assert os.path.isfile(out_file)
    expected_suffix = f"TidalTensor_{component[0] + 3 * component[1]}_{gaussian_scale}.h5"
    assert out_file.endswith(expected_suffix)

    with h5py.File(out_file, "r") as f:
        assert NAME_HD5 in f
        data = f[NAME_HD5][:]
    assert data.shape == density_field.shape
    assert data.dtype == np.float32


def test_tidal_tensor_component_calculation_gaussian_scale_none_uses_default(
    density_field, tmp_path
):
    component = (0, 0)

    out_file = tidal_tensor_component_calculation(
        delta=density_field,
        box_size=BOX_SIZE,
        path=str(tmp_path),
        component=component,
        gaussian_scale=None,
    )

    assert os.path.isfile(out_file)
    expected_suffix = f"TidalTensor_{component[0] + 3 * component[1]}_{GAUSSIAN_SCALE_DEFAULT}.h5"
    assert out_file.endswith(expected_suffix)


def test_interpolate_array_generator_interpolates_linearly():
    array_0 = np.zeros((2, 2))
    array_1 = np.ones((2, 2)) * 10.0
    gs0, gs1 = 0.5, 1.5

    interpolator = interpolate_array_generator(array_0, array_1, gs0, gs1)
    gs_mid = 1.0
    result = interpolator(gs_mid)

    assert result.shape == array_0.shape
    assert np.allclose(result, 5.0)


def test_tidal_tensor__get_tidal_tensor_rejects_non_integer_indices(density_field, tmp_path):
    tidal_tensor = TidalTensor.from_delta(
        delta=density_field, box_size=BOX_SIZE, path=str(tmp_path), gaussian_scale=1.0
    )

    with pytest.raises(ValueError, match="cell_x must be an integer"):
        tidal_tensor._get_tidal_tensor((0, 0), cell_x=1.5, cell_y=1, cell_z=1)

    with pytest.raises(ValueError, match="cell_y must be an integer"):
        tidal_tensor._get_tidal_tensor((0, 0), cell_x=1, cell_y=1.5, cell_z=1)

    with pytest.raises(ValueError, match="cell_z must be an integer"):
        tidal_tensor._get_tidal_tensor((0, 0), cell_x=1, cell_y=1, cell_z=1.5)


def test_tidal_tensor_get_tidal_tensor_is_symmetric(density_field, tmp_path):
    tidal_tensor = TidalTensor.from_delta(
        delta=density_field, box_size=BOX_SIZE, path=str(tmp_path), gaussian_scale=1.0
    )
    t = tidal_tensor.get_tidal_tensor(1, 1, 1)[0]
    assert np.allclose(t, t.T)


def test_tidal_tensor_eigenvalues_scalar_and_array_input(density_field, tmp_path):
    tidal_tensor = TidalTensor.from_delta(
        delta=density_field, box_size=BOX_SIZE, path=str(tmp_path), gaussian_scale=1.0
    )

    eig_single = tidal_tensor.eigenvalues(1, 1, 1)
    assert eig_single.shape == (1, 3)

    xs = np.array([0, 1])
    ys = np.array([1, 1])
    zs = np.array([2, 2])
    eig_multi = tidal_tensor.eigenvalues(xs, ys, zs)
    assert eig_multi.shape == (2, 3)

    assert np.all(np.diff(eig_single[0]) >= -1e-10)
    assert np.all(np.diff(eig_multi[0]) >= -1e-10)
    assert np.all(np.diff(eig_multi[1]) >= -1e-10)


def test_tidal_tensor_array_gaussian_scale_list_sorted(tidal_tensor_array):
    gs_list = tidal_tensor_array.gaussian_scale_list
    assert gs_list == sorted(gs_list)


def test_tidal_tensor_array_get_gaussian_scale_bin_inside_range(tidal_tensor_array):
    gs = np.array([0.75, 1.5])
    gs0, gs1 = tidal_tensor_array.get_gaussian_scale_bin(gs)

    assert np.all(gs0 == np.array([0.5, 1.0]))
    assert np.all(gs1 == np.array([1.0, 2.0]))


def test_tidal_tensor_array_get_gaussian_scale_bin_outside_range_raises(tidal_tensor_array):
    with pytest.raises(ValueError):
        tidal_tensor_array.get_gaussian_scale_bin(0.1)

    with pytest.raises(ValueError):
        tidal_tensor_array.get_gaussian_scale_bin(5.0)


def test_tidal_tensor_array_from_folder_roundtrip(density_field, tmp_path):
    gaussian_scales = [0.5, 1.0]
    TidalTensorArray.from_delta(
        delta=density_field,
        box_size=BOX_SIZE,
        path=str(tmp_path),
        gaussian_scale_list=gaussian_scales,
    )

    base_path = os.path.join(str(tmp_path), "tidal_tensor")
    tta_from_folder = TidalTensorArray.from_folder(base_path)

    assert isinstance(tta_from_folder, TidalTensorArray)
    assert sorted(tta_from_folder.gaussian_scale_list) == sorted(gaussian_scales)


def test_tidal_tensor_array_get_tidal_tensor_shapes_and_interpolation(tidal_tensor_array):
    tt_single = tidal_tensor_array.get_tidal_tensor(
        cell_x=1, cell_y=1, cell_z=1, gaussian_scale=1.0
    )
    assert tt_single.shape == (1, 3, 3)

    xs = np.array([0, 1])
    ys = np.array([1, 1])
    zs = np.array([2, 2])
    gs = np.array([0.75, 1.5])

    tt_multi = tidal_tensor_array.get_tidal_tensor(xs, ys, zs, gs)
    assert tt_multi.shape == (2, 3, 3)

    assert not np.isnan(tt_multi).any()


def test_tidal_tensor_array_get_tidal_tensor_mismatched_shapes_raise(tidal_tensor_array):
    xs = np.array([0, 1])
    ys = np.array([1, 1])
    zs = np.array([2, 2])

    gs_bad = np.array([0.75])
    with pytest.raises(ValueError):
        tidal_tensor_array.get_tidal_tensor(xs, ys, zs, gs_bad)


def test_tidal_tensor_array_eigenvalues(tidal_tensor_array):
    xs = np.array([0, 1])
    ys = np.array([1, 1])
    zs = np.array([2, 2])
    gs = np.array([0.75, 1.5])

    eigvals = tidal_tensor_array.eigenvalues(xs, ys, zs, gs)
    assert eigvals.shape == (2, 3)
    assert not np.isnan(eigvals).any()

    assert np.all(np.diff(eigvals[0]) >= -1e-10)
    assert np.all(np.diff(eigvals[1]) >= -1e-10)


def test_tidal_tensor_array_get_tidal_anisotropy_and_overdensity(tidal_tensor_array):
    xs = np.array([0, 1, 2])
    ys = np.array([0, 1, 2])
    zs = np.array([0, 1, 2])
    gs = np.array([0.75, 1.0, 1.5])

    tidal_anisotropy, delta_s = tidal_tensor_array.get_tidal_anisotropy_and_overdensity(
        xs, ys, zs, gs
    )

    assert tidal_anisotropy.shape == (3,)
    assert delta_s.shape == (3,)

    assert not np.isnan(tidal_anisotropy).any()
    assert not np.isnan(delta_s).any()
    assert np.all(tidal_anisotropy[delta_s > -1] >= 0.0)
    assert np.all(tidal_anisotropy[delta_s < -1] <= 0.0)
