"""Unit tests for SIM-to-FastPM assembly-bias helpers."""

from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pytest

from density_field_properties.density_field.cic_deposit import (
    DensityFieldInfo,
    save_density_field_cic,
)
from density_field_properties.haloscope.sim_to_fastpm.assembly_bias import (
    assembly_bias_curves_for_catalog,
    halo_overdensity_field_cic,
    joint_assembly_masks,
    load_matter_overdensity,
    paranjape_halo_by_halo_bias,
    solve_joint_percentile,
)

BOX_SIZE = 3.0
MASS_PARTICLE = 1.2e9


@pytest.mark.parametrize("n_properties", [2, 4])
def test_joint_masks_target_fraction(n_properties):
    """Joint tails should bracket the requested population fraction."""
    rng = np.random.default_rng(0)
    properties = rng.normal(size=(5000, n_properties))
    lower_mask, upper_mask, percentile = joint_assembly_masks(properties, target_fraction=0.25)
    assert 0.0 < percentile < 100.0
    assert 0.15 < lower_mask.mean() < 0.35
    assert 0.15 < upper_mask.mean() < 0.35


def test_solve_joint_percentile_monotonic():
    """Larger target fractions require a higher per-property percentile."""
    properties = np.column_stack([np.linspace(0.0, 1.0, 200), np.linspace(0.0, 1.0, 200)])
    low_p = solve_joint_percentile(properties, target_fraction=0.10, tail="lower")
    high_p = solve_joint_percentile(properties, target_fraction=0.30, tail="lower")
    assert high_p > low_p


def test_paranjape_bias_constant_field_near_zero():
    """A uniform overdensity field should yield near-zero halo bias."""
    n_grid = 32
    boxsize = 100.0
    delta_field = np.zeros((n_grid, n_grid, n_grid), dtype=np.float64)
    positions = np.array([[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]])
    bias = paranjape_halo_by_halo_bias(positions, delta_field, boxsize, k_max_h_mpc=0.2)
    assert np.allclose(bias, 0.0, atol=1e-10)


def test_load_matter_overdensity_saved_cic():
    expected_mass_field = np.zeros((3, 3, 3))
    expected_mass_field[1, 1, 1] = 6 * MASS_PARTICLE
    with TemporaryDirectory() as tmpdirname:
        dm_particles_file = f"{tmpdirname}/dm_particles.txt"
        density_info = DensityFieldInfo(
            n_grid=3, box_size=BOX_SIZE, n_particles=6, mass_particle=MASS_PARTICLE
        )
        output_file = save_density_field_cic(
            expected_mass_field, tmpdirname, dm_particles_file, density_info
        )
        output_info_file = f"{tmpdirname}/dm_particles_density_info.txt"
        delta, mode = load_matter_overdensity(
            tmpdirname,
            BOX_SIZE,
            3,
            MASS_PARTICLE,
            dm_particles_path=None,
            saved_density_path=Path(output_file).name,
            saved_density_info_path=Path(output_info_file).name,
        )
        assert mode == "saved CIC"
        assert delta is not None
        assert delta.shape == (3, 3, 3)


def test_halo_overdensity_field_cic_uniform_weights():
    """CIC halo field should have mean overdensity near zero on a small periodic grid."""
    positions = np.array([[0.5, 0.5, 0.5], [1.5, 1.5, 1.5]], dtype=np.float64)
    weights = np.array([1.0, 1.0], dtype=np.float64)
    delta = halo_overdensity_field_cic(positions, weights, boxsize_mpc_h=3.0, n_grid=3)
    assert delta.shape == (3, 3, 3)
    assert np.isclose(delta.mean(), 0.0, atol=1e-12)


def test_assembly_bias_curves_returns_finite_bins():
    """Binned curves should return aligned mass and bias arrays."""
    rng = np.random.default_rng(1)
    mass = 10 ** rng.uniform(11.0, 13.0, size=800)
    bias = 1.0 + 0.2 * rng.normal(size=800)
    properties = rng.normal(size=(800, 4))
    bin_edges = np.linspace(11.0, 13.0, 6)
    curves = assembly_bias_curves_for_catalog(mass, bias, properties, bin_edges)
    lower_mass, lower_bias, _, upper_mass, upper_bias, _ = curves
    assert len(lower_mass) == len(lower_bias)
    assert len(upper_mass) == len(upper_bias)
    assert np.all(lower_mass > 0.0)
    assert np.all(upper_mass > 0.0)
