"""Tests for mass calibration benchmark helpers and FoF catalog loading."""

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from density_field_properties.preprocessing.catalog_loaders import (
    load_fastpm_fof_target_catalog,
    load_fastpm_rockstar_for_mass_calibration,
    load_fastpm_target_catalog,
    rockstar_list_supports_host_pid_filter,
)
from density_field_properties.preprocessing.mass_calibration_ml import build_ml_feature_matrix

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from mass_calibration_benchmark_helpers import (  # noqa: E402
    FORERO_ML_FEATURE_NAMES,
    LOADER_FASTPM_FOF,
    LOADER_ROCKSTAR_LIST,
    parse_benchmark_catalogs,
    parse_benchmark_hmf_config,
    parse_benchmark_methods,
    resolve_catalog_methods,
    rmse_log_density_vs_reference,
)

FASTPM_BIGFILE_PATCH = "density_field_properties.preprocessing.catalog_loaders.FastPMCatalogReader"


def test_parse_benchmark_catalogs_resolves_relative_paths(tmp_path):
    """
    Relative catalog paths are resolved against ``repo_root``.
    """
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    catalog_path = repo_root / "output/fastpm/out_8.list"
    catalog_path.parent.mkdir(parents=True)
    catalog_path.write_text("# header\n", encoding="utf-8")
    payload = {
        "benchmark_catalogs": {
            "rockstar_pm": {
                "path": "output/fastpm/out_8.list",
                "loader": LOADER_ROCKSTAR_LIST,
            }
        }
    }
    entries = parse_benchmark_catalogs(payload, repo_root)
    assert len(entries) == 1
    assert entries[0].key == "rockstar_pm"
    assert entries[0].path == catalog_path.resolve()


def test_parse_benchmark_catalogs_rejects_unknown_loader():
    """
    Unknown loader keys raise ``ValueError``.
    """
    payload = {"benchmark_catalogs": {"bad": {"path": "/tmp/x", "loader": "unknown"}}}
    with pytest.raises(ValueError, match="loader must be one of"):
        parse_benchmark_catalogs(payload, Path("/tmp"))


def test_parse_benchmark_methods_defaults():
    """
    Default methods include abundance matching and matching_ml.
    """
    methods = parse_benchmark_methods({})
    assert "abundance_matching" in methods
    assert "matching_ml" in methods


def test_rmse_log_density_vs_reference_ignores_non_finite():
    """
    RMSE uses only bins where both curves are finite.
    """
    reference = np.array([0.0, 1.0, np.nan])
    target = np.array([0.5, 1.5, 1.0])
    rmse = rmse_log_density_vs_reference(reference, target)
    assert rmse == pytest.approx(0.5)


@patch(f"{FASTPM_BIGFILE_PATCH}.read_catalog")
def test_load_fastpm_fof_target_catalog_builds_dataframe(mock_read_catalog):
    """
    FoF groups are exposed with the standard preprocessing columns.
    """
    mock_catalog = MagicMock()
    mock_catalog.halo_id = np.array([1, 2], dtype=np.int64)
    mock_catalog.halo_x = np.array([1.0, 2.0])
    mock_catalog.halo_y = np.array([3.0, 4.0])
    mock_catalog.halo_z = np.array([5.0, 6.0])
    mock_catalog.halo_m200b = np.array([1.0e13, 2.0e13])
    mock_read_catalog.return_value = mock_catalog

    frame = load_fastpm_fof_target_catalog(Path("/data/fof/LL-0.200"), max_halos=2)
    mock_read_catalog.assert_called_once_with("/data/fof/LL-0.200", n_lines=2)
    assert list(frame.columns) == ["id", "x", "y", "z", "M200b"]
    assert len(frame) == 2


def test_parse_benchmark_hmf_config_defaults():
    """
    Default HMF limits match Ramakrishnan et al. (2025) when block is absent.
    """
    config = parse_benchmark_hmf_config({})
    assert config.min_mass_msun_h == pytest.approx(2.0e11)
    assert config.max_mass_msun_h == pytest.approx(3.2e14)


def test_forero_ml_feature_names_match_repo_default():
    """
    Documented Forero-like feature list matches ``DEFAULT_ML_FEATURES``.
    """
    assert FORERO_ML_FEATURE_NAMES == ("log10_M200b", "delta", "log10_sigma_v")


def test_build_ml_feature_matrix_accepts_delta_from_overdensity():
    """
    ``delta`` resolves from an ``overdensity`` column (Haloscope naming).
    """
    lr = pd.DataFrame(
        {
            "M200b": [1.0e13, 2.0e13],
            "overdensity": [0.1, -0.2],
            "sigma_v": [100.0, 150.0],
        }
    )
    matrix = build_ml_feature_matrix(
        lr,
        ("log10_M200b", "delta", "log10_sigma_v"),
        lr_mass_column="M200b",
    )
    assert matrix.shape == (2, 3)
    assert matrix[0, 1] == pytest.approx(0.1)


def test_load_fastpm_rockstar_for_mass_calibration_keeps_hosts_only():
    """
    Mass-calibration Rockstar loader keeps ``PID == -1`` on local_sample lists.
    """
    list_path = REPO_ROOT / "output/local_sample/fastpm/rockstar_out_pm/out_8.list"
    if not list_path.is_file():
        pytest.skip("local_sample Rockstar list not available.")
    assert rockstar_list_supports_host_pid_filter(list_path) is True
    frame = load_fastpm_rockstar_for_mass_calibration(list_path, max_halos=200)
    all_hosts = load_fastpm_target_catalog(list_path, max_halos=5000)
    assert len(frame) < len(all_hosts)
    assert "sigma_v" in frame.columns


def test_load_fastpm_target_catalog_includes_sigma_v():
    """
    Rockstar FastPM loader exposes ``sigma_v`` from ``Vrms``.
    """
    repo_root = Path(__file__).resolve().parents[3]
    list_path = repo_root / "output/local_sample/fastpm/rockstar_out_pm/out_8.list"
    if not list_path.is_file():
        pytest.skip("local_sample Rockstar list not available.")
    frame = load_fastpm_target_catalog(list_path, max_halos=5)
    assert "sigma_v" in frame.columns
    assert np.all(frame["sigma_v"] > 0.0)


def test_benchmark_config_json_parses_catalogs():
    """
    The local_sample benchmark JSON contains three FastPM catalogs including FoF.
    """
    config_path = (
        Path(__file__).resolve().parents[3]
        / "config/local/mass_calibration_four_catalogs_local_sample.json"
    )
    if not config_path.is_file():
        pytest.skip("Benchmark config not present in this checkout.")
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    entries = parse_benchmark_catalogs(payload, config_path.parent.parent.parent)
    keys = {entry.key for entry in entries}
    assert keys == {"rockstar_nbody", "rockstar_pm", "fastpm_fof"}
    fof_entry = next(entry for entry in entries if entry.loader == LOADER_FASTPM_FOF)
    assert fof_entry.path.name == "LL-0.200"
    hmf = parse_benchmark_hmf_config(payload)
    assert hmf.min_mass_msun_h == pytest.approx(2.0e11)


def test_taurus_benchmark_config_runs_matching_ml_for_fof():
    """
    Taurus benchmark JSON runs ``matching_ml`` for FoF (full box, no coord offset).
    """
    config_path = REPO_ROOT / "config/mass_calibration_four_catalogs_taurus.json"
    if not config_path.is_file():
        pytest.skip("Taurus benchmark config not present in this checkout.")
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    default_methods = parse_benchmark_methods(payload)
    entries = parse_benchmark_catalogs(payload, REPO_ROOT)
    fof_entry = next(entry for entry in entries if entry.key == "fastpm_fof")
    assert fof_entry.methods is None
    catalog_methods = resolve_catalog_methods(fof_entry, default_methods)
    assert catalog_methods == ("abundance_matching", "matching_ml")
    train = payload["mass_calibration"]["matching_ml"]["train_subbox"]
    assert train["x_max_mpc_h"] == pytest.approx(500.0)
    assert payload["sample"]["box_size_mpc_h"] == pytest.approx(1000.0)
