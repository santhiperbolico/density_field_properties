"""Tests for assembly-bias panel helpers."""

from pathlib import Path

import pytest

from density_field_properties.pipelines.config import (
    PIPELINE_TARGET_FASTPM,
    PIPELINE_TARGET_UNIT,
    EnvironmentFeaturesConfig,
    GeneralConfig,
    HaloscopeConfig,
    HaloscopeEnrichmentConfig,
    HaloscopePipelineFlowConfig,
    MassCorrectionConfig,
    PreprocessingConfig,
    SampleConfig,
    TidalAnisotropyTargetConfig,
    ValidationConfig,
)


def _assembly_bias_config(tmp_path: Path) -> HaloscopeEnrichmentConfig:
    """
    Build a minimal Haloscope config for assembly-bias path resolution tests.

    Parameters
    ----------
    tmp_path : Path
        Temporary directory.

    Returns
    -------
    HaloscopeEnrichmentConfig
        Config with tidal targets and sample box size.
    """
    sample = SampleConfig(
        box_size_mpc_h=200.0,
        min_mass_msun_h=1.0e11,
        get_min_mass_lr_from_hmf=False,
        max_mass_msun_h=None,
        max_sim_halos=None,
        max_fastpm_halos=None,
    )
    return HaloscopeEnrichmentConfig(
        run_name="ab",
        general=GeneralConfig(
            sim_hlist=tmp_path / "sim.list",
            fastpm_list=tmp_path / "fastpm.list",
            repo_root=tmp_path,
            output_dir=tmp_path / "out",
        ),
        pipeline=HaloscopePipelineFlowConfig(
            environment_features=EnvironmentFeaturesConfig(
                enabled=True,
                n_grid=512,
                cic_batch_size=None,
                descriptor_batch_size=1000,
                targets=[
                    TidalAnisotropyTargetConfig(
                        name=PIPELINE_TARGET_UNIT,
                        work_dir=Path("output/tidal_mn5/unit"),
                    ),
                    TidalAnisotropyTargetConfig(
                        name=PIPELINE_TARGET_FASTPM,
                        work_dir=Path("output/tidal_mn5/fastpm"),
                    ),
                ],
            ),
            preprocessing=PreprocessingConfig(
                enabled=False,
                hr_parquet_name="hr.parquet",
                lr_parquet_name="lr.parquet",
                sample=sample,
                mass_correction=MassCorrectionConfig(
                    method="abundance_matching",
                    correction_column="M200b_cal",
                    params_method={},
                ),
            ),
            haloscope=HaloscopeConfig(
                enabled=False,
                input_features=[],
                run_holdout_validation=False,
                min_bin_size=5,
                enriched_parquet_name="enriched.parquet",
            ),
            validation=ValidationConfig(enabled=False, n_grid=128),
        ),
    )


@pytest.mark.parametrize(
    "input_features,expected_feature_text",
    [
        (("env",), "env"),
        (("t_over_u", "tidal_anisotropy"), "T/|U| + tidal anisotropy"),
        (("env", "t_over_u"), "env + T/|U|"),
    ],
)
def test_format_assembly_bias_panel_title(input_features, expected_feature_text):
    """
    Panel titles should reflect the Haloscope INPUT features of the run.
    """
    pytest.importorskip("bigfile")
    from density_field_properties.validation.assembly_bias_panel import (
        format_assembly_bias_panel_title,
    )

    title = format_assembly_bias_panel_title(
        input_features,
        sim_delta_mode="saved CIC",
        fastpm_delta_mode="halo CIC",
    )

    assert title == (f"input: {expected_feature_text} " "(HR δ: saved CIC, FastPM δ: halo CIC)")


def test_resolve_saved_cic_density_paths_use_tidal_work_dirs(tmp_path):
    """
    Assembly bias should prefer CIC grids under pipeline tidal work directories.
    """
    from density_field_properties.validation.assembly_bias_panel import (
        _resolve_saved_cic_density_paths,
    )

    unit_work = tmp_path / "output" / "tidal_mn5" / "unit"
    fastpm_work = tmp_path / "output" / "tidal_mn5" / "fastpm"
    unit_work.mkdir(parents=True)
    fastpm_work.mkdir(parents=True)
    unit_density = unit_work / "dm_particles_0.5_128_density"
    unit_info = unit_work / "dm_particles_0.5_128_density_info.txt"
    fastpm_density = fastpm_work / "snap_1.0000_density"
    fastpm_info = fastpm_work / "snap_1.0000_density_info.txt"
    unit_density.write_bytes(b"u")
    unit_info.write_text("#box_size,n_grid,n_particles,mass_particle\n", encoding="utf-8")
    fastpm_density.write_bytes(b"f")
    fastpm_info.write_text("#box_size,n_grid,n_particles,mass_particle\n", encoding="utf-8")

    config = _assembly_bias_config(tmp_path)

    (sim_density_path, sim_info_path), (fastpm_density_path, fastpm_info_path) = (
        _resolve_saved_cic_density_paths(tmp_path, config)
    )

    assert sim_density_path == unit_density
    assert sim_info_path == unit_info
    assert fastpm_density_path == fastpm_density
    assert fastpm_info_path == fastpm_info


def test_resolve_assembly_bias_context_uses_run_config(tmp_path):
    """
    Assembly-bias context should honor box size and DM paths from the run config.
    """
    from density_field_properties.validation.assembly_bias_panel import (
        _resolve_assembly_bias_context,
    )

    unit_dm = tmp_path / "unit_dm.dat"
    fastpm_dm = tmp_path / "fastpm_dm"
    config = _assembly_bias_config(tmp_path)
    config.pipeline.environment_features.targets = [
        TidalAnisotropyTargetConfig(
            name=PIPELINE_TARGET_UNIT,
            dm_particles_file=unit_dm,
            mass_particle=9.9e8,
            work_dir=tmp_path / "unit",
        ),
        TidalAnisotropyTargetConfig(
            name=PIPELINE_TARGET_FASTPM,
            dm_particles_file=fastpm_dm,
            work_dir=tmp_path / "fastpm",
        ),
    ]

    sim_box, fastpm_box, sim_dm, fastpm_dm_path, dm_mass = _resolve_assembly_bias_context(config)

    assert sim_box == 200.0
    assert fastpm_box == 200.0
    assert sim_dm == unit_dm
    assert fastpm_dm_path == fastpm_dm
    assert dm_mass == 9.9e8
