"""Integration smoke test for SIM-to-FastPM Haloscope (cluster data, small subsets)."""

from pathlib import Path

import pytest

from density_field_properties.pipelines.config import (
    EnvironmentFeaturesConfig,
    GeneralConfig,
    HaloscopeConfig,
    HaloscopeEnrichmentConfig,
    HaloscopePipelineFlowConfig,
    MassCorrectionConfig,
    PreprocessingConfig,
    SampleConfig,
    ValidationConfig,
)
from density_field_properties.pipelines.haloscope_enrichment import (
    run_haloscope_enrichment_pipeline,
)
from density_field_properties.pipelines.run_defaults import (
    default_fastpm_list_path,
    default_sim_hlist_path,
)


@pytest.mark.integration
def test_sim_to_fastpm_haloscope_smoke(tmp_path):
    """
    Run the Haloscope pipeline on capped row counts when cluster catalogs exist.

    Reads only the first N lines from each file (not full 1 GB decompress), so
    memory stays modest while exercising load, env, fit, and Parquet export.
    """
    sim_path = default_sim_hlist_path()
    fastpm_path = default_fastpm_list_path()
    if not sim_path.is_file() or not fastpm_path.is_file():
        pytest.skip("UNIT or FastPM catalog paths not available on this machine")

    sample = SampleConfig(
        box_size_mpc_h=1000.0,
        min_mass_msun_h=1.0e11,
        get_min_mass_lr_from_hmf=False,
        max_mass_msun_h=None,
        max_sim_halos=4000,
        max_fastpm_halos=4000,
    )
    config = HaloscopeEnrichmentConfig(
        run_name="smoke",
        general=GeneralConfig(
            sim_hlist=sim_path,
            fastpm_list=fastpm_path,
            repo_root=Path.cwd(),
            output_dir=Path(tmp_path),
            input_features=["env"],
        ),
        pipeline=HaloscopePipelineFlowConfig(
            environment_features=EnvironmentFeaturesConfig(
                enabled=False,
                n_grid=512,
                cic_batch_size=None,
                descriptor_batch_size=100_000,
                targets=[],
            ),
            preprocessing=PreprocessingConfig(
                enabled=True,
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
                enabled=True,
                input_features=["env"],
                run_holdout_validation=True,
                min_bin_size=5,
                enriched_parquet_name="enriched.parquet",
            ),
            validation=ValidationConfig(enabled=False, n_grid=128),
        ),
    )
    out = run_haloscope_enrichment_pipeline(config)
    if hasattr(out, "output_path"):
        out_path = out.output_path
    else:
        out_path = out
    assert out_path.is_file()
    assert out_path.stat().st_size > 0
