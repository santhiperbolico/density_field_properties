"""Integration smoke test for SIM-to-FastPM Haloscope (cluster data, small subsets)."""

from pathlib import Path

import pytest

from density_field_properties.pipelines.config import HaloscopeEnrichmentConfig
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

    config = HaloscopeEnrichmentConfig(
        sim_hlist_path=sim_path,
        fastpm_list_path=fastpm_path,
        output_dir=Path(tmp_path),
        max_sim_halos=4000,
        max_fastpm_halos=4000,
        min_bin_size=5,
        run_holdout_validation=True,
    )
    out = run_haloscope_enrichment_pipeline(config)
    assert out.is_file()
    assert out.stat().st_size > 0
