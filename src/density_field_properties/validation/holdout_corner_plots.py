"""Hold-out SIM validation corner plots for Haloscope runs."""

from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from density_field_properties.haloscope.training import holdout_validate_sim_bins
from density_field_properties.pipelines.config import HOLDOUT_CORNER_PLOT_FILENAME_TEMPLATE
from density_field_properties.validation.marginals import corner_plot_sim_validation
from density_field_properties.validation.memory import release_validation_memory

_INPUT_FEATURE_LABELS = {
    "env": "env",
    "t_over_u": "T/|U|",
    "tidal_anisotropy": "tidal anisotropy",
}


def format_holdout_corner_plot_title(
    input_features: Sequence[str],
    bin_index: int,
) -> str:
    """
    Build the corner-plot title for a SIM hold-out mass bin.

    Parameters
    ----------
    input_features : Sequence[str]
        Haloscope INPUT feature names for the run.
    bin_index : int
        Mass-bin index used by ``holdout_validate_sim_bins``.

    Returns
    -------
    str
        Figure suptitle.
    """
    labels = [_INPUT_FEATURE_LABELS.get(feature, feature) for feature in input_features]
    feature_text = " + ".join(labels)
    return f"SIM hold-out validation — mass bin {bin_index} (input: {feature_text})"


def holdout_corner_plot_path(output_dir: Path, bin_index: int) -> Path:
    """
    Resolve the output PDF path for one hold-out mass bin.

    Parameters
    ----------
    output_dir : Path
        Run output directory.
    bin_index : int
        Mass-bin index.

    Returns
    -------
    Path
        Destination PDF path.
    """
    filename = HOLDOUT_CORNER_PLOT_FILENAME_TEMPLATE.format(bin_index=bin_index)
    return output_dir / filename


def write_sim_holdout_corner_plots(
    halos_sim: pd.DataFrame,
    bin_edges: np.ndarray,
    output_dir: Path,
    input_features: Sequence[str],
    output_features: Sequence[str],
    min_bin_size: int,
    write_plots: bool = True,
    random_seed: int = 0,
) -> List[Tuple[int, Optional[np.ndarray], Optional[np.ndarray]]]:
    """
    Run SIM hold-out validation and optionally save per-bin corner plots.

    Parameters
    ----------
    halos_sim : pd.DataFrame
        HR training catalog with INPUT and OUTPUT feature columns.
    bin_edges : np.ndarray
        log10 mass bin edges.
    output_dir : Path
        Directory for validation PDFs.
    input_features : Sequence[str]
        Haloscope conditioning columns.
    output_features : Sequence[str]
        Target secondary-property columns.
    min_bin_size : int
        Minimum train and test halos per mass bin.
    write_plots : bool, optional
        When True, write one corner PDF per populated mass bin.
    random_seed : int, optional
        RNG seed for the train/test split.

    Returns
    -------
    list[tuple[int, Optional[np.ndarray], Optional[np.ndarray]]]
        Per-bin hold-out tuples from ``holdout_validate_sim_bins``.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    results = holdout_validate_sim_bins(
        halos_sim,
        bin_edges,
        input_features=input_features,
        output_features=output_features,
        min_bin_size=min_bin_size,
        random_seed=random_seed,
    )
    if not write_plots:
        release_validation_memory()
        return results

    output_feature_names = list(output_features)
    for bin_index, y_true, y_pred in results:
        if y_true is None or y_pred is None:
            continue
        y_true_frame = pd.DataFrame(y_true, columns=output_feature_names)
        corner_plot_sim_validation(
            y_true_frame,
            y_pred,
            format_holdout_corner_plot_title(input_features, bin_index),
            str(holdout_corner_plot_path(output_dir, bin_index)),
        )
        del y_true_frame
        release_validation_memory()

    release_validation_memory()
    return results
