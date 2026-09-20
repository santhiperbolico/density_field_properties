"""Marginal and joint distribution plots for hold-out validation."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from density_field_properties.pipelines.run_defaults import OUTPUT_FEATURES
from density_field_properties.utils.plotting import compare_2d_contours


def corner_plot_sim_validation(
    y_true: pd.DataFrame,
    y_pred: np.ndarray,
    title: str,
    output_path: str,
) -> None:
    """
    Save a 3x3 corner figure comparing true and predicted halo properties.

    Parameters
    ----------
    y_true : pd.DataFrame
        True ``OUTPUT_FEATURES`` columns.
    y_pred : np.ndarray
        Predicted array with shape ``(n, len(OUTPUT_FEATURES))``.
    title : str
        Figure suptitle.
    output_path : str
        Path for ``plt.savefig``.
    """
    ranges = {"ca": [0.25, 1.05], "conc": [0.1, 30], "ba": [0.45, 1.05], "spin": [0, 0.1]}
    column_index = {name: index for index, name in enumerate(OUTPUT_FEATURES)}

    def predict_column(name):
        return y_pred[:, column_index[name]]

    fig, axes = plt.subplots(3, 3, figsize=(15, 15), gridspec_kw={"wspace": 0, "hspace": 0})
    for axis in (axes[0, 1], axes[0, 2]):
        axis.axis("off")
    axes[1, 2].set_visible(False)

    handle_true, handle_pred = compare_2d_contours(
        axes[0, 0],
        y_true["Spin"],
        y_true["cv"],
        predict_column("Spin"),
        predict_column("cv"),
        ranges["spin"],
        ranges["conc"],
    )
    axes[0, 0].set_ylabel("Halo Concentration", fontsize=20)
    compare_2d_contours(
        axes[1, 0],
        y_true["Spin"],
        y_true["ca"],
        predict_column("Spin"),
        predict_column("ca"),
        ranges["spin"],
        ranges["ca"],
    )
    axes[1, 0].set_ylabel("Halo Shape c/a", fontsize=20)
    compare_2d_contours(
        axes[2, 0],
        y_true["Spin"],
        y_true["ba"],
        predict_column("Spin"),
        predict_column("ba"),
        ranges["spin"],
        ranges["ba"],
    )
    axes[2, 0].set_ylabel("Halo Shape b/a", fontsize=20)
    axes[2, 0].set_xlabel("Halo Spin", fontsize=20)
    compare_2d_contours(
        axes[1, 1],
        y_true["cv"],
        y_true["ca"],
        predict_column("cv"),
        predict_column("ca"),
        ranges["conc"],
        ranges["ca"],
    )
    compare_2d_contours(
        axes[2, 1],
        y_true["cv"],
        y_true["ba"],
        predict_column("cv"),
        predict_column("ba"),
        ranges["conc"],
        ranges["ba"],
    )
    axes[2, 1].set_xlabel("Halo Concentration", fontsize=20)
    shape_axis = axes[2, 2]
    xline = np.linspace(0, 1.1, 4)
    shape_axis.fill_between(xline, xline, where=(xline > 0), color="grey", alpha=0.1)
    shape_axis.set_xlim(0.3, 1.05)
    shape_axis.set_ylim(0.38, 1.05)
    compare_2d_contours(
        shape_axis,
        y_true["ca"],
        y_true["ba"],
        predict_column("ca"),
        predict_column("ba"),
        ranges["ca"],
        ranges["ba"],
    )
    axes[2, 2].set_xlabel("Halo Shape c/a", fontsize=20)
    axes[0, 0].legend(
        [handle_true[-2], handle_pred[-2]],
        ["SIM (truth)", "Prediction"],
        fontsize=25,
        bbox_to_anchor=(3.05, 1),
    )
    fig.suptitle(title, fontsize=22)
    plt.savefig(output_path)
    plt.close(fig)
