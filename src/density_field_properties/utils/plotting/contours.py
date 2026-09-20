"""2D contour comparison helpers."""

from typing import Sequence

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from density_field_properties.utils.stats import bin_midpoints, confidence_intervals


def compare_2d_contours(
    axis: plt.Axes,
    x_true: np.ndarray,
    y_true: np.ndarray,
    x_pred: np.ndarray,
    y_pred: np.ndarray,
    xrange: Sequence[float],
    yrange: Sequence[float],
    bins_2d: int = 15,
    alpha: float = 0.26,
):
    """
    Overlay confidence contours for truth versus prediction on one axes.

    Parameters
    ----------
    axis : matplotlib.axes.Axes
        Target axes.
    x_true, y_true, x_pred, y_pred : np.ndarray
        Samples for contour histograms.
    xrange, yrange : Sequence[float]
        Histogram ranges ``(min, max)``.
    bins_2d : int, optional
        Number of bins per axis.
    alpha : float, optional
        Fill alpha for true distribution.

    Returns
    -------
    tuple
        Legend handles for true and predicted contours.
    """
    orange = matplotlib.colormaps["Oranges"]
    binary = matplotlib.colormaps["binary"]
    color_true = [orange(0.3), orange(0.5), orange(0.7), orange(0.8)]
    color_true_fill = [orange(0.3), orange(0.5), orange(0.7), orange(0.89)]
    color_pred = [binary(0.6)] * 4

    edges = np.histogram2d(x_true, y_true, bins=bins_2d, range=[xrange, yrange])
    x_centers = bin_midpoints(edges[1])
    y_centers = bin_midpoints(edges[2])
    pdf_true = edges[0].T / edges[0].sum()
    levels = confidence_intervals(pdf_true)
    axis.contourf(
        x_centers, y_centers, pdf_true, levels=levels, colors=color_true_fill, alpha=alpha
    )
    handle_true = axis.contour(
        x_centers, y_centers, pdf_true, levels=levels, colors=color_true, linewidths=3
    ).legend_elements()[0]

    pdf_pred, _, _ = np.histogram2d(x_pred, y_pred, bins=bins_2d, range=[xrange, yrange])
    pdf_pred = pdf_pred.T / pdf_pred.sum()
    levels = confidence_intervals(pdf_pred)
    handle_pred = axis.contour(
        x_centers,
        y_centers,
        pdf_pred,
        levels=levels,
        colors=color_pred,
        linewidths=4,
        linestyles="dashed",
    ).legend_elements()[0]
    axis.tick_params(axis="both", direction="in")
    axis.yaxis.set_ticks_position("both")
    axis.xaxis.set_ticks_position("both")
    return handle_true, handle_pred
