"""Diagnostic plots for Haloscope SIM-to-FastPM transfer."""

from typing import Optional, Sequence, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import interpolate, stats

from density_field_properties.haloscope.sim_to_fastpm.config import OUTPUT_FEATURES


def plot_assembly_bias_env_panel(
    reference_lower_mass: np.ndarray,
    reference_lower_bias: np.ndarray,
    reference_lower_scatter: np.ndarray,
    reference_upper_mass: np.ndarray,
    reference_upper_bias: np.ndarray,
    reference_upper_scatter: np.ndarray,
    haloscope_lower_mass: np.ndarray,
    haloscope_lower_bias: np.ndarray,
    haloscope_upper_mass: np.ndarray,
    haloscope_upper_bias: np.ndarray,
    title: str = "input: env",
    output_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot HALOSCOPE-style assembly bias (Fig. 4, env input) with HR reference and model curves.

    Parameters
    ----------
    reference_lower_mass, reference_lower_bias, reference_lower_scatter : np.ndarray
        HR lower-tail ``b_1(M)`` (red in the paper).
    reference_upper_mass, reference_upper_bias, reference_upper_scatter : np.ndarray
        HR upper-tail ``b_1(M)`` (blue).
    haloscope_lower_mass, haloscope_lower_bias : np.ndarray
        LR+HALOSCOPE lower tail split by predicted properties.
    haloscope_upper_mass, haloscope_upper_bias : np.ndarray
        LR+HALOSCOPE upper tail split by predicted properties.
    title : str, optional
        Panel title (input description).
    output_path : Optional[str], optional
        If set, save the figure to this path.

    Returns
    -------
    matplotlib.figure.Figure
        The assembled figure.
    """
    fig, axes = plt.subplots(
        2,
        1,
        figsize=(7, 8),
        sharex=True,
        gridspec_kw={"height_ratios": [3, 1], "hspace": 0.05},
    )
    bias_axis, ratio_axis = axes

    bias_axis.plot(
        reference_upper_mass,
        reference_upper_bias,
        color="tab:blue",
        marker="^",
        linestyle="-",
        linewidth=2,
        label="upper 25% HR (cv, Spin, c/a, b/a)",
    )
    bias_axis.fill_between(
        reference_upper_mass,
        reference_upper_bias - reference_upper_scatter,
        reference_upper_bias + reference_upper_scatter,
        color="tab:blue",
        alpha=0.2,
    )
    bias_axis.plot(
        reference_lower_mass,
        reference_lower_bias,
        color="tab:red",
        marker="^",
        linestyle="-",
        linewidth=2,
        label="lower 25% HR (cv, Spin, c/a, b/a)",
    )
    bias_axis.fill_between(
        reference_lower_mass,
        reference_lower_bias - reference_lower_scatter,
        reference_lower_bias + reference_lower_scatter,
        color="tab:red",
        alpha=0.2,
    )
    bias_axis.plot(
        haloscope_upper_mass,
        haloscope_upper_bias,
        color="black",
        linewidth=2.5,
        label="LR + HALOSCOPE (upper, predicted props)",
    )
    bias_axis.plot(
        haloscope_lower_mass,
        haloscope_lower_bias,
        color="black",
        linewidth=2.5,
        linestyle="--",
        label="LR + HALOSCOPE (lower, predicted props)",
    )
    bias_axis.set_ylabel(r"Halo bias $b_1$")
    bias_axis.set_xscale("log")
    bias_axis.legend(loc="best", fontsize=9)
    bias_axis.set_title(title)
    bias_axis.grid(alpha=0.3)

    ratio_upper = _interpolated_ratio(
        haloscope_upper_mass,
        haloscope_upper_bias,
        reference_upper_mass,
        reference_upper_bias,
    )
    ratio_lower = _interpolated_ratio(
        haloscope_lower_mass,
        haloscope_lower_bias,
        reference_lower_mass,
        reference_lower_bias,
    )
    ratio_axis.axhline(1.0, color="grey", linewidth=1)
    ratio_axis.plot(reference_upper_mass, ratio_upper, color="black", linewidth=2)
    ratio_axis.plot(reference_lower_mass, ratio_lower, color="black", linewidth=2, linestyle="--")
    ratio_axis.set_ylabel("ratio with HR")
    ratio_axis.set_xlabel(r"$M_{200b}$ [$h^{-1} M_\odot$]")
    ratio_axis.set_ylim(0.88, 1.12)

    fig.tight_layout()
    if output_path is not None:
        fig.savefig(output_path)
    return fig


def _interpolated_ratio(
    model_mass: np.ndarray,
    model_bias: np.ndarray,
    reference_mass: np.ndarray,
    reference_bias: np.ndarray,
) -> np.ndarray:
    """
    Ratio of model to reference bias on the reference mass grid.

    Parameters
    ----------
    model_mass : np.ndarray
        Mass support for the model curve.
    model_bias : np.ndarray
        Model bias on ``model_mass``.
    reference_mass : np.ndarray
        Mass grid for the output ratio.
    reference_bias : np.ndarray
        Reference bias on ``reference_mass``.

    Returns
    -------
    np.ndarray
        Interpolated ``model / reference`` on ``reference_mass``.
    """
    if len(model_mass) < 2:
        return np.full_like(reference_bias, np.nan, dtype=float)
    interpolator = interpolate.interp1d(
        model_mass,
        model_bias,
        bounds_error=False,
        fill_value=np.nan,
    )
    return _safe_ratio(interpolator(reference_mass), reference_bias)


def _safe_ratio(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    """
    Element-wise ratio with NaNs where the denominator is non-finite or zero.

    Parameters
    ----------
    numerator : np.ndarray
        Values in the numerator.
    denominator : np.ndarray
        Values in the denominator.

    Returns
    -------
    np.ndarray
        ``numerator / denominator`` with invalid entries set to NaN.
    """
    safe_denominator = np.where(np.abs(denominator) > 0.0, denominator, np.nan)
    return numerator / safe_denominator


def bin_midpoints(edges: np.ndarray) -> np.ndarray:
    """
    Midpoints of a monotonic bin-edge vector.

    Parameters
    ----------
    edges : np.ndarray
        Bin edges.

    Returns
    -------
    np.ndarray
        Bin centers.
    """
    return (edges[1:] + edges[:-1]) / 2


def confidence_intervals(pdf_2d: np.ndarray, dx: float = 1.0, dy: float = 1.0) -> np.ndarray:
    """
    Contour levels enclosing 95%, 68%, and 40% of a normalized 2D PDF.

    Parameters
    ----------
    pdf_2d : np.ndarray
        Normalized 2D probability grid.
    dx : float, optional
        Cell width in x.
    dy : float, optional
        Cell height in y.

    Returns
    -------
    np.ndarray
        Contour levels plus a trailing sentinel level.
    """
    sample_count = 20
    thresholds = np.linspace(0, pdf_2d.max(), sample_count)
    integral = ((pdf_2d >= thresholds[:, None, None]) * pdf_2d).sum(axis=(1, 2)) * dx * dy
    interpolator = interpolate.interp1d(integral, thresholds)
    return np.append(interpolator(np.array([0.95, 0.68, 0.40])), 1)


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


def median_property_vs_mass(
    mass: np.ndarray,
    values: np.ndarray,
    mass_range: Tuple[float, float],
    n_bins: int = 11,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Binned median property versus mass.

    Parameters
    ----------
    mass : np.ndarray
        Halo masses (linear Msun/h).
    values : np.ndarray
        Property values.
    mass_range : tuple[float, float]
        ``(m_min, m_max)`` for binning in log10 mass.
    n_bins : int, optional
        Number of mass bins.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Mass bin centers and median property per bin.
    """
    log_mass = np.log10(mass)
    mass_centers = (
        10 ** stats.binned_statistic(log_mass, log_mass, "mean", bins=n_bins, range=mass_range)[0]
    )
    medians = (
        10
        ** stats.binned_statistic(
            log_mass, np.log10(values), "median", bins=n_bins, range=mass_range
        )[0]
    )
    return mass_centers, medians
