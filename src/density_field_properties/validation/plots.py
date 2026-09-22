"""Scientific validation figures for Haloscope runs."""

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

from density_field_properties.utils.stats import interpolated_ratio


def plot_assembly_bias_env_panel(
    reference_lower_mass: np.ndarray,
    reference_lower_bias: np.ndarray,
    reference_lower_sem: np.ndarray,
    reference_upper_mass: np.ndarray,
    reference_upper_bias: np.ndarray,
    reference_upper_sem: np.ndarray,
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
    reference_lower_mass, reference_lower_bias, reference_lower_sem : np.ndarray
        HR lower-tail ``b_1(M)`` (red in the paper).
    reference_upper_mass, reference_upper_bias, reference_upper_sem : np.ndarray
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
        reference_upper_bias - reference_upper_sem,
        reference_upper_bias + reference_upper_sem,
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
        reference_lower_bias - reference_lower_sem,
        reference_lower_bias + reference_lower_sem,
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

    ratio_upper = interpolated_ratio(
        haloscope_upper_mass,
        haloscope_upper_bias,
        reference_upper_mass,
        reference_upper_bias,
    )
    ratio_lower = interpolated_ratio(
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
