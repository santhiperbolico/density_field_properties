#!/usr/bin/env python
"""
Plot binned halo mass functions for four halo catalogs (UNIT + FastPM Rockstar x2 + FoF).

Rockstar catalogs keep host halos only (PID == -1). FoF uses every group in the
BigFile catalog. Axes match ``plot_hmf.py``: log10 mass vs log10(n/V).

Examples (repository root):

    # Default local_sample layout under output/local_sample/
    PYTHONPATH=src python scripts/plot_hmf_local_sample_four_catalogs.py

    # Full MN5 box on Taurus (/data21, 1000 Mpc/h)
    PYTHONPATH=src python scripts/plot_hmf_local_sample_four_catalogs.py \\
        --preset taurus_full \\
        --output-dir output/hmf_four_catalogs_taurus

    # Explicit paths (overrides preset defaults)
    PYTHONPATH=src python scripts/plot_hmf_local_sample_four_catalogs.py \\
        --box-size-mpc-h 1000 \\
        --unit-hlist /data21/UNITSIM/.../hlist_1.00000.list.bz2 \\
        --fastpm-nbody-list /data21/.../rockstar_out_nbody/out_8.list \\
        --fastpm-pm-list /data21/.../rockstar_out_pm/out_8.list \\
        --fastpm-fof /data21/.../fof_1.0000/LL-0.200

    # JSON job config (CLI flags override JSON values)
    PYTHONPATH=src python scripts/plot_hmf_local_sample_four_catalogs.py \\
        --config config/hmf_four_catalogs_taurus.json
"""

import argparse
import bz2
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MultipleLocator

from density_field_properties.pipelines.catalog_columns import (
    HLIST_CATALOG_LAYOUT,
    ROCKSTAR_LIST_CATALOG_LAYOUT,
    rockstar_catalog_column_kwargs,
)
from density_field_properties.pipelines.run_defaults import (
    FASTPM_LIST_NAME,
    FASTPM_ROCKSTAR_DIR,
    SIM_BOXSIZE_MPC_H,
    SIM_HLIST_RELATIVE,
    SIM_ROCKSTAR_ROOT,
)
from density_field_properties.preprocessing.catalog_loaders import (
    rockstar_halo_catalog_to_dataframe,
)
from density_field_properties.preprocessing.filters import filter_positive_mass
from density_field_properties.read_data.halos.column_maps import ROCKSTAR_LIST_COLUMNS
from density_field_properties.read_data.halos.fastpm import FastPMCatalogReader
from density_field_properties.read_data.halos.rockstar import rockstar_pid_column_index

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from plot_hmf import (  # noqa: E402
    BIN_EDGES_COUNT,
    halo_mass_function_from_masses_not_cumulative,
)

BIN_EDGES_LOG10_MAX = 15.5
BIN_EDGES_LOG10_MIN = 10.5
PLOT_MIN_LOG_MASS = 10.5

PRESET_LOCAL_SAMPLE = "local_sample"
PRESET_TAURUS_FULL = "taurus_full"
PRESET_CHOICES = (PRESET_LOCAL_SAMPLE, PRESET_TAURUS_FULL)

DEFAULT_LOCAL_SAMPLE = REPO_ROOT / "output/local_sample"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "output/local_sample_hmf_four_catalogs"
TAURUS_FASTPM_TFM_ROOT = Path("/data21/users/mruiz/fastpm_MN5/fastpm_tfm")

FIND_PARENTS_PID_COLUMN = ROCKSTAR_LIST_COLUMNS["halo_pid"]


@dataclass(frozen=True)
class CatalogSpec:
    """
    One halo catalog entry for the four-way HMF plot.

    Parameters
    ----------
    key : str
        Stable key for CSV columns (``unit``, ``rockstar_nbody``, etc.).
    label : str
        Legend label.
    path : Path
        Catalog path (Rockstar ``.list`` or FoF BigFile group directory).
    kind : str
        ``hlist``, ``rockstar_list``, or ``fastpm_fof``.
    """

    key: str
    label: str
    path: Path
    kind: str


@dataclass(frozen=True)
class RunConfig:
    """
    Resolved paths and volume for one HMF comparison run.

    Parameters
    ----------
    preset : str
        Preset name used to fill defaults.
    box_size_mpc_h : float
        Periodic box side length for n/V.
    output_dir : Path
        Directory for PNG and CSV export.
    output_stem : str
        Base filename without extension.
    plot_title : str
        Figure title.
    plot_min_log_mass : float
        Lower x-axis limit in log10(M).
    fof_group : str
        FoF BigFile group folder name.
    specs : tuple[CatalogSpec, ...]
        Catalogs to load and plot.
    """

    preset: str
    box_size_mpc_h: float
    output_dir: Path
    output_stem: str
    plot_title: str
    plot_min_log_mass: float
    fof_group: str
    specs: tuple[CatalogSpec, ...]


def _build_argument_parser() -> argparse.ArgumentParser:
    """
    Build the CLI argument parser for the four-catalog HMF plot.

    Returns
    -------
    argparse.ArgumentParser
        Configured parser instance.
    """
    parser = argparse.ArgumentParser(
        description="Compare HMFs for UNIT, FastPM Rockstar (nbody/pm), and FoF."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Optional JSON file with paths and box size (CLI overrides JSON).",
    )
    parser.add_argument(
        "--preset",
        choices=PRESET_CHOICES,
        default=PRESET_LOCAL_SAMPLE,
        help="Default catalog paths and box size when paths are not set explicitly.",
    )
    parser.add_argument(
        "--local-sample-dir",
        type=Path,
        default=DEFAULT_LOCAL_SAMPLE,
        help=f"Root for preset=local_sample (default: {DEFAULT_LOCAL_SAMPLE}).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for PNG and CSV export.",
    )
    parser.add_argument(
        "--output-stem",
        type=str,
        default=None,
        help="Output filename stem without extension (default: hmf_four_catalogs_<preset>).",
    )
    parser.add_argument(
        "--plot-title",
        type=str,
        default=None,
        help="Figure title (default depends on preset).",
    )
    parser.add_argument(
        "--box-size-mpc-h",
        type=float,
        default=None,
        help="Box side in Mpc/h for n/V; preset/manifest defaults apply when omitted.",
    )
    parser.add_argument(
        "--fof-group",
        type=str,
        default="LL-0.200",
        help="FoF BigFile group name (last path segment for fastpm-fof).",
    )
    parser.add_argument(
        "--plot-min-log-mass",
        type=float,
        default=PLOT_MIN_LOG_MASS,
        help=f"Minimum log10(M) shown on the x-axis (default: {PLOT_MIN_LOG_MASS}).",
    )
    parser.add_argument("--unit-hlist", type=Path, default=None, help="UNIT hlist path.")
    parser.add_argument(
        "--fastpm-nbody-list",
        type=Path,
        default=None,
        help="FastPM Rockstar nbody out_*.list path.",
    )
    parser.add_argument(
        "--fastpm-pm-list",
        type=Path,
        default=None,
        help="FastPM Rockstar pm out_*.list path.",
    )
    parser.add_argument(
        "--fastpm-fof",
        type=Path,
        default=None,
        help="FastPM FoF BigFile group directory (.../fof_1.0000/LL-0.200).",
    )
    parser.add_argument(
        "--unit-label",
        type=str,
        default=None,
        help="Legend label for the UNIT curve.",
    )
    parser.add_argument(
        "--fastpm-nbody-label",
        type=str,
        default=None,
        help="Legend label for the FastPM nbody Rockstar curve.",
    )
    parser.add_argument(
        "--fastpm-pm-label",
        type=str,
        default=None,
        help="Legend label for the FastPM pm Rockstar curve.",
    )
    parser.add_argument(
        "--fastpm-fof-label",
        type=str,
        default=None,
        help="Legend label for the FastPM FoF curve.",
    )
    return parser


def _parse_args(
    argv: list[str],
    parser_defaults: Optional[dict[str, Any]] = None,
) -> argparse.Namespace:
    """
    Parse CLI arguments for the four-catalog HMF plot.

    Parameters
    ----------
    argv : list[str]
        Command-line arguments without the program name.
    parser_defaults : Optional[dict[str, Any]], optional
        Extra defaults (for example from ``--config`` JSON).

    Returns
    -------
    argparse.Namespace
        Parsed options.
    """
    parser = _build_argument_parser()
    if parser_defaults:
        parser.set_defaults(**parser_defaults)
    return parser.parse_args(argv)


def _load_json_config(path: Path) -> dict[str, Any]:
    """
    Load a JSON run configuration file.

    Parameters
    ----------
    path : Path
        JSON file path.

    Returns
    -------
    dict[str, Any]
        Parsed configuration object.
    """
    return json.loads(path.read_text(encoding="utf-8"))


def _json_to_parser_defaults(file_config: dict[str, Any]) -> dict[str, Any]:
    """
    Map a JSON run file onto ``ArgumentParser.set_defaults`` keywords.

    Parameters
    ----------
    file_config : dict[str, Any]
        Parsed JSON configuration.

    Returns
    -------
    dict[str, Any]
        Keyword arguments for ``set_defaults``.
    """
    defaults: dict[str, Any] = {}
    path_keys = {
        "local_sample_dir",
        "output_dir",
        "unit_hlist",
        "fastpm_nbody_list",
        "fastpm_pm_list",
        "fastpm_fof",
    }
    for key, value in file_config.items():
        if key == "catalogs" and isinstance(value, dict):
            for catalog_key, catalog_path in value.items():
                if catalog_key in path_keys and catalog_path is not None:
                    defaults[catalog_key] = Path(catalog_path)
            continue
        if key in path_keys and value is not None:
            defaults[key] = Path(value)
        else:
            defaults[key] = value
    return defaults


def _parser_defaults_from_config_arg(argv: list[str]) -> dict[str, Any]:
    """
    Load JSON defaults when ``--config`` appears on the command line.

    Parameters
    ----------
    argv : list[str]
        Command-line arguments without the program name.

    Returns
    -------
    dict[str, Any]
        Defaults for ``ArgumentParser.set_defaults``, or empty when no config.
    """
    if "--config" not in argv:
        return {}
    config_index = argv.index("--config")
    if config_index + 1 >= len(argv):
        return {}
    config_path = Path(argv[config_index + 1])
    return _json_to_parser_defaults(_load_json_config(config_path))


def _preset_catalog_paths(
    preset: str,
    local_sample_dir: Path,
    fof_group: str,
) -> dict[str, Path]:
    """
    Return default catalog paths for a named preset.

    Parameters
    ----------
    preset : str
        ``local_sample`` or ``taurus_full``.
    local_sample_dir : Path
        Root of the local spatial sample tree.
    fof_group : str
        FoF group directory name.

    Returns
    -------
    dict[str, Path]
        Keys ``unit_hlist``, ``fastpm_nbody_list``, ``fastpm_pm_list``, ``fastpm_fof``.

    Raises
    ------
    ValueError
        If ``preset`` is not supported.
    """
    if preset == PRESET_LOCAL_SAMPLE:
        fastpm = local_sample_dir / "fastpm"
        return {
            "unit_hlist": local_sample_dir / "unitsim/hlist_1.00000.list",
            "fastpm_nbody_list": fastpm / "rockstar_out_nbody/out_8.list",
            "fastpm_pm_list": fastpm / "rockstar_out_pm/out_8.list",
            "fastpm_fof": fastpm / f"fof_1.0000/{fof_group}",
        }
    if preset == PRESET_TAURUS_FULL:
        return {
            "unit_hlist": SIM_ROCKSTAR_ROOT / SIM_HLIST_RELATIVE,
            "fastpm_nbody_list": FASTPM_ROCKSTAR_DIR / FASTPM_LIST_NAME,
            "fastpm_pm_list": TAURUS_FASTPM_TFM_ROOT / "rockstar_out_pm/out_8.list",
            "fastpm_fof": TAURUS_FASTPM_TFM_ROOT / f"output_01/fof_1.0000/{fof_group}",
        }
    raise ValueError(f"Unsupported preset: {preset}")


def _default_box_size_mpc_h(preset: str, local_sample_dir: Path) -> float:
    """
    Return the default box size for a preset when not set on the CLI.

    Parameters
    ----------
    preset : str
        ``local_sample`` or ``taurus_full``.
    local_sample_dir : Path
        Used to read ``sample_manifest.json`` for the local preset.

    Returns
    -------
    float
        Box side length in Mpc/h.
    """
    if preset == PRESET_TAURUS_FULL:
        return float(SIM_BOXSIZE_MPC_H)
    manifest_path = local_sample_dir / "sample_manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        return float(manifest["box_size_mpc_h"])
    return 200.0


def _resolve_run_config(args: argparse.Namespace) -> RunConfig:
    """
    Build the fully resolved run configuration from CLI and JSON.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed arguments after optional JSON merge.

    Returns
    -------
    RunConfig
        Paths, volume, and plot metadata for this run.
    """
    preset_paths = _preset_catalog_paths(args.preset, args.local_sample_dir, args.fof_group)
    unit_path = args.unit_hlist or preset_paths["unit_hlist"]
    nbody_path = args.fastpm_nbody_list or preset_paths["fastpm_nbody_list"]
    pm_path = args.fastpm_pm_list or preset_paths["fastpm_pm_list"]
    fof_path = args.fastpm_fof or preset_paths["fastpm_fof"]

    box_size = args.box_size_mpc_h
    if box_size is None:
        box_size = _default_box_size_mpc_h(args.preset, args.local_sample_dir)

    output_dir = args.output_dir
    if output_dir is None:
        if args.preset == PRESET_LOCAL_SAMPLE:
            output_dir = DEFAULT_OUTPUT_DIR
        else:
            output_dir = REPO_ROOT / f"output/hmf_four_catalogs_{args.preset}"

    output_stem = args.output_stem or f"hmf_four_catalogs_{args.preset}"
    if args.plot_title is not None:
        plot_title = args.plot_title
    elif args.preset == PRESET_LOCAL_SAMPLE:
        plot_title = "local_sample HMF (four catalogs)"
    else:
        plot_title = f"Full-box HMF ({args.preset})"

    specs = (
        CatalogSpec(
            key="unit",
            label=args.unit_label or "UNIT (hlist, hosts)",
            path=Path(unit_path),
            kind="hlist",
        ),
        CatalogSpec(
            key="rockstar_nbody",
            label=args.fastpm_nbody_label or "FastPM Rockstar nbody (hosts)",
            path=Path(nbody_path),
            kind="rockstar_list",
        ),
        CatalogSpec(
            key="rockstar_pm",
            label=args.fastpm_pm_label or "FastPM Rockstar pm (hosts)",
            path=Path(pm_path),
            kind="rockstar_list",
        ),
        CatalogSpec(
            key="fastpm_fof",
            label=args.fastpm_fof_label or "FastPM FoF (all groups)",
            path=Path(fof_path),
            kind="fastpm_fof",
        ),
    )
    return RunConfig(
        preset=args.preset,
        box_size_mpc_h=float(box_size),
        output_dir=Path(output_dir),
        output_stem=output_stem,
        plot_title=plot_title,
        plot_min_log_mass=float(args.plot_min_log_mass),
        fof_group=args.fof_group,
        specs=specs,
    )


def _open_catalog_text(path: Path):
    """
    Open a Rockstar catalog as text, transparently handling ``.bz2``.

    Parameters
    ----------
    path : Path
        Catalog file path.

    Returns
    -------
    context manager
        File-like object for line iteration.
    """
    if path.suffix == ".bz2":
        return bz2.open(path, "rt")
    return path.open(encoding="utf-8")


def _rockstar_list_pid_column(path: Path, default_pid_column: int) -> int:
    """
    Return the PID column index for Rockstar ``.list`` host filtering.

    Parameters
    ----------
    path : Path
        Rockstar catalog path.
    default_pid_column : int
        Layout default when the header omits PID.

    Returns
    -------
    int
        Zero-based PID column index for data rows.
    """
    first_row_width = None
    with _open_catalog_text(path) as handle:
        for line in handle:
            if line.startswith("#") or not line.strip():
                continue
            first_row_width = len(line.split())
            break
    header_pid = rockstar_pid_column_index(str(path))
    if first_row_width is not None and header_pid is not None and first_row_width <= header_pid:
        return FIND_PARENTS_PID_COLUMN
    if header_pid is not None:
        return header_pid
    return default_pid_column


def _load_rockstar_list_host_masses(path: Path) -> np.ndarray:
    """
    Load positive M200b masses for host halos from a Rockstar ``.list`` file.

    Parameters
    ----------
    path : Path
        Path to ``out_*.list`` (including ``find_parents`` short format).

    Returns
    -------
    np.ndarray
        Host halo masses in Msun/h.
    """
    column_kwargs = rockstar_catalog_column_kwargs(ROCKSTAR_LIST_CATALOG_LAYOUT)
    pid_column = _rockstar_list_pid_column(path, column_kwargs["halo_pid_position"])
    m200b_column = column_kwargs["halo_m200b_position"]
    masses: list[float] = []
    with _open_catalog_text(path) as handle:
        for line in handle:
            if line.startswith("#") or not line.strip():
                continue
            cols = line.split()
            if len(cols) <= max(pid_column, m200b_column):
                continue
            if int(cols[pid_column]) != -1:
                continue
            mass = float(cols[m200b_column])
            if mass > 0.0:
                masses.append(mass)
    return np.asarray(masses, dtype=np.float64)


def _load_hlist_host_masses(path: Path) -> np.ndarray:
    """
    Load UNIT consistent-trees host masses via the shared catalog loader.

    Parameters
    ----------
    path : Path
        Path to ``hlist_*.list`` or ``.list.bz2``.

    Returns
    -------
    np.ndarray
        Host halo M200b values in Msun/h.
    """
    column_kwargs = rockstar_catalog_column_kwargs(HLIST_CATALOG_LAYOUT)
    frame = rockstar_halo_catalog_to_dataframe(
        path,
        catalog_column_kwargs=column_kwargs,
        filter_by_pid=True,
    )
    frame = filter_positive_mass(frame)
    return frame["M200b"].to_numpy(dtype=np.float64)


def _load_fastpm_fof_masses(fof_group_path: Path) -> np.ndarray:
    """
    Load FoF group masses from a FastPM BigFile group directory.

    Parameters
    ----------
    fof_group_path : Path
        Path ending in the FoF group name (for example ``.../LL-0.200``).

    Returns
    -------
    np.ndarray
        Group masses Length * m_part in Msun/h.
    """
    catalog = FastPMCatalogReader.read_catalog(str(fof_group_path))
    masses = catalog.halo_m200b.astype(np.float64)
    return masses[np.isfinite(masses) & (masses > 0.0)]


def _load_catalog_masses(spec: CatalogSpec) -> np.ndarray:
    """
    Load halo masses for one catalog specification.

    Parameters
    ----------
    spec : CatalogSpec
        Catalog metadata and path.

    Returns
    -------
    np.ndarray
        Positive finite masses in Msun/h.

    Raises
    ------
    FileNotFoundError
        If ``spec.path`` is missing.
    ValueError
        If ``spec.kind`` is not supported.
    """
    if not spec.path.exists():
        raise FileNotFoundError(f"Catalog not found for {spec.label}: {spec.path}")
    if spec.kind == "hlist":
        masses = _load_hlist_host_masses(spec.path)
    elif spec.kind == "rockstar_list":
        masses = _load_rockstar_list_host_masses(spec.path)
    elif spec.kind == "fastpm_fof":
        masses = _load_fastpm_fof_masses(spec.path)
    else:
        raise ValueError(f"Unsupported catalog kind: {spec.kind}")
    logging.info(
        "Loaded %d halos from %s (min=%.3e max=%.3e Msun/h)",
        masses.size,
        spec.path,
        masses.min() if masses.size else 0.0,
        masses.max() if masses.size else 0.0,
    )
    return masses


def _binned_log_hmf(
    masses: np.ndarray,
    bin_edges: np.ndarray,
    volume_mpc_h3: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute log10 bin centers and log10 number density for one mass sample.

    Parameters
    ----------
    masses : np.ndarray
        Halo masses in Msun/h.
    bin_edges : np.ndarray
        Linear mass bin edges in Msun/h.
    volume_mpc_h3 : float
        Sample volume in (Mpc/h)^3.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        ``log10`` bin centers and ``log10(n/V)`` for bins with positive counts.
    """
    bin_centers, counts = halo_mass_function_from_masses_not_cumulative(
        masses,
        bin_edges=bin_edges,
    )
    density = counts.astype(np.float64) / volume_mpc_h3
    positive = density > 0.0
    log_mass = np.log10(bin_centers[positive])
    log_density = np.log10(density[positive])
    return log_mass, log_density


def _plot_four_catalog_hmf(
    curves: dict[str, tuple[np.ndarray, np.ndarray]],
    output_path: Path,
    run_config: RunConfig,
) -> None:
    """
    Save a four-curve HMF comparison figure.

    Parameters
    ----------
    curves : dict[str, tuple[np.ndarray, np.ndarray]]
        Mapping from legend label to ``(log10 M, log10 n/V)`` arrays.
    output_path : Path
        Destination PNG path.
    run_config : RunConfig
        Volume and axis limits for the figure.
    """
    fig, ax = plt.subplots(figsize=(10, 10))
    for label, (log_mass, log_density) in curves.items():
        mask = log_mass >= run_config.plot_min_log_mass
        ax.plot(
            log_mass[mask],
            log_density[mask],
            "-",
            linewidth=1.2,
            label=label,
        )
    box_size = run_config.box_size_mpc_h
    volume = box_size**3
    ax.set_title(
        rf"{run_config.plot_title} ($L_{{\rm box}} = {box_size:.0f}$ Mpc/h, "
        rf"$V = {volume:.3e}$ (Mpc/h)$^3$)"
    )
    ax.set_xlabel(r"$\log_{10}(M\,[\mathrm{M}_\odot/h])$")
    ax.set_ylabel(r"$\log_{10}(n(M)\,[\mathrm{Mpc}/h]^{-3})$")
    ax.xaxis.set_major_locator(MultipleLocator(0.2))
    ax.xaxis.set_minor_locator(MultipleLocator(0.1))
    ax.yaxis.set_major_locator(MultipleLocator(0.5))
    ax.yaxis.set_minor_locator(MultipleLocator(0.1))
    ax.set_xlim(left=run_config.plot_min_log_mass)
    ax.grid(which="major", linestyle="-", alpha=0.35)
    ax.grid(which="minor", linestyle=":", alpha=0.25)
    ax.legend()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main(argv: list[str]) -> int:
    """
    Load configured catalogs and plot their binned HMFs.

    Parameters
    ----------
    argv : list[str]
        Command-line arguments without the program name.

    Returns
    -------
    int
        Process exit code.
    """
    json_defaults = _parser_defaults_from_config_arg(argv)
    run_config = _resolve_run_config(_parse_args(argv, json_defaults))

    logging.info("Preset: %s", run_config.preset)
    logging.info("Box size (Mpc/h): %.1f", run_config.box_size_mpc_h)
    for spec in run_config.specs:
        logging.info("Catalog [%s]: %s", spec.key, spec.path)

    volume_mpc_h3 = run_config.box_size_mpc_h**3
    bin_edges = np.logspace(BIN_EDGES_LOG10_MIN, BIN_EDGES_LOG10_MAX, BIN_EDGES_COUNT)
    curves: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    csv_columns: dict[str, np.ndarray] = {}
    log_bin_centers = np.log10(0.5 * (bin_edges[1:] + bin_edges[:-1]))

    for spec in run_config.specs:
        masses = _load_catalog_masses(spec)
        log_mass, log_density = _binned_log_hmf(masses, bin_edges, volume_mpc_h3)
        curves[spec.label] = (log_mass, log_density)
        _, counts = halo_mass_function_from_masses_not_cumulative(
            masses,
            bin_edges=bin_edges,
        )
        density = counts.astype(np.float64) / volume_mpc_h3
        with np.errstate(divide="ignore"):
            csv_columns[spec.key] = np.log10(density)

    run_config.output_dir.mkdir(parents=True, exist_ok=True)
    figure_path = run_config.output_dir / f"{run_config.output_stem}.png"
    _plot_four_catalog_hmf(curves, figure_path, run_config)
    logging.info("Saved plot to %s", figure_path)

    csv_path = run_config.output_dir / f"{run_config.output_stem}.csv"
    ordered_keys = [spec.key for spec in run_config.specs]
    csv_header = "log10_bin_center," + ",".join(f"log10_n_over_V_{key}" for key in ordered_keys)
    csv_data = np.column_stack([log_bin_centers] + [csv_columns[key] for key in ordered_keys])
    np.savetxt(
        csv_path,
        csv_data,
        delimiter=",",
        header=csv_header,
        comments="",
    )
    logging.info("Saved binned curves to %s", csv_path)

    metadata_path = run_config.output_dir / f"{run_config.output_stem}_run.json"
    metadata_path.write_text(
        json.dumps(
            {
                "preset": run_config.preset,
                "box_size_mpc_h": run_config.box_size_mpc_h,
                "fof_group": run_config.fof_group,
                "catalogs": {spec.key: str(spec.path) for spec in run_config.specs},
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    logging.info("Wrote run metadata to %s", metadata_path)

    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    sys.exit(main(sys.argv[1:]))
