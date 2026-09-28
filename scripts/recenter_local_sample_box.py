#!/usr/bin/env python
"""
Re-center local_sample coordinates from [origin, origin + box_size] to [0, box_size].

By default the script only reports min/max checks (dry run) on the input tree. Pass
``--apply`` to copy the sample into a separate output directory and write recentered
files there; the input tree is never modified.

Example (review only):

    PYTHONPATH=src python scripts/recenter_local_sample_box.py

Example (write recentered copy):

    PYTHONPATH=src python scripts/recenter_local_sample_box.py --apply
"""

import argparse
import json
import logging
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Optional, Sequence

import numpy as np

try:
    from bigfile import BigFile
except ImportError:
    BigFile = None

from density_field_properties.read_data.halos.column_maps import (
    ROCKSTAR_LIST_COLUMNS,
    UNIT_HLIST_COLUMNS,
)
from density_field_properties.read_data.particles import (
    detect_dm_particle_format,
    iter_dm_particle_batches,
)
from density_field_properties.read_data.particles.particle_io import _fastpm_block_paths

DEFAULT_LOCAL_SAMPLE_DIR = Path("output/local_sample")
DEFAULT_OUTPUT_DIR = Path("output/local_sample_recentered")
DEFAULT_ORIGIN_MPC_H = 400.0
DEFAULT_BOX_SIZE_MPC_H = 200.0
DEFAULT_SCAN_ROWS = 5000
DEFAULT_PARTICLE_BATCH_SIZE = 100_000
HLIST_POSITION_INDICES = (
    UNIT_HLIST_COLUMNS["x"],
    UNIT_HLIST_COLUMNS["y"],
    UNIT_HLIST_COLUMNS["z"],
)
ROCKSTAR_POSITION_INDICES = (
    ROCKSTAR_LIST_COLUMNS["halo_x"],
    ROCKSTAR_LIST_COLUMNS["halo_y"],
    ROCKSTAR_LIST_COLUMNS["halo_z"],
)
TEXT_PARTICLE_POSITION_INDICES = (0, 1, 2)
HLIST_BOX_HEADER_PREFIX = "#Full box size = "
ROCKSTAR_BOX_HEADER_PREFIX = "#Box size: "


@dataclass(frozen=True)
class TargetSpec:
    """
    One recenter target under ``local_sample``.

    Parameters
    ----------
    relative_path : Path
        Path relative to the local sample root.
    kind : str
        Target kind: ``hlist``, ``rockstar_list``, ``text_particles``,
        or ``bigfile_positions``.
    position_indices : tuple[int, ...]
        0-based column indices to shift for text-like targets.
    """

    relative_path: Path
    kind: str
    position_indices: tuple[int, ...] = ()


DEFAULT_TARGETS: tuple[TargetSpec, ...] = (
    TargetSpec(
        relative_path=Path("unitsim/hlist_1.00000.list"),
        kind="hlist",
        position_indices=HLIST_POSITION_INDICES,
    ),
    TargetSpec(
        relative_path=Path("fastpm/rockstar_out_nbody/out_8.list"),
        kind="rockstar_list",
        position_indices=ROCKSTAR_POSITION_INDICES,
    ),
    TargetSpec(
        relative_path=Path("fastpm/rockstar_out_pm/out_8.list"),
        kind="rockstar_list",
        position_indices=ROCKSTAR_POSITION_INDICES,
    ),
    TargetSpec(
        relative_path=Path("unitsim/dm_particles_0.5_128.dat"),
        kind="text_particles",
        position_indices=TEXT_PARTICLE_POSITION_INDICES,
    ),
    TargetSpec(
        relative_path=Path("fastpm/snap_1.0000/1"),
        kind="bigfile_positions",
    ),
)


@dataclass
class AxisRange:
    """
    Min/max summary for one coordinate axis.

    Parameters
    ----------
    minimum : float
        Minimum value observed.
    maximum : float
        Maximum value observed.
    count : int
        Number of samples used for the summary.
    """

    minimum: float
    maximum: float
    count: int


@dataclass
class TargetReport:
    """
    Scan report for one recenter target.

    Parameters
    ----------
    spec : TargetSpec
        Target specification.
    exists : bool
        Whether the target path exists.
    before : dict[str, AxisRange]
        Coordinate ranges before recentering.
    after : dict[str, AxisRange]
        Expected coordinate ranges after subtracting the origin shift.
    notes : list[str]
        Human-readable validation notes.
    """

    spec: TargetSpec
    exists: bool
    before: dict[str, AxisRange]
    after: dict[str, AxisRange]
    notes: list[str]


def _parse_args(argv: Sequence[str]) -> argparse.Namespace:
    """
    Parse CLI arguments for the local_sample recenter utility.

    Parameters
    ----------
    argv : Sequence[str]
        Command-line arguments without the program name.

    Returns
    -------
    argparse.Namespace
        Parsed options.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Re-center local_sample halo and particle coordinates "
            "from [origin, origin + box_size] to [0, box_size]."
        )
    )
    parser.add_argument(
        "--local-sample-dir",
        type=Path,
        default=DEFAULT_LOCAL_SAMPLE_DIR,
        help=f"Input local sample tree (default: {DEFAULT_LOCAL_SAMPLE_DIR}).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output tree for recentered copy (default: {DEFAULT_OUTPUT_DIR}).",
    )
    parser.add_argument(
        "--origin-min-mpc-h",
        type=float,
        default=DEFAULT_ORIGIN_MPC_H,
        help=f"Current box minimum in Mpc/h (default: {DEFAULT_ORIGIN_MPC_H}).",
    )
    parser.add_argument(
        "--box-size-mpc-h",
        type=float,
        default=DEFAULT_BOX_SIZE_MPC_H,
        help=f"Local box side length in Mpc/h (default: {DEFAULT_BOX_SIZE_MPC_H}).",
    )
    parser.add_argument(
        "--scan-rows",
        type=int,
        default=DEFAULT_SCAN_ROWS,
        help=f"Rows sampled per text catalog target (default: {DEFAULT_SCAN_ROWS}).",
    )
    parser.add_argument(
        "--particle-batch-size",
        type=int,
        default=DEFAULT_PARTICLE_BATCH_SIZE,
        help=(
            "Particle batch size for BigFile scans and writes "
            f"(default: {DEFAULT_PARTICLE_BATCH_SIZE})."
        ),
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Copy the input tree to --output-dir and write recentered files there.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Remove an existing --output-dir before writing.",
    )
    parser.add_argument(
        "--skip-bigfile",
        action="store_true",
        help="Skip FastPM BigFile Position targets.",
    )
    return parser.parse_args(argv)


def _axis_range(values: Iterable[float]) -> AxisRange:
    """
    Build an axis range summary from numeric values.

    Parameters
    ----------
    values : Iterable[float]
        Coordinate samples.

    Returns
    -------
    AxisRange
        Min/max summary for the provided samples.
    """
    array = np.asarray(list(values), dtype=np.float64)
    if array.size == 0:
        return AxisRange(minimum=np.nan, maximum=np.nan, count=0)
    return AxisRange(minimum=float(array.min()), maximum=float(array.max()), count=int(array.size))


def _shifted_range(axis_range: AxisRange, shift_mpc_h: float) -> AxisRange:
    """
    Shift an axis range by subtracting a constant offset.

    Parameters
    ----------
    axis_range : AxisRange
        Input coordinate range.
    shift_mpc_h : float
        Value subtracted from each coordinate.

    Returns
    -------
    AxisRange
        Shifted min/max range with the same sample count.
    """
    if axis_range.count == 0:
        return axis_range
    return AxisRange(
        minimum=axis_range.minimum - shift_mpc_h,
        maximum=axis_range.maximum - shift_mpc_h,
        count=axis_range.count,
    )


def _scan_text_catalog(
    path: Path,
    position_indices: Sequence[int],
    max_rows: int,
) -> dict[str, AxisRange]:
    """
    Scan x/y/z ranges from a whitespace-separated catalog file.

    Parameters
    ----------
    path : Path
        Catalog path.
    position_indices : Sequence[int]
        0-based indices for x, y, and z.
    max_rows : int
        Maximum data rows to scan.

    Returns
    -------
    dict[str, AxisRange]
        Ranges keyed by ``x``, ``y``, and ``z``.
    """
    axis_names = ("x", "y", "z")
    buckets = {name: [] for name in axis_names}
    rows_read = 0
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) <= max(position_indices):
                continue
            for name, index in zip(axis_names, position_indices):
                buckets[name].append(float(parts[index]))
            rows_read += 1
            if rows_read >= max_rows:
                break
    return {name: _axis_range(values) for name, values in buckets.items()}


def _scan_text_particles(path: Path, max_rows: int) -> dict[str, AxisRange]:
    """
    Scan x/y/z ranges from a whitespace-separated particle text file.

    Parameters
    ----------
    path : Path
        Particle text file path.
    max_rows : int
        Maximum rows to scan.

    Returns
    -------
    dict[str, AxisRange]
        Ranges keyed by ``x``, ``y``, and ``z``.
    """
    buckets = {"x": [], "y": [], "z": []}
    rows_read = 0
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            parts = line.split()
            if len(parts) < 3:
                continue
            buckets["x"].append(float(parts[0]))
            buckets["y"].append(float(parts[1]))
            buckets["z"].append(float(parts[2]))
            rows_read += 1
            if rows_read >= max_rows:
                break
    return {name: _axis_range(values) for name, values in buckets.items()}


def _scan_bigfile_positions(path: Path, batch_size: int) -> dict[str, AxisRange]:
    """
    Scan x/y/z ranges from a FastPM BigFile Position block.

    Parameters
    ----------
    path : Path
        FastPM block directory (for example ``snap_1.0000/1``).
    batch_size : int
        Particle batch size used while scanning.

    Returns
    -------
    dict[str, AxisRange]
        Ranges keyed by ``x``, ``y``, and ``z``.
    """
    mins = np.full(3, np.inf, dtype=np.float64)
    maxs = np.full(3, -np.inf, dtype=np.float64)
    count = 0
    for positions, _, _ in iter_dm_particle_batches(str(path), batch_size=batch_size):
        if positions.size == 0:
            continue
        mins = np.minimum(mins, positions.min(axis=0))
        maxs = np.maximum(maxs, positions.max(axis=0))
        count += positions.shape[0]
    if count == 0:
        nan_range = AxisRange(minimum=np.nan, maximum=np.nan, count=0)
        return {"x": nan_range, "y": nan_range, "z": nan_range}
    return {
        "x": AxisRange(minimum=float(mins[0]), maximum=float(maxs[0]), count=count),
        "y": AxisRange(minimum=float(mins[1]), maximum=float(maxs[1]), count=count),
        "z": AxisRange(minimum=float(mins[2]), maximum=float(maxs[2]), count=count),
    }


def _validate_ranges(
    before: dict[str, AxisRange],
    after: dict[str, AxisRange],
    origin_min_mpc_h: float,
    box_size_mpc_h: float,
) -> list[str]:
    """
    Build validation notes for one target scan.

    Parameters
    ----------
    before : dict[str, AxisRange]
        Observed coordinate ranges.
    after : dict[str, AxisRange]
        Expected ranges after recentering.
    origin_min_mpc_h : float
        Current box minimum in Mpc/h.
    box_size_mpc_h : float
        Local box side length in Mpc/h.

    Returns
    -------
    list[str]
        Validation notes for the report.
    """
    notes: list[str] = []
    expected_max = origin_min_mpc_h + box_size_mpc_h
    already_recentered = all(
        axis.count > 0 and axis.minimum >= -1.0 and axis.maximum <= box_size_mpc_h + 1.0
        for axis in before.values()
    )
    if already_recentered:
        notes.append("Coordinates already look recentered to [0, box_size].")
        return notes

    for axis_name, axis_range in before.items():
        if axis_range.count == 0:
            notes.append(f"No samples found for axis {axis_name}.")
            continue
        if axis_range.minimum < origin_min_mpc_h - 1.0:
            notes.append(
                f"{axis_name} min {axis_range.minimum:.3f} is below origin {origin_min_mpc_h}."
            )
        if axis_range.maximum > expected_max + 1.0:
            notes.append(
                f"{axis_name} max {axis_range.maximum:.3f} exceeds "
                f"expected upper bound {expected_max}."
            )

    for axis_name, axis_range in after.items():
        if axis_range.count == 0:
            continue
        if axis_range.minimum < -1.0 or axis_range.maximum > box_size_mpc_h + 1.0:
            notes.append(
                f"After shift, {axis_name} would be [{axis_range.minimum:.3f}, "
                f"{axis_range.maximum:.3f}] outside [0, {box_size_mpc_h}]."
            )
    if not notes:
        notes.append("Ranges are consistent with origin shift.")
    return notes


def scan_target(
    spec: TargetSpec,
    path: Path,
    origin_min_mpc_h: float,
    box_size_mpc_h: float,
    scan_rows: int,
    particle_batch_size: int,
) -> TargetReport:
    """
    Scan one recenter target and build a before/after report.

    Parameters
    ----------
    spec : TargetSpec
        Target specification.
    path : Path
        Absolute target path.
    origin_min_mpc_h : float
        Current box minimum in Mpc/h.
    box_size_mpc_h : float
        Local box side length in Mpc/h.
    scan_rows : int
        Maximum rows to scan for text catalogs and particle text files.
    particle_batch_size : int
        Batch size for BigFile scans.

    Returns
    -------
    TargetReport
        Scan report for the target.
    """
    if not path.exists():
        empty = AxisRange(minimum=np.nan, maximum=np.nan, count=0)
        empty_axes = {"x": empty, "y": empty, "z": empty}
        return TargetReport(
            spec=spec,
            exists=False,
            before=empty_axes,
            after=empty_axes,
            notes=["Target path does not exist."],
        )

    if spec.kind in {"hlist", "rockstar_list"}:
        before = _scan_text_catalog(path, spec.position_indices, scan_rows)
    elif spec.kind == "text_particles":
        before = _scan_text_particles(path, scan_rows)
    elif spec.kind == "bigfile_positions":
        before = _scan_bigfile_positions(path, particle_batch_size)
    else:
        raise ValueError(f"Unsupported target kind: {spec.kind}")

    after = {
        axis_name: _shifted_range(axis_range, origin_min_mpc_h)
        for axis_name, axis_range in before.items()
    }
    notes = _validate_ranges(before, after, origin_min_mpc_h, box_size_mpc_h)
    return TargetReport(
        spec=spec,
        exists=True,
        before=before,
        after=after,
        notes=notes,
    )


def _format_axis_range(axis_range: AxisRange) -> str:
    """
    Format one axis range for logging.

    Parameters
    ----------
    axis_range : AxisRange
        Axis range to format.

    Returns
    -------
    str
        Human-readable range string.
    """
    if axis_range.count == 0:
        return "no data"
    return f"[{axis_range.minimum:.5f}, {axis_range.maximum:.5f}] " f"(n={axis_range.count})"


def log_target_report(report: TargetReport) -> None:
    """
    Log one target scan report.

    Parameters
    ----------
    report : TargetReport
        Scan report to print.
    """
    rel_path = report.spec.relative_path
    logging.info("Target: %s (%s)", rel_path, report.spec.kind)
    if not report.exists:
        logging.warning("  missing path")
        for note in report.notes:
            logging.warning("  - %s", note)
        return
    for axis_name in ("x", "y", "z"):
        logging.info("  %s before: %s", axis_name, _format_axis_range(report.before[axis_name]))
        logging.info("  %s after:  %s", axis_name, _format_axis_range(report.after[axis_name]))
    for note in report.notes:
        logging.info("  note: %s", note)


def _update_box_header_line(line: str, box_size_mpc_h: float, prefix: str) -> str:
    """
    Rewrite a box-size comment line for the recentered local box.

    Parameters
    ----------
    line : str
        Original comment line including the trailing newline.
    box_size_mpc_h : float
        Target local box size in Mpc/h.
    prefix : str
        Comment prefix identifying the box-size line.

    Returns
    -------
    str
        Updated comment line.
    """
    if not line.startswith(prefix):
        return line
    return f"{prefix}{box_size_mpc_h:.6f} Mpc/h\n"


def _shift_text_row(
    parts: list[str], position_indices: Sequence[int], shift_mpc_h: float
) -> list[str]:
    """
    Subtract a constant shift from selected columns in one tokenized row.

    Parameters
    ----------
    parts : list[str]
        Tokenized data row.
    position_indices : Sequence[int]
        0-based indices to shift.
    shift_mpc_h : float
        Value subtracted from each selected coordinate.

    Returns
    -------
    list[str]
        Updated tokenized row.
    """
    updated = list(parts)
    for index in position_indices:
        updated[index] = f"{float(updated[index]) - shift_mpc_h:.6f}"
    return updated


def _recenter_text_catalog(
    source_path: Path,
    destination_path: Path,
    spec: TargetSpec,
    origin_min_mpc_h: float,
    header_updater: Callable[[str], str],
) -> None:
    """
    Re-center one whitespace-separated catalog file into a destination path.

    Parameters
    ----------
    source_path : Path
        Input catalog path.
    destination_path : Path
        Output catalog path.
    spec : TargetSpec
        Target specification.
    origin_min_mpc_h : float
        Current box minimum in Mpc/h.
    header_updater : Callable[[str], str]
        Function that updates box-size comment lines.
    """
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = destination_path.with_suffix(destination_path.suffix + ".recenter_tmp")
    with (
        source_path.open(encoding="utf-8") as source,
        temp_path.open("w", encoding="utf-8") as target,
    ):
        for line in source:
            if line.startswith("#"):
                target.write(header_updater(line))
                continue
            parts = line.split()
            if len(parts) > max(spec.position_indices):
                parts = _shift_text_row(parts, spec.position_indices, origin_min_mpc_h)
            target.write(" ".join(parts) + "\n")
    temp_path.replace(destination_path)


def _recenter_text_particles(
    source_path: Path,
    destination_path: Path,
    origin_min_mpc_h: float,
) -> None:
    """
    Re-center one whitespace-separated particle text file into a destination path.

    Parameters
    ----------
    source_path : Path
        Input particle text file path.
    destination_path : Path
        Output particle text file path.
    origin_min_mpc_h : float
        Current box minimum in Mpc/h.
    """
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = destination_path.with_suffix(destination_path.suffix + ".recenter_tmp")
    with (
        source_path.open(encoding="utf-8") as source,
        temp_path.open("w", encoding="utf-8") as target,
    ):
        for line in source:
            parts = line.split()
            if len(parts) < 3:
                target.write(line)
                continue
            x_value = float(parts[0]) - origin_min_mpc_h
            y_value = float(parts[1]) - origin_min_mpc_h
            z_value = float(parts[2]) - origin_min_mpc_h
            remainder = parts[3:]
            row = [f"{x_value:.6f}", f"{y_value:.6f}", f"{z_value:.6f}"] + remainder
            target.write(" ".join(row) + "\n")
    temp_path.replace(destination_path)


def _require_bigfile():
    """
    Return the BigFile class or raise when the optional dependency is missing.

    Returns
    -------
    type
        BigFile class from the ``bigfile`` package.

    Raises
    ------
    ImportError
        If ``bigfile`` is not installed.
    """
    if BigFile is None:
        raise ImportError("bigfile is required for FastPM BigFile particle I/O")
    return BigFile


def _recenter_bigfile_positions(path: Path, origin_min_mpc_h: float, batch_size: int) -> None:
    """
    Re-center one FastPM BigFile Position dataset inside an output tree copy.

    Parameters
    ----------
    path : Path
        FastPM block directory inside the output tree.
    origin_min_mpc_h : float
        Current box minimum in Mpc/h.
    batch_size : int
        Particle batch size used while writing.
    """
    particle_format = detect_dm_particle_format(str(path))
    if particle_format != "fastpm_bigfile":
        raise ValueError(f"Expected FastPM BigFile positions at {path}")

    main_folder, complete_path = _fastpm_block_paths(str(path))
    bfile = _require_bigfile()(complete_path)
    position_data = bfile.open(f"{main_folder}/Position")
    n_total = position_data.size
    start = 0
    while start < n_total:
        end = min(start + batch_size, n_total)
        batch = np.asarray(position_data[start:end], dtype=np.float64)
        if batch.size == 0:
            break
        batch[:, :3] -= origin_min_mpc_h
        position_data[start:end] = batch
        start = end


def _prepare_output_tree(input_dir: Path, output_dir: Path, force: bool) -> None:
    """
    Copy the input local sample tree into the output directory.

    Parameters
    ----------
    input_dir : Path
        Source local sample root.
    output_dir : Path
        Destination root for the recentered copy.
    force : bool
        If True, remove an existing output directory before copying.

    Raises
    ------
    FileExistsError
        If ``output_dir`` exists and ``force`` is False.
    ValueError
        If ``output_dir`` is inside ``input_dir``.
    """
    input_resolved = input_dir.resolve()
    output_resolved = output_dir.resolve()
    if output_resolved == input_resolved:
        raise ValueError("output-dir must differ from local-sample-dir")
    if input_resolved in output_resolved.parents:
        raise ValueError("output-dir must not be nested inside local-sample-dir")
    if output_resolved.exists():
        if not force:
            raise FileExistsError(
                f"Output directory already exists: {output_resolved}. "
                "Pass --force to replace it."
            )
        shutil.rmtree(output_resolved)
    shutil.copytree(input_resolved, output_resolved)


def _update_sample_manifest(
    manifest_path: Path,
    box_size_mpc_h: float,
    apply_changes: bool,
) -> None:
    """
    Report or update ``sample_manifest.json`` box metadata.

    Parameters
    ----------
    manifest_path : Path
        Manifest path under the local sample root.
    box_size_mpc_h : float
        Local box side length in Mpc/h.
    apply_changes : bool
        If True, rewrite the manifest on disk.
    """
    if not manifest_path.is_file():
        logging.warning("Manifest not found: %s", manifest_path)
        return

    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    logging.info(
        "Manifest box metadata: min=%s max=%s size=%s",
        payload.get("box_min_mpc_h"),
        payload.get("box_max_mpc_h"),
        payload.get("box_size_mpc_h"),
    )
    logging.info(
        "Manifest after recenter: min=0.0 max=%.1f size=%.1f",
        box_size_mpc_h,
        box_size_mpc_h,
    )
    if not apply_changes:
        return

    payload["box_min_mpc_h"] = 0.0
    payload["box_max_mpc_h"] = float(box_size_mpc_h)
    payload["box_size_mpc_h"] = float(box_size_mpc_h)
    manifest_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def write_recentered_copy(
    input_dir: Path,
    output_dir: Path,
    origin_min_mpc_h: float,
    box_size_mpc_h: float,
    particle_batch_size: int,
    skip_bigfile: bool,
    force: bool,
) -> Path:
    """
    Copy the input tree and write recentered targets into the output tree.

    Parameters
    ----------
    input_dir : Path
        Source local sample root.
    output_dir : Path
        Destination root for the recentered copy.
    origin_min_mpc_h : float
        Current box minimum in Mpc/h.
    box_size_mpc_h : float
        Local box side length in Mpc/h.
    particle_batch_size : int
        Particle batch size for BigFile writes.
    skip_bigfile : bool
        If True, skip FastPM BigFile Position targets.
    force : bool
        If True, replace an existing output directory.

    Returns
    -------
    Path
        Output directory containing the recentered copy.
    """
    _prepare_output_tree(input_dir, output_dir, force)

    for spec in DEFAULT_TARGETS:
        if skip_bigfile and spec.kind == "bigfile_positions":
            logging.info("Skipping BigFile target during apply: %s", spec.relative_path)
            continue
        source_path = input_dir / spec.relative_path
        destination_path = output_dir / spec.relative_path
        if not source_path.exists():
            logging.warning("Skipping missing target during apply: %s", source_path)
            continue
        if spec.kind == "hlist":
            _recenter_text_catalog(
                source_path,
                destination_path,
                spec,
                origin_min_mpc_h,
                lambda line: _update_box_header_line(
                    line,
                    box_size_mpc_h,
                    HLIST_BOX_HEADER_PREFIX,
                ),
            )
        elif spec.kind == "rockstar_list":
            _recenter_text_catalog(
                source_path,
                destination_path,
                spec,
                origin_min_mpc_h,
                lambda line: _update_box_header_line(
                    line,
                    box_size_mpc_h,
                    ROCKSTAR_BOX_HEADER_PREFIX,
                ),
            )
        elif spec.kind == "text_particles":
            _recenter_text_particles(source_path, destination_path, origin_min_mpc_h)
        elif spec.kind == "bigfile_positions":
            _recenter_bigfile_positions(destination_path, origin_min_mpc_h, particle_batch_size)
        else:
            raise ValueError(f"Unsupported target kind: {spec.kind}")
        logging.info("Recentered %s -> %s", spec.relative_path, destination_path)

    manifest_path = output_dir / "sample_manifest.json"
    _update_sample_manifest(manifest_path, box_size_mpc_h, apply_changes=True)
    return output_dir.resolve()


def main(argv: Optional[Sequence[str]] = None) -> int:
    """
    Scan or apply local_sample box recentering.

    Parameters
    ----------
    argv : Optional[Sequence[str]], optional
        Command-line arguments without the program name.

    Returns
    -------
    int
        Process exit code.
    """
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = _parse_args(argv or sys.argv[1:])
    local_sample_dir = args.local_sample_dir.resolve()

    output_dir = args.output_dir.resolve()
    logging.info("Input dir: %s", local_sample_dir)
    logging.info("Output dir: %s", output_dir)
    logging.info(
        "Shift: subtract origin_min=%.1f Mpc/h to map [%.1f, %.1f] -> [0, %.1f]",
        args.origin_min_mpc_h,
        args.origin_min_mpc_h,
        args.origin_min_mpc_h + args.box_size_mpc_h,
        args.box_size_mpc_h,
    )
    logging.info("Mode: %s", "WRITE COPY" if args.apply else "DRY RUN")

    reports: list[TargetReport] = []
    for spec in DEFAULT_TARGETS:
        if args.skip_bigfile and spec.kind == "bigfile_positions":
            logging.info("Skipping BigFile target: %s", spec.relative_path)
            continue
        path = local_sample_dir / spec.relative_path
        report = scan_target(
            spec=spec,
            path=path,
            origin_min_mpc_h=args.origin_min_mpc_h,
            box_size_mpc_h=args.box_size_mpc_h,
            scan_rows=args.scan_rows,
            particle_batch_size=args.particle_batch_size,
        )
        reports.append(report)
        log_target_report(report)

    manifest_path = local_sample_dir / "sample_manifest.json"
    _update_sample_manifest(manifest_path, args.box_size_mpc_h, apply_changes=False)

    if not args.apply:
        logging.info(
            "Dry run complete. Re-run with --apply to write a recentered copy under %s.",
            output_dir,
        )
        return 0

    written_dir = write_recentered_copy(
        input_dir=local_sample_dir,
        output_dir=output_dir,
        origin_min_mpc_h=args.origin_min_mpc_h,
        box_size_mpc_h=args.box_size_mpc_h,
        particle_batch_size=args.particle_batch_size,
        skip_bigfile=args.skip_bigfile,
        force=args.force,
    )
    logging.info("Recentered copy written to %s", written_dir)
    logging.info("Input tree was not modified.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
