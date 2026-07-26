import os
from typing import Generator, Optional, Tuple

import numpy as np
from bigfile import BigFile


def _fastpm_block_paths(path: str) -> Tuple[str, str]:
    """
    Split a FastPM block path into block folder name and BigFile root path.

    Parameters
    ----------
    path : str
        Path ending in the block subdirectory (e.g. ``.../snap_1.0000/1``).

    Returns
    -------
    tuple[str, str]
        ``(main_folder, complete_path)`` where ``main_folder`` is the last path
        segment and ``complete_path`` is the parent BigFile directory.
    """
    normalized = os.path.normpath(path)
    main_folder = os.path.basename(normalized)
    complete_path = os.path.dirname(normalized)
    return main_folder, complete_path


def detect_dm_particle_format(path: str) -> str:
    """
    Detect whether DM particles are stored as a text file or FastPM BigFile block.

    Parameters
    ----------
    path : str
        Path to particle data (text file or FastPM block directory).

    Returns
    -------
    str
        ``"text"`` or ``"fastpm_bigfile"``.

    Raises
    ------
    ValueError
        If the path does not exist or does not match a supported layout.
    """
    normalized = os.path.normpath(path)
    if not os.path.exists(normalized):
        raise ValueError(f"DM particle path does not exist: {path}")
    if os.path.isfile(normalized):
        return "text"
    if not os.path.isdir(normalized):
        raise ValueError(f"Unsupported DM particle path: {path}")

    main_folder, complete_path = _fastpm_block_paths(normalized)
    try:
        bfile = BigFile(complete_path)
    except OSError as exc:
        raise ValueError(
            f"Directory does not look like a FastPM BigFile block path: {path}"
        ) from exc
    try:
        bfile["Header"]
    except KeyError as exc:
        raise ValueError(f"FastPM BigFile at {complete_path} has no Header/") from exc
    try:
        bfile.open(f"{main_folder}/Position")
    except (KeyError, OSError) as exc:
        raise ValueError(
            f"No Position dataset at {main_folder}/Position under {complete_path}"
        ) from exc
    return "fastpm_bigfile"


def _as_position_batch(data: np.ndarray) -> np.ndarray:
    """
    Normalize loadtxt or BigFile output to an ``(n, 3)`` position array.

    Parameters
    ----------
    data : np.ndarray
        Raw particle coordinates.

    Returns
    -------
    np.ndarray
        Array with shape ``(n, 3)``.
    """
    if data.ndim == 0 or data.size == 0:
        return np.empty((0, 3), dtype=np.float64)
    if data.ndim == 1:
        return data.reshape(1, -1)[:, :3]
    return data[:, :3]


def _count_text_particle_lines(path: str) -> int:
    """
    Count newline-separated particle records in a text positions file.

    Parameters
    ----------
    path : str
        Path to a whitespace-separated positions file.

    Returns
    -------
    int
        Number of lines in the file.
    """
    with open(path, encoding="utf-8") as handle:
        return sum(1 for _ in handle)


def _iter_text_batches(
    path: str, batch_size: Optional[int]
) -> Generator[Tuple[np.ndarray, int, int], None, None]:
    """
    Yield text-file particle batches with cumulative line-based indices.

    Parameters
    ----------
    path : str
        Path to a whitespace-separated positions file.
    batch_size : Optional[int]
        Maximum rows per batch; ``None`` reads the entire file once.

    Yields
    ------
    tuple[np.ndarray, int, int]
        Positions ``(n, 3)``, start index (inclusive), end index (exclusive).
    """
    if batch_size is None:
        data = np.loadtxt(path)
        positions = _as_position_batch(data)
        n_rows = positions.shape[0]
        if n_rows > 0:
            yield positions, 0, n_rows
        return

    skip = 0
    n_total = _count_text_particle_lines(path)
    while skip < n_total:
        max_rows = min(batch_size, n_total - skip)
        data = np.loadtxt(path, skiprows=skip, max_rows=max_rows)
        positions = _as_position_batch(data)
        n_rows = positions.shape[0]
        if n_rows == 0:
            break
        yield positions, skip, skip + n_rows
        skip += n_rows


def _iter_fastpm_bigfile_batches(
    path: str, batch_size: Optional[int]
) -> Generator[Tuple[np.ndarray, int, int], None, None]:
    """
    Yield FastPM BigFile position batches with particle index ranges.

    Parameters
    ----------
    path : str
        Path to the block subdirectory inside a BigFile snapshot.
    batch_size : Optional[int]
        Maximum particles per batch; ``None`` uses a single batch of all particles.

    Yields
    ------
    tuple[np.ndarray, int, int]
        Positions ``(n, 3)``, start index (inclusive), end index (exclusive).
    """
    main_folder, complete_path = _fastpm_block_paths(os.path.normpath(path))
    bfile = BigFile(complete_path)
    position_data = bfile.open(f"{main_folder}/Position")
    # position_data.size is equal to position_data[:].shape[0]
    n_total = position_data.size
    if n_total == 0:
        return

    effective_batch = n_total if batch_size is None else batch_size
    start = 0
    while start < n_total:
        end = min(start + effective_batch, n_total)
        batch = np.asarray(position_data[start:end], dtype=np.float64)
        positions = _as_position_batch(batch)
        yield positions, start, end
        start = end


def iter_dm_particle_batches(
    path: str, batch_size: Optional[int] = None
) -> Generator[Tuple[np.ndarray, int, int], None, None]:
    """
    Iterate DM particle positions in batches for text or FastPM BigFile inputs.

    Parameters
    ----------
    path : str
        Path to particle text file or FastPM block directory.
    batch_size : Optional[int], optional
        Maximum particles per batch; ``None`` loads all particles in one batch.

    Yields
    ------
    tuple[np.ndarray, int, int]
        Positions ``(n, 3)``, start index (inclusive), end index (exclusive).
    """
    fmt = detect_dm_particle_format(path)
    if fmt == "text":
        yield from _iter_text_batches(path, batch_size)
    else:
        yield from _iter_fastpm_bigfile_batches(path, batch_size)
