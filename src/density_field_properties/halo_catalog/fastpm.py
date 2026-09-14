from pathlib import Path
from typing import Any, Generator, Optional, Tuple

import numpy as np
from bigfile import BigFile

from density_field_properties.cosmology import Cosmology
from density_field_properties.halo_catalog.halo_catalog import HaloCatalogData, HaloCatalogReader

FASTPM_HALO_COLUMNS_POSITION = {
    "halo_id": "ID",
    "halo_position": "Position",
    "halo_length": "Length",
}

MSUN_G = 1.989e33


def read_fastpm_cosmology_header(header: dict[str, Any]) -> Cosmology:
    return Cosmology(
        omega_matter=float(header["OmegaM"][0]),
        omega_lambda=float(header["OmegaLambda"][0]),
        h0=float(header["HubbleParam"][0]),
    )


def _resolve_fastpm_catalog_layout(path: str) -> tuple[BigFile, str]:
    """
    Resolve BigFile root and halo block prefix for a FastPM catalog path.

    Parameters
    ----------
    path : str
        Either a self-contained FOF directory with an internal ``Header`` block,
        or a ``parent/subfolder`` path where ``Header`` lives in ``parent``.

    Returns
    -------
    tuple[BigFile, str]
        Open BigFile handle and block prefix (for example ``LL-0.200`` or ``000``).

    Raises
    ------
    ValueError
        If no halo blocks with ``ID`` and ``Length`` can be found.
    """
    catalog_path = Path(path)
    if (catalog_path / "Header").is_dir():
        bfile = BigFile(str(catalog_path))
        for child in sorted(catalog_path.iterdir()):
            if not child.is_dir() or child.name == "Header":
                continue
            if (child / "ID").is_dir() and (child / "Length").is_dir():
                return bfile, child.name
        raise ValueError(f"No FastPM halo blocks found under {path}")

    main_folder = path.split("/")[-1]
    complete_path = "/".join(path.split("/")[:-1])
    return BigFile(complete_path), main_folder


class FastPMCatalogReader(HaloCatalogReader):
    calog_name: str = "fastpm"

    @staticmethod
    def read_catalog(
        path: str,
        n_lines: Optional[int] = None,
        halo_id_block: int = FASTPM_HALO_COLUMNS_POSITION["halo_id"],
        halo_position_block: int = FASTPM_HALO_COLUMNS_POSITION["halo_position"],
        halo_length_block: int = FASTPM_HALO_COLUMNS_POSITION["halo_length"],
    ) -> HaloCatalogData:
        """
        Reads a halo catalog from a given file path and processes its data to construct
        a HaloCatalogData object. This function relies on reading attributes and data
        blocks from a `BigFile` object to extract halo IDs, particle counts, positions,
        and masses, along with additional processing supported for specific cosmologies.

        Parameters
        ----------
        path : str
            File path to the halo catalog, specifying the folder and subdirectory to read.
        n_lines : Optional[int], default None
            Number of rows to read from the halo catalog. If None, reads all available rows.
        halo_id_block : int, optional
            The block index for reading halo IDs in the catalog file, defaulting to the
            corresponding index based on the FASTPM_HALO_COLUMNS_POSITION dictionary.
        halo_position_block : int, optional
            The block index for reading halo position coordinates, defaulting to the
            corresponding index based on the FASTPM_HALO_COLUMNS_POSITION dictionary.
        halo_length_block : int, optional
            The block index for reading halo particle counts, defaulting to the
            corresponding index based on the FASTPM_HALO_COLUMNS_POSITION dictionary.

        Returns
        -------
        HaloCatalogData
            A processed object that contains the extracted and formatted data from the
            halo catalog, such as halo IDs, spatial positions, masses, and additional
            radius-based metrics for specific cosmologies.
        """

        bfile, block_prefix = _resolve_fastpm_catalog_layout(path)
        cosmology = read_fastpm_cosmology_header(bfile["Header"].attrs)
        mp = float(bfile["Header"].attrs["UnitMass_in_g"][0]) / MSUN_G / cosmology.h0

        positions = (0, 1, 2, 3, 4)
        if isinstance(cosmology, Cosmology):
            positions = (0, 1, 2, 3, 4, 5)

        if n_lines is None:
            n_lines = bfile.open(f"{block_prefix}/{halo_id_block}").size
        halo_id = bfile.open(f"{block_prefix}/{halo_id_block}")[:n_lines]
        halo_particles = bfile.open(f"{block_prefix}/{halo_length_block}")[:n_lines]
        halo_x = bfile.open(f"{block_prefix}/{halo_position_block}")[:n_lines][:, 0]
        halo_y = bfile.open(f"{block_prefix}/{halo_position_block}")[:n_lines][:, 1]
        halo_z = bfile.open(f"{block_prefix}/{halo_position_block}")[:n_lines][:, 2]

        mfof = halo_particles * mp

        row = (halo_id, halo_x, halo_y, halo_z, mfof)
        if isinstance(cosmology, Cosmology):
            rg = 4.0 * cosmology.convert_m200b_to_r200b(mfof) / np.sqrt(5.0)
            row = (halo_id, halo_x, halo_y, halo_z, mfof, rg)

        results = np.concatenate(tuple(r.reshape(-1, 1) for r in row), axis=1)

        results = HaloCatalogData(results, *positions)
        return results

    @staticmethod
    def read_catalog_batch_generator(
        path: str,
        batch_size: int,
        start_offset: int = 0,
        n_lines: Optional[int] = None,
        halo_id_block: int = FASTPM_HALO_COLUMNS_POSITION["halo_id"],
        halo_position_block: int = FASTPM_HALO_COLUMNS_POSITION["halo_position"],
        halo_length_block: int = FASTPM_HALO_COLUMNS_POSITION["halo_length"],
    ) -> Generator[Tuple[HaloCatalogData, int, int], None, None]:
        """
        Reads a batch of halo catalog data from a file and yields it as a generator
        along with the starting and ending indices of the batch in the file.

        This static method enables efficient reading of large datasets by dividing
        the data into manageable batches. Each batch consists of halo IDs,
        particle positions, and other metadata. The generator yields tuples
        containing the batch data, the start index of the batch in the file,
        and the next offset after the batch.

        Parameters
        ----------
        path : str
            Path to the folder containing the halo catalog data.
        batch_size : int
            Number of entries to include in each batch.
        start_offset : int, optional
            The starting offset position in the file from which to begin reading.
            Defaults to 0.
        n_lines : Optional[int], optional
            The number of entries to read in total. If not specified, the method
            reads until the end of the file.
        halo_id_block : int
            The block ID indicating the position of halo IDs in the file.
        halo_position_block : int
            The block ID indicating the position of halo positions in the file.
        halo_length_block : int
            The block ID indicating the position of the length of halo data in the
            file.

        Returns
        -------
        Generator[Tuple[HaloCatalogData, int, int], None, None]
            A generator that yields a tuple containing the following:
            - `HaloCatalogData`: The batch of halo catalog data.
            - `int`: The starting index of the batch in the file.
            - `int`: The next offset index after the batch.
        """

        bfile, block_prefix = _resolve_fastpm_catalog_layout(path)
        cosmology = read_fastpm_cosmology_header(bfile["Header"].attrs)
        mp = float(bfile["Header"].attrs["UnitMass_in_g"][0]) / MSUN_G / cosmology.h0

        positions = (0, 1, 2, 3, 4)
        if isinstance(cosmology, Cosmology):
            positions = (0, 1, 2, 3, 4, 5)

        if n_lines is None:
            n_lines = bfile.open(f"{block_prefix}/{halo_id_block}").size

        batch_readed = 0
        while batch_readed < n_lines:
            next_offset = min(batch_size, n_lines - batch_readed) + batch_readed

            halo_id = bfile.open(f"{block_prefix}/{halo_id_block}")[batch_readed:next_offset]
            halo_particles = bfile.open(f"{block_prefix}/{halo_length_block}")[
                batch_readed:next_offset
            ]
            halo_x = bfile.open(f"{block_prefix}/{halo_position_block}")[batch_readed:next_offset][
                :, 0
            ]
            halo_y = bfile.open(f"{block_prefix}/{halo_position_block}")[batch_readed:next_offset][
                :, 1
            ]
            halo_z = bfile.open(f"{block_prefix}/{halo_position_block}")[batch_readed:next_offset][
                :, 2
            ]
            mfof = halo_particles * mp
            row = (halo_id, halo_x, halo_y, halo_z, mfof)
            if isinstance(cosmology, Cosmology):
                rg = 4.0 * cosmology.convert_m200b_to_r200b(mfof) / np.sqrt(5.0)
                row = (halo_id, halo_x, halo_y, halo_z, mfof, rg)

            batch = np.concatenate(tuple(r.reshape(-1, 1) for r in row), axis=1)
            batch = HaloCatalogData(batch, *positions)
            yield batch, batch_readed, next_offset
            batch_readed = next_offset
