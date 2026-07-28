from abc import ABC, abstractmethod
from typing import Generator, Optional, Tuple

import numpy as np


class HaloCatalogData:
    """
    Represents a catalog of halo data with positions, masses, and other properties.

    This class is designed to handle and manipulate astrophysical or simulation-related
    data for halos. It provides functionalities to extract specific properties such as
    halo ID, positions, mass (M200b), and R_G. Additionally, it allows saving customized
    subsets of this data into files.

    Attributes
    ----------
    id_position : int
        Index of the halo ID column within the data.
    x_position : int
        Index of the X-coordinate column within the data.
    y_position : int
        Index of the Y-coordinate column within the data.
    z_position : int
        Index of the Z-coordinate column within the data.
    m200b_position : Optional[int]
        Index of the M200b (halo mass) column within the data if specified; otherwise None.
    rg_position : Optional[int]
        Index of the R_G column within the data if specified; otherwise None.
    data : np.ndarray
        Processed subset of the input data, containing only the specified columns.
    """

    def __init__(
        self,
        data: np.ndarray,
        halo_id_position: int,
        halo_x_position: int,
        halo_y_position: int,
        halo_z_position: int,
        halo_m200b_position: Optional[int] = None,
        halo_rg_position: Optional[int] = None,
    ):
        columns = [halo_id_position, halo_x_position, halo_y_position, halo_z_position]
        self.id_position = 0
        self.x_position = 1
        self.y_position = 2
        self.z_position = 3
        self.m200b_position = None
        self.rg_position = None

        extra_position = 4
        if halo_m200b_position is not None:
            self.m200b_position = extra_position
            extra_position += 1
            columns.append(halo_m200b_position)
        if halo_rg_position is not None:
            self.rg_position = extra_position
            columns.append(halo_rg_position)
        self.data = data[:, columns]

    @property
    def n_halos(self):
        return self.data.shape[0]

    @property
    def halo_id(self):
        return self.data[:, self.id_position]

    @property
    def halo_x(self):
        return self.data[:, self.x_position]

    @property
    def halo_y(self):
        return self.data[:, self.y_position]

    @property
    def halo_z(self):
        return self.data[:, self.z_position]

    @property
    def halo_m200b(self):
        if self.m200b_position is None:
            raise ValueError("M200b column not found in catalog")
        return self.data[:, self.m200b_position]

    @property
    def halo_rg(self):
        if self.rg_position is None:
            raise ValueError("R_G = 4 R200b / sqrt(5) column not found in catalog")
        return self.data[:, self.rg_position]

    def save_properties(
        self,
        path: str,
        properties_data: np.ndarray | tuple[np.ndarray],
        properties_header: str,
        save_m200b: bool = True,
        save_rg: bool = True,
        save_positions: bool = True,
    ) -> None:
        """
        Save specified properties data to a text file with customizable columns and headers.

        This function allows saving astrophysical or simulation-related data to a text file in a
        specific format. The user can customize which columns to include (e.g., positions,
        halo mass, and custom properties). The provided `properties_data` can either be a
        1D or 2D NumPy array. Headers and relevant metadata are formatted and included in the
        output file.

        Parameters
        ----------
        path : str
            Path to the output file where the data will be saved.
        properties_data : np.ndarray or tuple[np.ndarray]
            The custom properties data to save in the file. Must be either a 1D or 2D
            NumPy array. Raises a ValueError if not correctly structured.
        properties_header : str
            The header string for the custom properties, describing the data columns in
            `properties_data`.
        save_m200b : bool, optional
            Whether to include the M200b column in the output file. Default is True.
        save_rg : bool, optional
            Whether to include the calculated R_G column in the output file. Default is True.
        save_positions : bool, optional
            Whether to include the position columns (X, Y, Z) in the output file. Default is True.

        Raises
        ------
        ValueError
            If `properties_data` is not a 1D or 2D NumPy array.

        Notes
        -----
        The function expects `self.id_position`, `self.x_position`, `self.y_position`,
        `self.z_position`, `self.m200b_position`, and `self.rg_position` to refer to valid index
        or slicing operations on `self.data`. Custom properties provided in `properties_data` are
        appended to these columns.

        """
        if isinstance(properties_data, np.ndarray) and len(properties_data.shape) > 2:
            raise ValueError("properties_data must be a 1D or 2D array")
        if isinstance(properties_data, np.ndarray) and len(properties_data.shape) == 2:
            properties_data = (properties_data[:, col] for col in range(properties_data.shape[1]))
        if isinstance(properties_data, np.ndarray) and len(properties_data.shape) == 1:
            properties_data = (properties_data,)

        cols_to_save = [self.id_position]
        header_data = "Halo ID"
        if save_positions:
            header_data += ", X, Y, Z"
            cols_to_save += [self.x_position, self.y_position, self.z_position]
        if save_m200b:
            header_data += ", M200b"
            cols_to_save.append(self.m200b_position)
        if save_rg:
            header_data += ", R_G = 4 R200b / sqrt(5)"
            cols_to_save.append(self.rg_position)
        header_data += f", {properties_header} \n"
        properties_data = tuple(self.data[:, col] for col in cols_to_save) + properties_data
        np.savetxt(path, properties_data, header=header_data)


class HaloCatalogReader(ABC):
    calog_name: str = "general"

    @staticmethod
    @abstractmethod
    def read_catalog(path: str, **kwargs) -> HaloCatalogData:
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def read_catalog_batch_generator(
        path: str, batch_size: int, start_offset: int = 0, **kwargs
    ) -> Generator[Tuple[HaloCatalogData, int, int], None, None]:
        raise NotImplementedError
