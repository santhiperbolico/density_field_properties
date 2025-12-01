from density_field_properties.halo_catalog.fastpm import FastPMCatalogReader
from density_field_properties.halo_catalog.halo_catalog import HaloCatalogReader
from density_field_properties.halo_catalog.rockstar import RockstarCatalogReader


class HaloCatalogError(Exception):
    pass


def get_halo_catalog_reader(catalog_name: str) -> HaloCatalogReader:
    """
    Retrieve a halo catalog reader based on the given catalog name.

    This function checks if the provided catalog name corresponds to a supported
    halo catalog reader and returns the appropriate reader class. If the catalog
    name is not recognized, a ValueError is raised.

    Parameters
    ----------
    catalog_name : str
        The name of the catalog for which the reader is required.

    Raises
    ------
    HaloCatalogError
        If the provided catalog name is not supported.

    Returns
    -------
    HaloCatalogReader
        The reader class object corresponding to the provided catalog name.
    """
    catalogs = {
        RockstarCatalogReader.calog_name: RockstarCatalogReader,
        FastPMCatalogReader.calog_name: FastPMCatalogReader,
    }
    try:
        return catalogs[catalog_name]
    except KeyError:
        raise HaloCatalogError(f"Catalog {catalog_name} not supported.")
