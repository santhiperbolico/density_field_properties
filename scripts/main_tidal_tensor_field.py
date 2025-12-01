import argparse
import logging
import os
import sys
import time

import numpy as np
from default_params import L_BOX, MP, NGRID

from density_field_properties.density_field.cic_deposit import (
    get_delta_density,
    load_density_field_cic,
)
from density_field_properties.density_field.utils import DensityFieldInfo
from density_field_properties.halo_catalog.utils import get_halo_catalog_reader
from density_field_properties.halo_environment_descriptors.tidal_anisotropy import (
    tidal_anisotropy_and_overdensity_from_halo_calaog,
)
from density_field_properties.tidal_tensor import TidalTensorArray

RMIN = 1000 / 4096
RMAX = 4
RBINS = 20
OMEGA_MATTER = 0.3


def get_params(argv: list[str]) -> argparse.Namespace:
    """
    Función de preprocesado de los argumentos del job de PreprocessingJob.

    Parameters
    ----------
    argv: list[str]
        Lista de argumentos del job.

    Returns
    -------
    params: dict[str, int]
        Parámetros con los identificadores del proecto y de la ETL.
    """
    logging.info("Running ArgumentParser")
    parser = argparse.ArgumentParser()

    parser.add_argument("--density_file", type=str)
    parser.add_argument("--halo_file", type=str)
    parser.add_argument("--max_halos", type=int, default=None)
    parser.add_argument("--read_from_path", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--halo_catalog_name", type=str, default="rockstar")
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--density_info", type=str, default=None)
    parser.add_argument("--path", type=str, default="output/")
    parser.add_argument("--r_bins", type=int, default=RBINS)
    parser.add_argument("--r_min", type=float, default=RMIN)
    parser.add_argument("--r_max", type=float, default=RMAX)
    # Only used if density_info is not provided
    parser.add_argument("--box_size", type=int, default=L_BOX)
    parser.add_argument("--n_particles", type=int, default=None)
    parser.add_argument("--mass_particle", type=float, default=MP)
    parser.add_argument("--n_grid", type=int, default=NGRID)

    return parser.parse_args(argv)


def main(argv: list[str]) -> None:
    params = get_params(argv)
    density_data, density_info = load_density_field_cic(params.density_file, params.density_info)
    if density_info is None:
        if params.n_particles is None:
            raise ValueError("n_particles must be specified if density_info is not provided.")
        density_info = DensityFieldInfo(
            n_grid=params.n_grid,
            box_size=params.box_size,
            mass_particle=params.mass_particle,
            n_particles=params.n_particles,
        )

    start_time = time.time()
    if not params.read_from_path:
        delta_density = get_delta_density(
            density_data,
            density_info.n_particles,
            density_info.mass_particle,
            density_info.box_size,
        )
        gaussian_scale_array = np.logspace(
            np.log10(params.r_min), np.log10(params.r_max), params.r_bins
        )
        gaussian_scale_array = gaussian_scale_array[1:]
        _ = TidalTensorArray.from_delta(
            delta=delta_density,
            box_size=density_info.box_size,
            path=params.path,
            gaussian_scale_list=gaussian_scale_array.tolist(),
        )

        del delta_density
        del gaussian_scale_array

        end_time = time.time()
        logging.info(
            "- Tidal tensor calculation time: %f minutes" % ((end_time - start_time) / 60)
        )

    start_time = time.time()
    halo_catalog_reader = get_halo_catalog_reader(params.halo_catalog_name)
    output_path = tidal_anisotropy_and_overdensity_from_halo_calaog(
        path=params.path,
        halo_catalog=halo_catalog_reader,
        halo_catalog_path=params.halo_file,
        n_grid=density_info.n_grid,
        n_lines=params.max_halos,
        box_size=density_info.box_size,
        batch_size=params.batch_size,
    )
    end_time = time.time()
    logging.info(
        "- Environment descriptors calculation time: %f minutes" % ((end_time - start_time) / 60)
    )
    logging.info("- Environment descriptors saved in %s" % output_path)
    logging.info("- End Process")


if __name__ == "__main__":
    root = logging.getLogger()
    root.setLevel(os.environ.get("LOGLEVEL", "INFO"))
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] [%(asctime)s] %(message)s")
    main(sys.argv[1:])
