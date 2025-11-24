import argparse
import logging
import os
import sys
import time

import numpy as np
from default_params import L_BOX, MP, NGRID

from density_field_properties.data.read_data import read_rockstar_halo_catalog
from density_field_properties.density_field.cic_deposit import (
    get_delta_density,
    load_density_field_cic,
)
from density_field_properties.density_field.utils import DensityFieldInfo, get_grid_cell
from density_field_properties.tidal_tensor import TIDAL_TENSOR_PATH, TidalTensorArray

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
    parser.add_argument("--density_info", type=str, default=None)
    parser.add_argument("--path", type=str, default="output/")
    parser.add_argument("--r_bins", type=int, default=RBINS)
    parser.add_argument("--r_min", type=float, default=RMIN)
    parser.add_argument("--r_max", type=float, default=RMAX)
    parser.add_argument("--rho_c_h2_msun_mpc3", type=float, default=RMIN)
    parser.add_argument("--omega_matter", type=float, default=RMAX)
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
    if params.read_from_path:
        output_path = os.path.join(params.path, f"{TIDAL_TENSOR_PATH}")
        tidal_tensor_array = TidalTensorArray.from_folder(path=output_path)
    else:
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
        tidal_tensor_array = TidalTensorArray.from_delta(
            delta=delta_density,
            box_size=density_info.box_size,
            path=params.path,
            gaussian_scale_list=gaussian_scale_array.tolist(),
        )

        del delta_density
        del gaussian_scale_array

    end_time = time.time()
    logging.info("- Tidal tensor calculation time: %f minutes" % ((end_time - start_time) / 60))

    halo_data = read_rockstar_halo_catalog(params.halo_file, n_lines=params.max_halos)
    halo_data[:, 1:4] = get_grid_cell(
        halo_data[:, 1:4], density_info.n_grid, density_info.box_size
    )

    halo_selected = (halo_data[:, 4] > RMIN) & (halo_data[:, 4] < RMAX)
    logging.info(
        f"Number of halos in the selected radius: {np.sum(halo_selected)} of {halo_selected.size}"
    )

    halo_data = halo_data[halo_selected]
    tidal_anisotropy, overdensity = tidal_tensor_array.get_tidal_anisotropy_and_overdensity(
        halo_data[:, 1], halo_data[:, 2], halo_data[:, 3], halo_data[:, 4]
    )

    output_path = os.path.join(params.path, f"{TIDAL_TENSOR_PATH}/")
    np.savetxt(output_path + "anisotropy.txt", tidal_anisotropy)
    np.savetxt(output_path + "overdensity.txt", overdensity)


if __name__ == "__main__":
    root = logging.getLogger()
    root.setLevel(os.environ.get("LOGLEVEL", "INFO"))
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] [%(asctime)s] %(message)s")
    main(sys.argv[1:])
