import argparse
import logging
import os
import sys

from default_params import L_BOX, MP, NGRID

from density_field_properties.density_field.cic_deposit import (
    get_delta_density,
    load_density_field_cic,
)
from density_field_properties.density_field.utils import DensityFieldInfo
from density_field_properties.tidal_tensor import TIDAL_TENSOR_PATH, TidalTensor

RMIN = 1000 * 1000 / 4096


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
    parser.add_argument("--density_info", type=str, default=None)
    parser.add_argument("--path", type=str, default="output/")
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

    delta_density = get_delta_density(
        density_data, density_info.n_particles, density_info.mass_particle, density_info.box_size
    )

    _ = TidalTensor.from_delta(
        delta=delta_density, box_size=density_info.box_size, path=params.path, gaussian_scale=RMIN
    )

    output_path = os.path.join(params.path, f"{TIDAL_TENSOR_PATH}_{RMIN: .0f}")
    tidal_tensor = TidalTensor.from_folder(path=output_path, gaussian_scale=RMIN)

    eigenvalues = tidal_tensor.eigenvalues(1, 2, 3)
    logging.info("Eigenvalues: %f, %f, %f" % tuple(eigenvalues[0]))


if __name__ == "__main__":
    root = logging.getLogger()
    root.setLevel(os.environ.get("LOGLEVEL", "INFO"))
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] [%(asctime)s] %(message)s")
    main(sys.argv[1:])
