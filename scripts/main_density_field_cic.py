import argparse
import logging
import os
import sys
import time

from default_params import BATCH_SIZE, L_BOX, MP, NGRID

from density_field_properties.density_field.cic_deposit import (
    density_field_cic_main,
    save_density_field_cic,
)


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

    parser.add_argument("--dm_particles_file", type=str)
    parser.add_argument("--path", type=str, default="output/")
    parser.add_argument("--box_size", type=int, default=L_BOX)
    parser.add_argument("--mass_particle", type=float, default=MP)
    parser.add_argument("--n_grid", type=int, default=NGRID)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)

    return parser.parse_args(argv)


def main(argv: list[str]) -> None:
    params = get_params(argv)

    if not os.path.exists(params.path):
        os.makedirs(params.path)

    start_time = time.time()
    density, density_info = density_field_cic_main(
        dm_particles_file=params.dm_particles_file,
        mass_particle=params.mass_particle,
        box_size=params.box_size,
        n_grid=params.n_grid,
        batch_size=params.batch_size,
    )

    end_time = time.time()
    density_info.add_process_duration(end_time - start_time)
    duration = density_info.process_duration / 60  # minutes
    logging.info("- Saving density field")
    output_file = save_density_field_cic(
        density=density,
        path=params.path,
        dm_particles_file=params.dm_particles_file,
        density_info=density_info,
    )
    logging.info("- Density field saved in %s" % output_file)
    logging.info("- End Process: %f min." % duration)


if __name__ == "__main__":
    root = logging.getLogger()
    root.setLevel(os.environ.get("LOGLEVEL", "INFO"))
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] [%(asctime)s] %(message)s")
    main(sys.argv[1:])
