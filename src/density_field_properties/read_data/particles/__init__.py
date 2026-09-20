"""DM particle I/O for text catalogs and FastPM BigFile snapshots."""

from density_field_properties.read_data.particles.particle_io import (
    detect_dm_particle_format,
    iter_dm_particle_batches,
)

__all__ = [
    "detect_dm_particle_format",
    "iter_dm_particle_batches",
]
