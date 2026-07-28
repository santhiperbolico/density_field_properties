"""Haloscope third-party model and SIM-to-FastPM pipeline.

The ``ConditionalMultiVariateGaussian`` implementation is vendored from
https://github.com/computationalAstroUAM/haloscope (see ``haloscope.py``).
"""

from density_field_properties.haloscope.haloscope import ConditionalMultiVariateGaussian

__all__ = ["ConditionalMultiVariateGaussian"]
