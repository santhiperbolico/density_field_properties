#!/usr/bin/env python
"""
Deprecated wrapper — use pipelines/run_haloscope_enrichment.py with --tidal-preset.

Quick run:

    PYTHONPATH=src python pipelines/run_haloscope_enrichment.py --tidal-preset --quick-run
"""

import logging
import sys
import warnings

from run_haloscope_enrichment import main as run_haloscope_enrichment_main


def main(argv: list[str]) -> int:
    """
    Execute the deprecated tidal CLI wrapper.

    Parameters
    ----------
    argv : list[str]
        Command-line arguments without the program name.

    Returns
    -------
    int
        Process exit code (0 on success).
    """
    warnings.warn(
        "pipelines/run_haloscope_enrichment_tidal.py is deprecated; "
        "use pipelines/run_haloscope_enrichment.py with --tidal-preset",
        DeprecationWarning,
        stacklevel=2,
    )
    forwarded_argv = list(argv)
    if "--tidal-preset" not in forwarded_argv:
        forwarded_argv = ["--tidal-preset"] + forwarded_argv
    return run_haloscope_enrichment_main(forwarded_argv)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    raise SystemExit(main(sys.argv[1:]))
