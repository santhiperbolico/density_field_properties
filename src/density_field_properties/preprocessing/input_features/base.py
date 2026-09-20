"""Base contract for Haloscope input feature attachers."""

from abc import ABC, abstractmethod

import pandas as pd

from density_field_properties.preprocessing.context import SimulationRunContext


class InputFeatureAttacher(ABC):
    """Adds one or more Haloscope INPUT columns to a halo table."""

    @property
    @abstractmethod
    def feature_names(self) -> tuple[str, ...]:
        """
        Return column names produced or required by this attacher.

        Returns
        -------
        tuple[str, ...]
            Haloscope INPUT feature names.
        """

    @abstractmethod
    def attach(
        self,
        catalog: pd.DataFrame,
        run_context: SimulationRunContext,
    ) -> pd.DataFrame:
        """
        Attach input feature columns to a halo catalog.

        Parameters
        ----------
        catalog : pd.DataFrame
            Halo table loaded from Rockstar or consistent-trees.
        run_context : SimulationRunContext
            Simulation-specific paths and grid parameters.

        Returns
        -------
        pd.DataFrame
            The same ``catalog`` instance with feature columns attached.
        """
