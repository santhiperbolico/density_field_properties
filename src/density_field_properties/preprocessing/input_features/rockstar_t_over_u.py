"""Rockstar T/|U| column input feature."""

import pandas as pd

from density_field_properties.preprocessing.context import SimulationRunContext
from density_field_properties.preprocessing.input_features.base import InputFeatureAttacher
from density_field_properties.preprocessing.schemas import SchemaValidationError

T_OVER_U_FEATURE_NAME = "t_over_u"


class RockstarTOverUFeature(InputFeatureAttacher):
    """Verify Rockstar ``T/|U|`` is present on the loaded catalog."""

    @property
    def feature_names(self) -> tuple[str, ...]:
        """
        Return the Rockstar tidal column name.

        Returns
        -------
        tuple[str, ...]
            Single-element tuple with ``t_over_u``.
        """
        return (T_OVER_U_FEATURE_NAME,)

    def attach(
        self,
        catalog: pd.DataFrame,
        run_context: SimulationRunContext,
    ) -> pd.DataFrame:
        """
        Ensure ``t_over_u`` was loaded from the Rockstar catalog reader.

        Parameters
        ----------
        catalog : pd.DataFrame
            Halo table expected to contain ``t_over_u``.
        run_context : SimulationRunContext
            Unused; accepted for a uniform attacher interface.

        Returns
        -------
        pd.DataFrame
            The same ``catalog`` instance.

        Raises
        ------
        SchemaValidationError
            If ``t_over_u`` is missing from the catalog.
        """
        if T_OVER_U_FEATURE_NAME not in catalog.columns:
            raise SchemaValidationError(
                f"Catalog is missing Rockstar column {T_OVER_U_FEATURE_NAME!r}; "
                "ensure the Rockstar reader loads T/|U| for this list format."
            )
        return catalog
