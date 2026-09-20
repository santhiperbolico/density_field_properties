"""Registry mapping Haloscope INPUT feature names to attachers."""

from typing import Sequence, Type

from density_field_properties.preprocessing.input_features.base import InputFeatureAttacher
from density_field_properties.preprocessing.input_features.n_halos_env import (
    ENV_FEATURE_NAME,
    NHalosEnvironmentFeature,
)
from density_field_properties.preprocessing.input_features.rockstar_t_over_u import (
    T_OVER_U_FEATURE_NAME,
    RockstarTOverUFeature,
)
from density_field_properties.preprocessing.input_features.tidal_anisotropy import (
    TIDAL_ANISOTROPY_FEATURE_NAME,
    TidalAnisotropyFeature,
)

ATTACHER_REGISTRY: dict[str, Type[InputFeatureAttacher]] = {
    ENV_FEATURE_NAME: NHalosEnvironmentFeature,
    T_OVER_U_FEATURE_NAME: RockstarTOverUFeature,
    TIDAL_ANISOTROPY_FEATURE_NAME: TidalAnisotropyFeature,
}


def _attacher_class(feature_name: str) -> Type[InputFeatureAttacher]:
    """
    Resolve the attacher class for one INPUT feature name.

    Parameters
    ----------
    feature_name : str
        Haloscope INPUT feature name.

    Returns
    -------
    Type[InputFeatureAttacher]
        Concrete attacher class.

    Raises
    ------
    ValueError
        If ``feature_name`` is not registered.
    """
    attacher_cls = ATTACHER_REGISTRY.get(feature_name)
    if attacher_cls is not None:
        return attacher_cls

    known = sorted(ATTACHER_REGISTRY)
    raise ValueError(f"Unknown input feature {feature_name!r}; known: {', '.join(known)}")


def resolve_attachers(input_features: Sequence[str]) -> list[InputFeatureAttacher]:
    """
    Build attacher instances for the requested Haloscope INPUT features.

    Parameters
    ----------
    input_features : Sequence[str]
        Ordered Haloscope INPUT column names.

    Returns
    -------
    list[InputFeatureAttacher]
        Attacher instances in the same order as ``input_features``.

    Raises
    ------
    ValueError
        If ``input_features`` is empty or contains unknown names.
    """
    if len(input_features) == 0:
        raise ValueError("input_features must contain at least one feature name")

    attachers: list[InputFeatureAttacher] = []
    seen: set[str] = set()
    for feature_name in input_features:
        if feature_name in seen:
            raise ValueError(f"Duplicate input feature: {feature_name}")
        seen.add(feature_name)
        attachers.append(_attacher_class(feature_name)())
    return attachers
