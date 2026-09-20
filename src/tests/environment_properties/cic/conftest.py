"""Test fixtures for environment_properties.cic."""

import sys
from unittest.mock import MagicMock

_bigfile_module = MagicMock()
_bigfile_module.BigFile = MagicMock
sys.modules.setdefault("bigfile", _bigfile_module)
