"""
Deforestation ML package.

This package contains utilities and entry points for training, evaluating,
and explaining models that predict annual deforestation area (hectares)
at the district-year (UBIGEO × YEAR) level for Peru.

Typical usage (via uv):
- uv run python -m deforestation.train_xgb --data <path> --sep <';' or '\\t'>
"""

from __future__ import annotations

__all__ = [
    "__version__",
]

__version__ = "0.1.0"
