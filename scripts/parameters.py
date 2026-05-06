"""Configuration for robot + vision + PF.

**Edit values in ``vision/parameters.py``** (single source of truth). This file only
re-exports them so ``import parameters`` works when you run code from ``scripts/``.
"""
from __future__ import annotations

import os
import sys

_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _root not in sys.path:
    sys.path.insert(0, _root)

from vision.parameters import *  # noqa: F403
