"""Pytest configuration: ensure the repository root is importable.

Placing this file at the repository root means pytest prepends the root to
``sys.path``, so the top-level packages (``core``, ``var``, ``models`` ...) can
be imported from the test modules. GARCH-optimiser convergence warnings are
silenced to keep the test output readable.
"""

import os
import sys
import warnings

sys.path.insert(0, os.path.dirname(__file__))

warnings.filterwarnings("ignore")
