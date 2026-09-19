"""Put the analysis modules on the import path for the test suite.

The tests import pipeline modules by bare name (`from estimate import ...`),
which worked when everything sat at the repository root. Now that the code
lives in code/, pytest would otherwise only add code/tests/ to sys.path and
every one of those imports would fail. pytest loads this file before collecting
anything under code/, so adding this directory here fixes the whole suite
without touching a single test.

The SLURM wrappers do not need it: they run `python code/<script>.py` from the
repository root, which puts code/ at sys.path[0] automatically.
"""

from __future__ import annotations

import sys
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))
