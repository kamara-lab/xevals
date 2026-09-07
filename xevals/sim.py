"""Compatibility import for :mod:`xevals.environments.sim`.

Keep this entry point lazy so importing xevals does not register simulators.
"""

import sys

from .environments import sim as _implementation

sys.modules[__name__] = _implementation
