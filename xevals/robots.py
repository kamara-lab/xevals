"""Compatibility import for :mod:`xevals.environments.robots`.

Keep this entry point lazy so importing xevals does not register simulators.
"""

import sys

from .environments import robots as _implementation

sys.modules[__name__] = _implementation
