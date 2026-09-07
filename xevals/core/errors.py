"""The three ways an evaluation goes wrong, each with its own type.

They are separate because they call for different responses. A
:class:`MissingExtra` is fixed by an install. A :class:`CapabilityMissing` is
usually *not* an error at all -- the runner catches it and reports the dependent
metric as ``null`` with a reason, because "this model exposes no confidence, so
calibration is unmeasurable" is a result. A :class:`GateFailed` invalidates the
numbers that follow it, and is the one an automated pipeline should stop on.
"""

from __future__ import annotations

__all__ = ["CapabilityMissing", "GateFailed", "MissingExtra", "XevalsError"]


class XevalsError(Exception):
    """Base class, so callers can catch everything xevals raises deliberately."""


class MissingExtra(XevalsError, ImportError):
    """An optional dependency is needed here and is not installed.

    Also an :class:`ImportError`, so ``except ImportError`` around an optional
    code path keeps working.
    """

    def __init__(self, package: str, extra: str, purpose: str = "") -> None:
        tail = f" ({purpose})" if purpose else ""
        super().__init__(
            f"{package} is required{tail}. Install it with: pip install 'xevals[{extra}]'"
        )
        self.package = package
        self.extra = extra


class CapabilityMissing(XevalsError):
    """The model does not expose something a metric needs.

    Raised by metric code and caught by the runner, which records the metric as
    ``null`` and keeps the reason. A missing capability is a fact about the
    model, not a failure of the run.
    """

    def __init__(self, capability: str, metric: str = "") -> None:
        subject = f"metric {metric!r}" if metric else "this metric"
        super().__init__(f"{subject} needs {capability!r}, which the model does not provide")
        self.capability = capability
        self.metric = metric


class GateFailed(XevalsError):
    """A precondition of the protocol did not hold, so the numbers do not mean much.

    The canonical case is the replay gate: if replaying a demonstrator's own
    actions does not solve the task, the environment is not faithful to the data
    and no model's success rate on it is interpretable.
    """
