"""Wrapping a torch module, and measuring what it costs to run.

The point is the adapter, not the network: an untrained MLP is a terrible policy
and its accuracy says so. What it demonstrates is that ``xevals.wrap`` puts the
module in eval mode, runs it under ``no_grad``, moves tensors to the device and
back, and reads the parameter count into the efficiency dimension -- the four
things a hand-written evaluation script gets subtly wrong.

Needs ``xevals[torch]``.

    python examples/03_torch_policy.py
"""

from __future__ import annotations

import sys

from _common import setup

import xevals


def main() -> int:
    try:
        import torch
    except ImportError:
        print("this example needs torch: pip install 'xevals[torch]'", file=sys.stderr)
        return 0

    torch.manual_seed(0)
    module = torch.nn.Sequential(
        torch.nn.Linear(8, 64), torch.nn.Tanh(), torch.nn.Linear(64, 2), torch.nn.Tanh()
    )
    module.train()  # deliberately: the adapter is what puts it back in eval mode

    policy = xevals.wrap(module)
    assert not policy.module.training
    print(f"wrapped {policy.describe()}")

    result = xevals.evaluate(policy, "synthetic/reach", suite="core",
                            **setup("03_torch_policy"))
    print()
    print(result.table(kind="metrics"))
    print()
    print(
        "An untrained network scores near the random baseline, which is the\n"
        "answer. What the run adds is the latency, the parameter count and the\n"
        "control-rate headroom -- the numbers that decide whether it could run\n"
        "on the robot even if it were good."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
