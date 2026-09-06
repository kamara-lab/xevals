# Examples

Every script in `examples/` runs on a bare install unless it says otherwise, and
reads its size from the environment so the same file serves a five-second check
and a real evaluation:

```bash
python examples/01_synthetic.py
XEVALS_EPISODES=2 XEVALS_SEEDS=1 python examples/01_synthetic.py    # what CI runs
```

| Variable | Default | Meaning |
|---|---|---|
| `XEVALS_EPISODES` | 10 | episodes per seed |
| `XEVALS_SEEDS` | 3 | how many seeds |
| `XEVALS_RECORD` | 2 | episodes per cell to keep frames for |
| `XEVALS_OUT` | `examples/outputs` | where run directories go |

## The scripts

**`01_synthetic.py`**: the full seven-dimension evaluation on the built-in
world. Produces every artefact the library makes, in under a minute, with no
extras. The one to read first. It ends by printing the worst cells, which is
usually the only part anyone needs.

**`02_state_vs_image.py`**: two policies, one suite. Both solve the task and
are indistinguishable on success rate; under perturbation they are not. This is the
argument for the whole library, run rather than asserted.

**`03_torch_policy.py`**: wrapping a `torch.nn.Module`. The network is
untrained and terrible on purpose: what the example demonstrates is the adapter
putting it in eval mode, running under `no_grad`, and reading the parameter count
into the efficiency dimension. Needs `xevals[torch]`.

**`04_offline_dataset.py`**: evaluation against a recorded corpus with no
simulator. Shows action error as a real measurement and success rate reporting
`null` with its reason, which is the point.

**`05_llm_planner.py`**: plan correctness, refusal on unsafe instructions,
and injection resistance, measured separately. The stand-in planner writes correct
plans, declines unsafe requests, and does whatever the last sentence in its prompt
says, which is a common shape, and invisible to a plan-quality benchmark.

**`06_compare_runs.py`**: three runs into a leaderboard, an overlaid radar
and a diff.

**`07_benchmark.py`**: four models in one run. Two of them are close on
purpose: the interesting question is not which is better on average but under
which conditions they differ at all, and `disagreements()` is what answers it.

## Findings

The measured observations from these runs live on the [Findings](../findings.md)
page, with the conditions they were measured under.
