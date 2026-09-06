# Reports

One self-contained HTML page per run, written automatically and rebuildable
without the model.

!!! tip "See a real one"
    [What a run looks like](example-output.md) embeds an actual `report.html`,
    not a screenshot of one.

```bash
xevals report runs/synthetic-abc123def456/
```

```python
result.report()                       # or Result.load(path).report()
```

## Three constraints

**Self-contained.** CSS is inline and figures are base64 data URIs, so the file
survives being emailed, attached to an issue, or opened from a USB stick with no
network. Videos stay as relative links, since inlining a megabyte of mp4 per
cell would defeat the purpose, so a travelling page degrades to "video missing"
rather than failing.

**No template engine.** f-strings and `html.escape`. A dependency for one page is a
dependency in every install.

**The caveats come first.** A failed replay gate, a truncated run, and every
dimension that could not be measured appear **above** the numbers. The whole point
of the library is that a single confident number hides the conditions it was
measured under; a report that buried them would reproduce the problem in a nicer
font.

## What the page contains, in order

1. Caveat banners: gate, budget, unmeasured dimensions.
2. Score cards, one per dimension, each in its fixed colour. An unmeasured
   dimension reads *not measured* with the reason, never a zero.
3. The radar.
4. The dimension table and bars.
5. Every condition, with the severity curves and the per-cell heatmap.
6. Efficiency against the control budget.
7. Baselines and the gate verdict.
8. The video gallery, labelled by cell.
9. Every metric, with intervals and `n`.
10. **How this was produced**: the exact command, and the full model,
    environment and library fingerprint.

The page follows `prefers-color-scheme` using the same tokens as the docs.

::: xevals.report
    options:
      heading_level: 2
      members: false
