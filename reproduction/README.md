# Reproduction protocol

This directory keeps small, reviewable code and settings for aligning DeepInv
algorithms with their original repositories. Raw inputs, checkpoints,
trajectories, and reconstructions stay under the ignored
`reproduction/artifacts/` directory (or `DEEPINV_REPRO_ROOT`).

Every reproduction follows the same order:

1. Pin the original repository commit and checkpoint SHA256 in a setting.
2. Materialize canonical inputs, including the measurement, operator state,
   initial point, schedule/noise levels, and any stochastic transition noise.
3. Run the pinned reference and DeepInv against those same tensors.
4. Compare raw tensors and selected trajectory steps before image metrics.
5. Commit a compact certification JSON only after all configured gates pass.

Canonical numerical artifacts are `dict[str, torch.Tensor]` `.pt` files. Load
them with `weights_only=True` and `map_location="cpu"`. Configuration,
provenance, hashes, and metrics are JSON. PNG files may be generated for human
inspection but are never metric inputs.

## Environment

From the repository root, restore the pinned Python 3.10 environment with:

```bash
"$HOME/uv-env-tool.sh" --source china --proxy off \
  sync --locked --group dev --extra reproduction
```

The environment lives in `.venv`; `uv.lock` is committed. Downloads are shared
automatically through `$HOME/.cache/uv`. Add a reproduction-only dependency
with `uv add` so both `pyproject.toml` and the lock stay current:

```bash
"$HOME/uv-env-tool.sh" --source china --proxy off \
  add --optional reproduction PACKAGE
```

Each new algorithm gets one subdirectory and adds only its runner, settings,
and reference notes. Do not copy an original implementation into DeepInv or
add a framework abstraction until a second working reproduction proves it is
needed.

See [`dps/README.md`](dps/README.md) for the first concrete workflow.
