# DPS alignment

This harness compares DeepInv's discrete DPS implementation with the sampler
from the pinned original DPS repository. It does not treat the existing
continuous-SDE `deepinv.sampling.DPS` as the same algorithm.

Two settings are tracked:

- `ffhq256_inpainting_ddpm1000_v1`: original 1000-step DDPM with learned-range
  variance.
- `ffhq256_inpainting_ddim100_eta0_v1`: deterministic 100-step DDIM using the
  original repository's integer `timestep_respacing=100` behavior.

Both use the FFHQ checkpoint, DPS scale `0.5`, Gaussian measurement noise
`sigma=0.05`, the model's native integer timestep input, epsilon prediction,
and the explicitly saved timestep-to-noise-level schedule. `noise_level`
remains a first-class artifact even though these reference runs feed native
timesteps to the model. The compatibility target has
`rescale_timesteps=false`; other timestep-rescaling conventions require a new
setting and adapter test.

## Artifact contract

The default artifact root is `reproduction/artifacts/` and is gitignored. Set
`DEEPINV_REPRO_ROOT` or pass `--artifact-root` to place it elsewhere. A fixture
contains:

```text
fixtures/<fixture-id>/
  manifest.json
  schedule.pt
  cases/00000.pt
  noise/00000.pt        # only when an explicit DDPM tape is requested
```

Each case contains `ground_truth`, `measurement`, `mask`,
`measurement_noise`, and `x_init`. `schedule.pt` contains integer timesteps,
betas, cumulative alphas, and noise levels. DDPM and DDIM with `eta>0` require
the same explicit transition-noise tape in both runners. Only DDIM with
`eta=0` may omit it, because its stochastic term is identically zero.

Outputs live under
`runs/dps/<setting-id>/<run-id>/cases/<case-id>/{reference,deepinv}.pt`.
Only the fixed initial state and configured trajectory probes are stored, not
the full trajectory. Probe values are reverse-loop iteration indices: step
`0` is the first update from the highest-noise state, not diffusion timestep
zero.

## Prepare the three fixed inputs

The converter accepts the already aligned legacy `[measurement, mask]` files
and clean PNGs. Paths are supplied by the caller; the scripts contain no
machine-specific paths.

```bash
python reproduction/dps/prepare_inputs.py \
  --setting ffhq256_inpainting_ddpm1000_v1 \
  --images "$DPS_IMAGES" \
  --measurements "$DPS_MEASUREMENTS" \
  --x-init "$DPS_XINIT" \
  --with-transition-noise
```

To test only a changed initial point for the third image without changing the
algorithm, create a new immutable fixture ID and override that case explicitly:

```bash
python reproduction/dps/prepare_inputs.py \
  --setting ffhq256_inpainting_ddim100_eta0_v1 \
  --fixture-id ffhq256_inpainting_xinit43_v1 \
  --images "$DPS_IMAGES" \
  --measurements "$DPS_MEASUREMENTS" \
  --x-init "$DPS_XINIT" \
  --x-init-seed 00002=43
```

The generated tensor, not the seed alone, is the comparison input. Use
`--dry-run` first to inspect resolved cases and destinations without writing.

## Run and compare

Use the same `run-id` for both implementations. The reference runner imports
the original modules rather than copying their formulas. It permits a newer
checkout only when `guided_diffusion/` is byte-equivalent to pinned commit
`effbde7325b22ce8dc3e2c06c160c021e743a12d`.
The runner delegates every denoising transition and DPS gradient to those
modules. It omits only the original loop's `q_sample(measurement)` call because
the `ps` conditioner never reads that value; the actual transition noise is
injected from the saved tape instead of relying on global RNG call order.

```bash
python reproduction/dps/run_reference.py \
  --setting ffhq256_inpainting_ddim100_eta0_v1 \
  --run-id alignment-001 \
  --reference-repo "$DPS_REFERENCE_REPO" \
  --checkpoint "$DPS_CHECKPOINT"

python reproduction/dps/run_deepinv.py \
  --setting ffhq256_inpainting_ddim100_eta0_v1 \
  --run-id alignment-001 \
  --checkpoint "$DPS_CHECKPOINT"

python reproduction/dps/compare.py \
  --setting ffhq256_inpainting_ddim100_eta0_v1 \
  --run-id alignment-001
```

All three commands accept repeatable `--case` arguments. For example, append
`--fixture-id ffhq256_inpainting_xinit43_v1 --case 00002` to each command for
the isolated third-image run.

Comparison fails with a non-zero exit code if timesteps or noise levels differ,
if a selected trajectory probe exceeds its MAE/relative-L2 gate, or if the
per-case/mean PSNR and SSIM deltas exceed the setting. Metrics are computed
from raw tensors with the original evaluation packages: skimage receives the
common `[-1,1]` to `[0,1]` conversion, while `lpips.LPIPS` receives NCHW
`[-1,1]` tensors. The raw, unclipped reconstruction remains in each `.pt`
file for numerical comparison.

The DeepInv runner refuses a certification run while its model adapter,
sampler, exports, or DPS harness are uncommitted, so the recorded revision
always identifies the executed implementation. `--dry-run` remains available
during development.

No certification result is included yet. A result is valid only after both
full GPU runs finish and `compare.py` reports `PASS`.
