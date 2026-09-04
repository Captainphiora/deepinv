# DSG alignment

This harness aligns DeepInv's discrete `deepinv.sampling.DSG` with the original
FFHQ inpainting implementation. The tracked setting is
`ffhq256_inpainting_ddim100_eta1_dsg_v1`: DDIM 100/1000, integer respacing
`100`, eta `1`, learned-range variance, clipping, Gaussian measurement noise
and `noise_level=0.05`, guidance scale `0.2`, and interval `1`. The original
`rescale_timesteps=true` operation is the identity because the training
schedule has 1000 steps.

The original algorithm is pinned to the upstream repository. The executable
reference is the modern-PyTorch compatibility fork and commit
`99b4e99c57886226fc852218286d951d65edab5e`; its unchanged upstream algorithm
comes from `b217c7a2463f9ebd68e12fe6b6d91344f195d1b8`. Pass
`--reference-repo` as the repository's `Linear_Inverse_Problems` directory.

## Fixed inputs

Reuse the shared fixture builder. Eta 1 is stochastic, so the transition tape
is mandatory even when all seeds are fixed:

```bash
python reproduction/dps/prepare_inputs.py \
  --setting ffhq256_inpainting_ddim100_eta1_dsg_v1 \
  --fixture-id ffhq256_inpainting_ddim100_eta1_dsg_v1 \
  --images "$DSG_IMAGES" \
  --measurements "$DSG_MEASUREMENTS" \
  --x-init "$DSG_XINIT" \
  --with-transition-noise
```

The fixture stores the clean image, random mask, noisy measurement, actual
measurement noise, initial point, schedule, and all 100 transition noises as
CPU `.pt` tensor dictionaries. The setting and its recorded hash preserve the
configured `noise_level`; seeds are provenance, while saved tensors are the
numerical inputs.

## Run and compare

Use the same fixture, run ID, environment, and GPU for both implementations:

```bash
python reproduction/dsg/run_reference.py \
  --setting ffhq256_inpainting_ddim100_eta1_dsg_v1 \
  --run-id alignment-001 \
  --reference-repo "$DSG_REPOSITORY/Linear_Inverse_Problems" \
  --checkpoint "$DSG_CHECKPOINT"

python reproduction/dsg/run_deepinv.py \
  --setting ffhq256_inpainting_ddim100_eta1_dsg_v1 \
  --run-id alignment-001 \
  --checkpoint "$DSG_CHECKPOINT"

python reproduction/dps/compare.py \
  --setting ffhq256_inpainting_ddim100_eta1_dsg_v1 \
  --fixture-id ffhq256_inpainting_ddim100_eta1_dsg_v1 \
  --run-id alignment-001
```

The reference loop imports DSG2024 rather than copying it. It injects exactly
one saved `randn_like` value into each `p_sample`, omits only the unused
`q_sample(measurement)` RNG draw, and calls the original conditioner with its
`mean`, `sigma_t`, and reduced-sampler `idx`. Outputs are raw tensors plus the
integer timestep map, explicit noise levels, and selected reverse-loop states.
They live under `reproduction/artifacts/runs/dsg/`; the shared comparison
script checks raw errors before PSNR, SSIM, and LPIPS.
