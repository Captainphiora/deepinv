# DiffPIR alignment

This directory aligns DeepInv's `DiffPIR` with the official DiffPIR
`main_ddpir.py` inpainting path at commit
`2a9898129a1b274131b98746e5b364bc20adc1e1`.

The fixed setting uses the first three `demo_test` FFHQ images, an official
NumPy random mask with exactly 50% missing pixels, zero measurement noise,
algorithm sigma `0.001`, 20 quadratic reverse steps, `lambda=1`, `zeta=1`,
`eta=0`, and guidance scale `1`. The fixture stores the actual RGB mask,
measurement, initial diffusion state, and all 19 effective transition noises
as CPU `.pt` tensor dictionaries. DeepInv's generic mask generator is not used
because it has a different RNG and does not guarantee the same missing count.

Run the complete three-image reproduction without command-line parameters:

```bash
bash reproduction/diffpir/reproduce.sh
```

The shell file contains the setting, task, paths, devices, and run ID. It runs
the pinned original code and DeepInv concurrently on two same-model GPUs, then
uses the shared DPS comparator on a third GPU. The resulting
`comparison.json` reports reference metrics, DeepInv metrics, their deltas,
raw-tensor differences, and trajectory-probe differences.

`prepare_inputs.py`, `run_reference.py`, and `run_deepinv.py` retain explicit
CLI options for harness development, but they are not needed for the normal
reproduction command.
