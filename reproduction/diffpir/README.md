# DiffPIR alignment

论文四个退化族、20/100 NFE 超参数及官方 demo 差异见
[`PAPER_BENCHMARK_SETTINGS.zh-CN.md`](PAPER_BENCHMARK_SETTINGS.zh-CN.md)。

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

## Gaussian deblur（当前第一阶段）

以下脚本固定使用 `DiffPIR/testsets/demo_test` 的五张 PNG、论文的 61×61 / std=3.0 Gaussian kernel，并覆盖有/无测量噪声及 20/100 NFE 四组论文参数：

```bash
bash reproduction/diffpir/reproduce_gaussian_deblur.sh
```

脚本不接受参数。它先在四张同型号 GPU 上运行锁定的原始仓库并保存 `reference_metrics.json`，然后才运行 DeepInv；最后每个 setting 都生成包含两边独立 PSNR/SSIM/LPIPS、差值及 tensor/trajectory 误差的 `comparison.json`，以及五图 `visualization.png`。论文表格是 100 图集合指标，五图运行不计算 FID，也不把五图均值标记为论文复现通过。

五图四组 setting 已于 run `gaussian-deblur-20260904T161716Z` 全部通过；固定结果摘要与 artifact SHA256 见 [`certifications/gaussian_deblur_ffhq5_v1.json`](certifications/gaussian_deblur_ffhq5_v1.json)。
