# DiffPIR alignment

论文四个退化族、20/100 NFE 超参数及官方 demo 差异见
[`PAPER_BENCHMARK_SETTINGS.zh-CN.md`](PAPER_BENCHMARK_SETTINGS.zh-CN.md)。

This directory aligns DeepInv's `DiffPIR` with the official DiffPIR
inpainting, deblurring and super-resolution paths at commit
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

## Gaussian deblur（已完成）

以下脚本固定使用 `DiffPIR/testsets/demo_test` 的五张 PNG、论文的 61×61 / std=3.0 Gaussian kernel，并覆盖有/无测量噪声及 20/100 NFE 四组论文参数：

```bash
bash reproduction/diffpir/reproduce_gaussian_deblur.sh
```

脚本不接受参数。它先在四张同型号 GPU 上运行锁定的原始仓库并保存 `reference_metrics.json`，然后才运行 DeepInv；最后每个 setting 都生成包含两边独立 PSNR/SSIM/LPIPS、差值及 tensor/trajectory 误差的 `comparison.json`，以及五图 `visualization.png`。论文表格是 100 图集合指标，五图运行不计算 FID，也不把五图均值标记为论文复现通过。

五图四组 setting 已于 run `gaussian-deblur-20260904T161716Z` 全部通过；固定结果摘要与 artifact SHA256 见 [`certifications/gaussian_deblur_ffhq5_v1.json`](certifications/gaussian_deblur_ffhq5_v1.json)。

## Motion deblur（已完成）

以下脚本同样固定五张 `demo_test` 图片，使用 61×61、intensity=0.5 的逐图 motion kernel，并覆盖有/无测量噪声及 20/100 NFE：

```bash
bash reproduction/diffpir/reproduce_motion_deblur.sh
```

脚本固定并校验外部 `motionblur` commit 与源码 SHA256；每张图最终使用的第二次生成 kernel 会直接保存到 `.pt`。执行顺序和 Gaussian deblur 相同，且不接受命令行参数。

五图四组 setting 已于 run `motion-deblur-20260905T140737Z` 全部通过；固定结果摘要、外部 kernel 依赖版本与 artifact SHA256 见 [`certifications/motion_deblur_ffhq5_v1.json`](certifications/motion_deblur_ffhq5_v1.json)。

## Bicubic SR ×4（已完成）

以下脚本固定同样五张图片和四组有噪/无噪 × 20/100 NFE 论文参数。`v2` 已修复初始噪声与第一步 transition noise 意外相同的问题；原始仓库五图 sanity gate 已通过。脚本依次生成 fixture、运行官方仓库、记录官方指标、运行 DeepInv、比较并可视化：

```bash
bash reproduction/diffpir/reproduce_sr4.sh
```

官方 fixture 与 reference 使用 `/mnt/afs/L202500464/DiffPIR/.venv`，DeepInv、统一指标和可视化使用本仓库 `.venv`；两边均通过个人目录的 uv wrapper 启动，共享下载缓存但不共享环境。脚本使用带 `_separate_uv_v1` 后缀的新 fixture，避免覆盖早期同环境产物。DeepInv 需要已安装 `reproduction` extra；如需从锁文件重建，在仓库根目录执行 `/mnt/afs/L202500464/uv-env-tool.sh --proxy off sync --locked --extra reproduction`。

官方测量为带抗混叠的 MATLAB 风格 bicubic resize，solver 为官方 25×25 MAT kernel 的圆周卷积加抽取；两者不完全等价，详细差异见论文配置文档中的 SR 小节。两边读取相同 `.pt` 测量、kernel、OpenCV 上采样初始图和全部随机量。SR PSNR/SSIM 裁边 4，LPIPS 不裁边，全部口径记录在 `comparison.json.metric_protocol`；可视化重用独立的 `visualize_deblur.py`。

参数、实际 timestep、原始仓库五图指标和门禁结论见 [`reports/sr4_reference_gate_20260905.zh-CN.md`](reports/sr4_reference_gate_20260905.zh-CN.md)。五图四组 setting 已于 run `sr4-20260905T173018Z` 在两个独立 uv 环境中全部通过；固定结果、环境锁哈希与 artifact SHA256 见 [`certifications/sr4_ffhq5_v2.json`](certifications/sr4_ffhq5_v2.json)。全项目结果及 PyTorch 数值栈归因见 [`../FINAL_ALIGNMENT_SUMMARY.zh-CN.md`](../FINAL_ALIGNMENT_SUMMARY.zh-CN.md)。
