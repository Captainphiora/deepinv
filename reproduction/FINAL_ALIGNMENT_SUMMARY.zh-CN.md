# DPS、DSG 与 DiffPIR 跨仓库对齐总报告

更新日期：2026-09-05。本文是当前阶段的交接入口；数值以 certification 和本地
`comparison.json` 为准，表中只做便于阅读的舍入。

## 1. 结论与边界

- DeepInv 的 DPS、DSG、DiffPIR inpainting、DiffPIR Gaussian deblur、motion deblur
  和 bicubic SR ×4 均已在固定 fixture 上通过原始仓库行为对齐。
- DiffPIR 的 Gaussian、motion 和 SR 各覆盖五张 `DiffPIR/testsets/demo_test` 图片、
  有噪/无噪与 20/100 NFE 四组 setting；不是论文 100 图均值复现，因此论文列只作
  参考，状态是 `NOT_COMPARABLE`。
- DiffPIR 论文中的 box/random inpainting × 20/100 NFE 四组 setting 尚未运行。
  已通过的 inpainting 是官方 demo 的 random/NFE=20/`lambda=1`，而论文对应值是
  `lambda=3`，两者不能混称。
- SR 已在两个仓库各自独立的 uv 环境中正式通过。较早的 DPS、DSG、DiffPIR
  inpainting/Gaussian/motion 认证是在同一个 DeepInv 运行环境中加载两份源码完成的；
  它们证明实现行为一致，但若把“独立环境”也作为新验收条件，需要按新工作流重跑。
- 所有认证比较的对象是 GPU 上的原始实现与 DeepInv 实现；不做 CPU/GPU 一致性验收。

## 2. Schedule 与 timestep

`noise_schedule="linear"` 和 `skip_type="quad"` 不是二选一：前者定义 1000 个训练
噪声级的 beta，后者从这 1000 个级别中选择推理子序列。

| 算法 | 训练 noise schedule | 推理 timestep 选择 | NFE / eta | 实际 model timestep |
| --- | --- | --- | --- | --- |
| DPS | linear beta，`1e-4 → 2e-2` | guided-diffusion `timestep_respacing=100`，等间隔 round | 100 / 0 | 从 999 反向到 0；完整 100 项在认证输出 `.pt` 中 |
| DSG | linear beta，`1e-4 → 2e-2` | 与 DPS 相同的 `timestep_respacing=100` | 100 / 1 | 与 DPS 完全相同 |
| DiffPIR | linear beta，`1e-4 → 2e-2` | 官方 `quad`：`int(sqrt(linspace(0,1000²,NFE)))`，末项减 1，再以 `999-seq` 送入模型 | 20 或 100 / 0 | 20 项见下；100 项见论文 setting 文档 |

DPS/DSG 的 100 项由 `round(k*999/99)` 后反向得到；开头为
`[999,989,979,969,959,949,938,...]`，结尾为
`[...,61,50,40,30,20,10,0]`。DiffPIR 20 项为：

```text
[999,770,675,602,541,487,438,393,351,311,
 274,239,205,172,141,111,82,54,26,0]
```

DiffPIR 100 项完整列表、论文与原 DDIM `quad` 公式的差异见
[`PAPER_BENCHMARK_SETTINGS.zh-CN.md`](diffpir/PAPER_BENCHMARK_SETTINGS.zh-CN.md)。

## 3. DPS 与 DSG：inpainting 三图

两项都使用 FFHQ 256、相同三张输入、固定 mask/measurement/初始点，测量噪声
`sigma_y=0.05`，并保存完整 timestep-to-noise-level schedule；DSG setting 还显式保留
标量 `noise_level=0.05`。论文指标未写入 setting，因此这里仅认证官方代码行为。

| 算法与 setting | 算法参数 | 原始仓库 PSNR / SSIM / LPIPS | DeepInv PSNR / SSIM / LPIPS | 最大 raw/trajectory MAE | 结果 |
| --- | --- | --- | --- | --- | --- |
| DPS `ddim100_eta0` | scale=0.5 | 18.128235 / 0.440499 / 0.476956 | 完全相同 | 0 / 0 | PASS |
| DSG `ddim100_eta1` | guidance=0.2, interval=1 | 35.338575 / 0.947389 / 0.101893 | 完全相同 | 0 / 0 | PASS |

DPS 第三张的 PSNR 只有 6.937231 dB，但原始仓库和 DeepInv 逐元素相同，所以它是
输入/初始点层面的质量问题，不是 DeepInv 算法偏差。单独更换第三张 `x_init` 的诊断
也保持两边逐元素一致；该诊断不替代 canonical 三图认证。

认证：[`DPS ddim100`](dps/certifications/ddim100_fixed_v1.json)、
[`DSG ddim100`](dsg/certifications/ddim100_eta1_fixed_v1.json)。DPS 的
`ffhq256_inpainting_ddpm1000_v1` 目前只有 setting，尚无正式 certification。两个 DPS
v1 setting 早于当前字段规范，保存了完整 `noise_levels` 张量但没有 task 级标量
`noise_level`；setting 不原地改写，新增 DPS setting 时必须显式补上该字段。

## 4. DiffPIR：inpainting demo 三图

固定 random 50% mask、无测量噪声、algorithm sigma=0.001、NFE=20、linear beta +
quad timestep、`lambda=1`、`zeta=1`、`eta=0`。

| 原始仓库 PSNR / SSIM / LPIPS | DeepInv PSNR / SSIM / LPIPS | 均值绝对差 PSNR / SSIM / LPIPS | 最大 final / trajectory MAE | 结果 |
| --- | --- | --- | --- | --- |
| 32.827609 / 0.938824 / 0.122921 | 32.827609 / 0.938824 / 0.122922 | 7.95e-8 / 7.95e-8 / 1.06e-6 | 5.57e-8 / 1.48e-7 | PASS |

认证：[`quad20_inpainting_fixed_v1.json`](diffpir/certifications/quad20_inpainting_fixed_v1.json)。
论文 random inpainting 使用 `(lambda,zeta)=(3,1)`/20 和 `(7,1)`/100，论文 100 图
结果分别为 PSNR/FID/LPIPS `34.03/30.81/0.116` 与 `36.17/13.68/0.066`；本项目尚未
运行这两组。box inpainting 20/100 也尚未运行。

## 5. DiffPIR：Gaussian deblur 五图

退化固定为 61×61 Gaussian、std=3.0、circular。PSNR/SSIM/LPIPS 均为本项目统一的
五图浮点口径；论文是 100 图且未报告 SSIM。

| `sigma_y` | NFE | `(lambda,zeta)` | 原始仓库 P / S / L | DeepInv P / S / L | 均值绝对差 P / S / L | 论文 P / FID / L |
| ---: | ---: | --- | --- | --- | --- | --- |
| 0.05 | 20 | (8,0.5) | 25.602561 / .749519 / .258925 | 25.602563 / .749519 / .258925 | 1.17e-6 / 1.55e-7 / 4.47e-7 | 未报告 |
| 0.05 | 100 | (7,0.3) | 26.063659 / .790042 / .222210 | 26.063659 / .790042 / .222210 | 2.45e-7 / 1.19e-8 / 1.79e-8 | 27.36 / 59.65 / .236 |
| 0 | 20 | (15,0.5) | 28.731286 / .853215 / .186139 | 28.731282 / .853214 / .186138 | 4.63e-6 / 1.79e-7 / 1.05e-6 | 30.74 / 46.64 / .170 |
| 0 | 100 | (12,0.4) | 29.108017 / .878441 / .156391 | 29.108017 / .878441 / .156388 | 2.51e-7 / 4.77e-8 / 3.26e-6 | 31.00 / 39.27 / .152 |

四组均 PASS，认证见
[`gaussian_deblur_ffhq5_v1.json`](diffpir/certifications/gaussian_deblur_ffhq5_v1.json)。

## 6. DiffPIR：motion deblur 五图

退化固定为 61×61、intensity=0.5；每张图保存官方 `case_index*10` 随机流第二次
`Kernel` 构造产生的实际 kernel。

| `sigma_y` | NFE | `(lambda,zeta)` | 原始仓库 P / S / L | DeepInv P / S / L | 均值绝对差 P / S / L | 论文 P / FID / L |
| ---: | ---: | --- | --- | --- | --- | --- |
| 0.05 | 20 | (7,0.8) | 26.629353 / .795462 / .242590 | 26.629353 / .795462 / .242590 | 4.62e-8 / 1.19e-8 / 6.26e-7 | 未报告 |
| 0.05 | 100 | (7,0.4) | 26.121222 / .786616 / .231037 | 26.121222 / .786616 / .231037 | 2.46e-7 / 1.07e-7 / 5.42e-7 | 26.57 / 65.78 / .255 |
| 0 | 20 | (25,1.0) | 35.873698 / .939276 / .114758 | 35.873693 / .939276 / .114758 | 4.60e-6 / 1.19e-8 / 3.71e-7 | 37.03 / 20.11 / .084 |
| 0 | 100 | (7,0.9) | 36.427253 / .949517 / .091620 | 36.427250 / .949517 / .091621 | 3.55e-6 / 2.38e-8 / 8.17e-7 | 37.53 / 11.54 / .064 |

四组均 PASS，认证见
[`motion_deblur_ffhq5_v1.json`](diffpir/certifications/motion_deblur_ffhq5_v1.json)。

## 7. DiffPIR：bicubic SR ×4 五图

以下是 run `sr4-20260905T173018Z` 的独立 uv 环境正式结果。PSNR/SSIM 对浮点 RGB
裁边 4，LPIPS 用完整图像；论文 PSNR 还涉及官方 uint8/Y-channel 口径，且数据集是
100 图，不能直接用五图差值判断论文复现成败。

| `sigma_y` | NFE | `(lambda,zeta)` | 原始仓库 P / S / L | DeepInv P / S / L | 均值绝对差 P / S / L | 论文 P / FID / L |
| ---: | ---: | --- | --- | --- | --- | --- |
| 0.05 | 20 | (8,0.4) | 25.352239 / .752174 / .276075 | 25.352240 / .752174 / .276076 | 1.39e-7 / 3.58e-8 / 8.79e-7 | 未报告 |
| 0.05 | 100 | (8,0.2) | 25.127537 / .752606 / .257548 | 25.127537 / .752606 / .257548 | 6.42e-8 / 3.58e-8 / 8.73e-7 | 26.64 / 65.77 / .260 |
| 0 | 20 | (9,0.2) | 26.979547 / .836770 / .195685 | 26.979547 / .836770 / .195685 | 7.43e-8 / 3.58e-8 / 6.71e-7 | 29.17 / 58.02 / .187 |
| 0 | 100 | (6,0.3) | 27.528535 / .853478 / .173950 | 27.528536 / .853478 / .173950 | 5.00e-7 / 0 / 2.62e-7 | 29.52 / 47.80 / .174 |

最坏 final MAE 为 `5.34e-7`，最坏 trajectory MAE 为 `7.29e-6`，四组均 PASS。
认证见 [`sr4_ffhq5_v2.json`](diffpir/certifications/sr4_ffhq5_v2.json)。

## 8. 退化算子与闭式解对应表

统一定义先退化、后加测量噪声：`y=A(x)+n`。DeepInv 的 `physics.A(x)` 只执行 A，
`physics(x)` 才执行 noise/sensor。完整的转置、伴随、伪逆、近端和 CT 说明见
[`LINEAR_DEGRADATION_OPERATORS.zh-CN.md`](LINEAR_DEGRADATION_OPERATORS.zh-CN.md)。

| 任务 | 原始仓库行为 | DeepInv 对齐实现 | `A*` / 数据闭式解 | 判断 |
| --- | --- | --- | --- | --- |
| DPS/DSG inpainting | `A(x)=M⊙x`，固定 mask；DPS/DSG 用 `A*(Ax-y)` 型梯度 | 复用 `deepinv.physics.Inpainting` | 二值 mask 下 `A*=A=M`，`A†=M` | 数学和离散实现相同；fixture 冻结 mask 后无需兼容层 |
| DiffPIR inpainting | 同为 `M⊙x`，官方 NumPy 精确生成 50% mask | 复用 `Inpainting`，不复用其随机 mask 生成器 | `prox(z,y,γ)`：观测处 `(z+γy)/(1+γ)`，缺失处为 z | 算子相同，RNG/缺失数量规则不同，所以保存实际 mask |
| Gaussian/motion deblur | `scipy.ndimage.convolve(mode="wrap")` 生成官方 measurement；FFT `data_solution` | 复用 `BlurFFT` 的 A/A*，子类只覆盖 `prox_l2`，用纯 Torch 保留官方 float32 求值顺序 | `A*` 为频域乘 `conj(H)`；`x=z+F⁻¹[conj(H)(Fy-HFz)/(|H|²+ρ)]`，`γ=1/ρ` | 数学目标相同；DeepInv 原生化简更稳定，官方顺序更适合逐数值复现 |
| DiffPIR SR ×4 measurement | 官方 MATLAB 风格 bicubic、antialias、symmetric boundary | fixture 直接保存官方 measurement；`DownsamplingMatlab` 仅用于算子诊断 | measurement 与 solver 所假设的 A 不完全相同 | 官方历史实现不统一；不能用 solver A 重生成 y |
| DiffPIR SR ×4 solver | 官方 25×25 MAT kernel 圆周卷积，再从左上角抽取 | 复用 `Downsampling(filter=...,factor=4,padding="circular")` 的 A/A*，覆盖 `prox_l2` 为官方 FFT aliasing 顺序 | `A*=H* S*`；`S*` 是相位 0 零插值；闭式解按 factor² 频谱块求均值 | DeepInv 原生接口更统一；官方兼容 prox 用于精度认证 |
| CT（目录规范，尚未做本轮算法认证） | 取决于参考仓库的 Radon/backend/geometry | `Tomography` 或 `TomographyWithAstra` | `A*` 是未滤波反投影；FBP/FDK 是近似逆，不是 A* 或严格 `A†` | 必须冻结 angles、geometry、spacing、normalize 和 backend 后再对齐 |

“哪种更好”取决于目的：后续新算法与消融优先使用 DeepInv 原生 physics，因为接口统一、
可做伴随测试、边界和近端语义明确；复现官方 DiffPIR 时保留官方求值顺序，因为早期
很小的 `rho` 会放大 FFT 浮点舍入差。当前兼容层只在 reproduction runner 中，不把
这段历史数值行为扩散到 DeepInv 通用算子。

## 9. PyTorch 版本问题的受控归因

同一 SR fixture、case、GPU、源码提交、参数和随机 tape 下做了三组实验：

| 实验 | reference / DeepInv 环境 | final MAE | 第一个 condition step MAE | PSNR 差 | 结果 |
| --- | --- | ---: | ---: | ---: | --- |
| A | 独立 prefix；Torch 2.8+cu128 / 2.14+cu130 | 4.53e-3 | 4.86e-1 | 0.1218 dB | FAIL |
| B | 共享 DeepInv prefix；两边均 2.14+cu130 | 4.39e-7 | 2.51e-7 | 1.44e-6 dB | PASS |
| C | 独立 prefix；两边均 2.14+cu130，NumPy 仍为 1.26/2.2 | 4.39e-7 | 2.51e-7 | 1.44e-6 dB | PASS |

因此已经排除“`.venv` 路径不同”和“NumPy 版本不同”是这次 SR 失败的原因；受控结果
把原因定位到关键 Torch/CUDA 数值栈。更精确地说，当前证据不能继续区分是 PyTorch
自身、其绑定的 cuFFT/cuBLAS，还是 CUDA/cuDNN 版本中的哪一个组件；所以报告结论是
“Torch 2.8+cu128 与 Torch 2.14+cu130 数值栈不同导致”，而不是武断写成纯 Python
层 PyTorch 算法错误。最可能的放大点是官方 SR float32 FFT 闭式解在极小 `rho` 下对
舍入误差敏感。

DiffPIR 环境已通过 `uv add torch==2.14.0 torchvision==0.29.0` 升级，`torchaudio`
因仓库源码未使用且不存在匹配的 2.14 版本而通过 `uv remove` 删除。环境仍独立：

| 仓库 | uv project / prefix | Torch | NumPy | lock |
| --- | --- | --- | --- | --- |
| DiffPIR | `/mnt/afs/L202500464/DiffPIR` / `.venv` | 2.14.0+cu130 | 1.26.4 | `8c89dc…` |
| DeepInv | `/mnt/afs/L202500464/deepinv` / `.venv` | 2.14.0+cu130 | 2.2.6 | `6a80b8…` |

两边可共享 `/mnt/afs/L202500464/.cache/uv` 的下载内容，但不会共享安装前缀。升级前后
缓存只增加约 100 KiB，说明大型 Torch/CUDA wheel 已从现有缓存复用。

## 10. SR 初始点复核

官方 task-specific SR 确实不是直接用纯 `N(0,I)`：先用 OpenCV `INTER_CUBIC` 把
64×64 measurement 放大到 256×256，再按 `t=999` 的 VP 公式混入固定初始噪声。
当前 reference 与 DeepInv 都读取同一 `x_init.pt`，已经对齐这一特殊取法。

早期 SR noiseless/20 的 19.5041 dB 异常不是初始化公式不同，而是旧 fixture 把
`x_init` 噪声和第一步 transition noise 都用 seed 42 重新生成，导致两张噪声逐元素
相同。v2 改成独立 seed 42/43 后，官方五图均值恢复为 26.9795 dB，并完成跨仓库
PASS。它仍低于论文 100 图的 29.17 dB，但五图集合和指标口径不同，不能据此继续
归咎于初始点。若以后做论文指标复现，应先固定论文 100 图及官方 uint8/Y-channel
口径，再决定是否增加“纯高斯初值 vs 官方 measurement-based 初值”的独立消融。

## 11. 复现入口与剩余工作

无需传参数的入口：

```bash
bash reproduction/dps/reproduce.sh
bash reproduction/dsg/reproduce.sh
bash reproduction/diffpir/reproduce.sh
bash reproduction/diffpir/reproduce_gaussian_deblur.sh
bash reproduction/diffpir/reproduce_motion_deblur.sh
bash reproduction/diffpir/reproduce_sr4.sh
```

可视化由独立 Python 文件完成：inpainting 使用
`reproduction/visualize_inpainting.py`，deblur/SR 使用
`reproduction/diffpir/visualize_deblur.py`。正式张量真值是 CPU `.pt`，PNG 只作预览。

剩余事项按优先级为：

1. 运行 DiffPIR 论文 box/random inpainting × 20/100 四组 setting；
2. 若新验收要求追溯到历史结果，使用各原始仓库自己的 uv 环境重跑 DPS、DSG、
   DiffPIR inpainting/Gaussian/motion；
3. 需要论文数值结论时再做 100 图和论文指标口径，当前用户已明确不做全量验证；
4. CT 仅完成算子规范，尚未进入具体算法跨仓库认证。
