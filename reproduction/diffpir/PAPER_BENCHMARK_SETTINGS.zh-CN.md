# DiffPIR 论文基准完整 setting

> 状态：论文 setting 已核对；Gaussian deblur 与 motion deblur 已按用户指定的 `demo_test` 五图完成小规模对齐。论文数值只作 100 图基准参考，不与五图均值混为一谈。

## 1. 范围与口径

论文把问题归为三个图像恢复任务：超分辨率、去模糊和图像修复；本项目为了配置与结果管理，将去模糊拆成 Gaussian deblur 和 motion deblur，因此按下面四个退化族管理：

1. inpainting；
2. Gaussian deblur；
3. motion deblur；
4. bicubic super-resolution（×4）。

inpainting 又包含 box 和 random 两种 mask，所以论文附录的超参数表实际有五行实验 setting。这里的“四个任务”与论文所说的“三类任务”并不冲突。

本文区分两个目标，后续不能混写：

- **论文基准 setting**：用于复现论文表格，退化参数和每个 NFE 的 `lambda`、`zeta` 以论文正文及附录为准。
- **官方 demo setting**：用于逐元素/数值对齐官方仓库某个固定 commit 的实际代码路径。官方 demo 的若干默认值与论文表格并不完全相同。

若两者冲突，必须建立两个不同的 setting ID；不能修改一个 setting 后仍沿用原名称和认证结果。

## 2. 权威来源与版本锁定

- 论文：[Denoising Diffusion Models for Plug-and-Play Image Restoration](https://arxiv.org/abs/2305.08995)；退化定义见正文 Implementation Details，20/100 NFE 超参数见附录 Experimental Details。
- 官方实现：[yuanzhi-zhu/DiffPIR](https://github.com/yuanzhi-zhu/DiffPIR)，本项目核对并锁定 commit [`2a9898129a1b274131b98746e5b364bc20adc1e1`](https://github.com/yuanzhi-zhu/DiffPIR/tree/2a9898129a1b274131b98746e5b364bc20adc1e1)。
- 官方 motion kernel 依赖：[LeviBorodenko/motionblur](https://github.com/LeviBorodenko/motionblur)。DiffPIR README 要求另行下载，但没有锁定版本；本机当前参考副本为 commit [`578137887b2067c532bd4e422ff5afdcabfccebf`](https://github.com/LeviBorodenko/motionblur/tree/578137887b2067c532bd4e422ff5afdcabfccebf)。

对齐时还必须记录 diffusion checkpoint 的 SHA256，而不能只记录文件名。

## 3. 所有任务共用的 setting

| 项目 | 论文/官方实现取值 |
| --- | --- |
| 图像尺寸 | FFHQ 256×256；ImageNet 256×256 |
| 测试集 | 每个数据集 100 张 hold-out validation 图像 |
| FFHQ diffusion prior | `diffusion_ffhq_10m.pt`，来自论文所引用的 FFHQ 预训练模型 |
| ImageNet diffusion prior | `256x256_diffusion_uncond.pt`，OpenAI guided-diffusion ImageNet 模型 |
| 训练扩散步数 | `N = 1000` |
| beta schedule | linear，`beta_start = 0.0001`，`beta_end = 0.02`，官方代码用 `np.float32` 生成 beta |
| 采样步数/NFE | 分别评估 20 和 100；一次 denoiser 调用计一个 NFE |
| timestep 子序列 | `quad`，低噪声区域更密集，精确规则见下一节 |
| 模型输出使用方式 | 官方实现取 `pred_xstart`，内部模型 checkpoint 为 epsilon prediction + learned variance |
| 数据子问题 | `sub_1_analytic = true`；线性任务使用闭式解 |
| `iter_num_U` | 1 |
| `guidance_scale` | 1.0 |
| DDIM `eta` | 0.0；论文算法中的随机性由 `zeta` 控制 |
| `t_start` | 官方 runner 默认强制为 999 |
| `noise_model_t` | 官方 runner 默认强制为 0，即不跳过低于测量噪声的模型步骤 |
| 图像/测量范围 | 退化与指标语义为 `[0,1]`；送入 diffusion model 前映射到 `[-1,1]` |
| 数据模型 | `y = A(x) + n`；`n ~ N(0, sigma_y^2 I)`，噪声在退化之后加入 |
| 无噪声数值下限 | 官方实现使用 `sigma = max(0.001, sigma_y)` 计算 `rho_t`，所以 `sigma_y=0` 时算法内部仍取 `0.001` |

DiffPIR 每步的数据权重为：

```text
rho_t = lambda * sigma^2 / sigma_bar_t^2
sigma = max(0.001, sigma_y)
```

因此 `lambda` 越大并不等于更强的测量约束；在上述目标函数写法中，它会增大 proximal 项中靠近 diffusion 预测的权重。

## 4. 20/100 NFE 的精确 sampling schedule

两个字段不能混淆：`noise_schedule="linear"` 先定义完整 1000 步中每个 timestep 的 beta、`alpha_cumprod` 和等效噪声强度；`skip_type="quad"` 再决定推理时从这 1000 步中抽取哪 20 或 100 个点。前者定义完整噪声轨迹，后者只定义稀疏采样位置。

官方代码不是均匀抽取 timestep，而是先生成索引：

```python
seq = np.sqrt(np.linspace(0, 1000**2, nfe))
seq = [int(value) for value in seq]
seq[-1] -= 1
```

随后用 `sigmas[seq[i]]`，其中 `sigmas` 是按训练 timestep 反向排列的等效噪声表。因此实际传给 diffusion model 的 timestep 等价于 `999 - seq[i]`。

### NFE = 20

```text
seq index:
[0, 229, 324, 397, 458, 512, 561, 606, 648, 688,
 725, 760, 794, 827, 858, 888, 917, 945, 973, 999]

model timestep:
[999, 770, 675, 602, 541, 487, 438, 393, 351, 311,
 274, 239, 205, 172, 141, 111, 82, 54, 26, 0]
```

### NFE = 100

```text
seq index:
[0, 100, 142, 174, 201, 224, 246, 265, 284, 301,
 317, 333, 348, 362, 376, 389, 402, 414, 426, 438,
 449, 460, 471, 481, 492, 502, 512, 522, 531, 541,
 550, 559, 568, 577, 586, 594, 603, 611, 619, 627,
 635, 643, 651, 659, 666, 674, 681, 689, 696, 703,
 710, 717, 724, 731, 738, 745, 752, 758, 765, 771,
 778, 784, 791, 797, 804, 810, 816, 822, 828, 834,
 840, 846, 852, 858, 864, 870, 876, 881, 887, 893,
 898, 904, 910, 915, 921, 926, 932, 937, 942, 948,
 953, 958, 963, 969, 974, 979, 984, 989, 994, 999]

model timestep:
[999, 899, 857, 825, 798, 775, 753, 734, 715, 698,
 682, 666, 651, 637, 623, 610, 597, 585, 573, 561,
 550, 539, 528, 518, 507, 497, 487, 477, 468, 458,
 449, 440, 431, 422, 413, 405, 396, 388, 380, 372,
 364, 356, 348, 340, 333, 325, 318, 310, 303, 296,
 289, 282, 275, 268, 261, 254, 247, 241, 234, 228,
 221, 215, 208, 202, 195, 189, 183, 177, 171, 165,
 159, 153, 147, 141, 135, 129, 123, 118, 112, 106,
 101, 95, 89, 84, 78, 73, 67, 62, 57, 51,
 46, 41, 36, 30, 25, 20, 15, 10, 5, 0]
```

20/100 NFE 不只是同一算法多跑几步：任务对应的 `lambda` 和 `zeta` 也必须按附录分别取值。NFE=20 有 20 次模型调用、19 次随机 transition；NFE=100 有 100 次模型调用、99 次随机 transition。

## 5. 四个退化族的完整定义

| 退化族 | `A(x)` | 论文参数 | 测量噪声 |
| --- | --- | --- | --- |
| Inpainting—box | `M ⊙ x` | 256×256 图像中缺失一个 128×128 box | 只评估 `sigma_y=0` |
| Inpainting—random | `M ⊙ x` | 随机移除总像素的 50%；RGB 三通道共用同一空间 mask | 只评估 `sigma_y=0` |
| Gaussian deblur | `x ⊗ k`，逐通道相同 kernel，圆周卷积 | kernel 61×61，Gaussian std=3.0 | `sigma_y=0` 和 `0.05` |
| Motion deblur | `x ⊗ k`，逐通道相同 kernel，圆周卷积 | 随机 motion kernel 61×61，intensity=0.5；不同方法使用同一组 kernel | `sigma_y=0` 和 `0.05` |
| Bicubic SR | `x ↓_4^bicubic` | scale factor=4，bicubic downsampling | `sigma_y=0` 和 `0.05` |

噪声的等价代码尺度是：YAML 中 `12.75` 先除以 255 得到 `0.05`；官方 task runner 在 `[-1,1]` 尺度加入标准差 `2*sigma_y` 的 AWGN，再映射回 `[0,1]`，最终仍等价于在 `[0,1]` 尺度加入标准差 `sigma_y` 的噪声。

论文的实验矩阵为：

- noisy (`sigma_y=0.05`)：FFHQ 与 ImageNet 上的 Gaussian deblur、motion deblur、×4 SR；
- noiseless (`sigma_y=0`)：FFHQ 上的两种 inpainting、两种 deblur、×4 SR；
- 论文没有给出 noisy inpainting，也没有在附录给出 ImageNet noiseless 的 `lambda/zeta`。

## 6. 论文附录中的任务超参数

### NFE = 20

| 任务 | FFHQ, `sigma_y=0.05` (`lambda`, `zeta`) | ImageNet, `sigma_y=0.05` (`lambda`, `zeta`) | FFHQ, `sigma_y=0` (`lambda`, `zeta`) |
| --- | --- | --- | --- |
| Inpaint box | — | — | (6.0, 1.0) |
| Inpaint random | — | — | (3.0, 1.0) |
| Deblur Gaussian | (8.0, 0.5) | (12.0, 0.9) | (15.0, 0.5) |
| Deblur motion | (7.0, 0.8) | (7.0, 1.0) | (25.0, 1.0) |
| SR ×4 | (8.0, 0.4) | (10.0, 0.5) | (9.0, 0.2) |

### NFE = 100

| 任务 | FFHQ, `sigma_y=0.05` (`lambda`, `zeta`) | ImageNet, `sigma_y=0.05` (`lambda`, `zeta`) | FFHQ, `sigma_y=0` (`lambda`, `zeta`) |
| --- | --- | --- | --- |
| Inpaint box | — | — | (6.0, 0.5) |
| Inpaint random | — | — | (7.0, 1.0) |
| Deblur Gaussian | (7.0, 0.3) | (8.0, 0.3) | (12.0, 0.4) |
| Deblur motion | (7.0, 0.4) | (8.0, 0.7) | (7.0, 0.9) |
| SR ×4 | (8.0, 0.2) | (9.0, 0.5) | (6.0, 0.3) |

`lambda`、`zeta` 必须与数据集、噪声、任务和 NFE 一起构成 setting ID，不能把其中任何一个当作全局默认值。

## 7. 论文报告的具体指标

论文只报告 PSNR、FID 和 LPIPS，没有报告 SSIM。以下均为每个数据集 **100 张**图像的集合结果。

### 7.1 Noisy，`sigma_y=0.05`，NFE=100

| 数据集 | 任务 | PSNR (dB) | FID | LPIPS |
| --- | --- | ---: | ---: | ---: |
| FFHQ | Gaussian deblur | 27.36 | 59.65 | 0.236 |
| FFHQ | motion deblur | 26.57 | 65.78 | 0.255 |
| FFHQ | SR ×4 | 26.64 | 65.77 | 0.260 |
| ImageNet | Gaussian deblur | 22.80 | 93.36 | 0.355 |
| ImageNet | motion deblur | 24.01 | 124.63 | 0.366 |
| ImageNet | SR ×4 | 23.18 | 106.32 | 0.371 |

论文附录给出了 noisy NFE=20 的超参数，但正文没有给出对应的 NFE=20 指标表，因此不能为该 setting 编造论文目标值。

### 7.2 FFHQ noiseless，`sigma_y=0`

| NFE | 任务 | PSNR (dB) | FID | LPIPS |
| ---: | --- | ---: | ---: | ---: |
| 20 | Inpaint box | — | 35.72 | 0.117 |
| 20 | Inpaint random | 34.03 | 30.81 | 0.116 |
| 20 | Gaussian deblur | 30.74 | 46.64 | 0.170 |
| 20 | motion deblur | 37.03 | 20.11 | 0.084 |
| 20 | SR ×4 | 29.17 | 58.02 | 0.187 |
| 100 | Inpaint box | — | 25.64 | 0.107 |
| 100 | Inpaint random | 36.17 | 13.68 | 0.066 |
| 100 | Gaussian deblur | 31.00 | 39.27 | 0.152 |
| 100 | motion deblur | 37.53 | 11.54 | 0.064 |
| 100 | SR ×4 | 29.52 | 47.80 | 0.174 |

box inpainting 的 PSNR 在论文中未报告，以“—”保留；不能当作 0。论文没有报告 noiseless ImageNet 指标。

本项目的五图运行额外计算 SSIM，并记录 reference/DeepInv 各自的逐图及均值 PSNR、SSIM、LPIPS。由于样本只有五张，不计算 FID，也不以五图均值是否接近上述 100 图均值作为通过条件。

## 8. 官方代码中必须保留或冻结的实现细节

### 8.1 Deblur 的边界与闭式解

论文假设圆周卷积，官方 task runner 用 `scipy.ndimage.convolve(..., mode="wrap")` 生成测量，并用 FFT 闭式 data solution。DeepInv 对齐时必须使用等价的 circular boundary，不能换成 zero padding 或 reflection padding。

官方脚本先对 uint8 图像卷积，再转换到 `[0,1]`。这会引入与“先转 float 再卷积”不同的量化行为。算法对齐阶段应把原始仓库生成的 `ground_truth`、`kernel`、`clean_measurement`、`measurement_noise` 和最终 `measurement` 冻结为 `.pt`，两边读取同一 tensor；算子实现正确性再单独测试。

DeepInv `BlurFFT.prox_l2` 与官方 FFT data solution 在数学上是同一个闭式解，但默认采用更稳定的化简计算式。官方代码在 float32 下使用“先相减、再除以 `rho`”的计算顺序；DiffPIR 早期 `rho` 极小时，两种求值顺序会出现可观的中间轨迹差异。为了认证原实现，DeepInv reproduction runner 使用纯 Torch 保留官方求值顺序；采样公式、目标函数和退化算子均未改变，也没有用 NumPy 重写算法。

### 8.2 Gaussian kernel：论文与 demo 不一致

论文基准明确写 `61×61, std=3.0`。但固定 commit 的 `main_ddpir_deblur.py` 和 `main_ddpir.py` 实际执行：

```python
np.random.seed(idx * 10)
kernel_std_i = 3.0 * abs(np.random.rand() * 2 + 1)
```

所以 demo 的逐图 Gaussian std 位于 `[3.0, 9.0)`，并非固定 3.0。这是两个不同实验，必须使用不同 setting：

- `paper_gaussian61_std3_*`：论文表格复现；
- `official_demo_gaussian61_randomstd_*`：固定 commit 代码级对齐。

### 8.3 Motion kernel 的依赖与 RNG

官方代码对第 `idx` 张图设置 NumPy seed `idx*10`，用 `kernel_size=61, intensity=0.5` 生成 motion kernel。`MotionBlurOperator` 构造过程中会调用外部 `Kernel` 两次，最终使用第二次生成的 kernel；只调用一次会得到不同结果。

由于官方 DiffPIR 仓库没有锁定 `motionblur` commit，严格复现时必须同时记录依赖 commit，并直接把每张图最终使用的 kernel tensor 存入 fixture。论文中的“same motion blur kernel”应理解为不同被比较方法使用同一组固定 kernel，而不是所有输入共用一个 kernel。

### 8.4 随机性与初始点

三个 task-specific runner 只显式固定部分 NumPy 随机数，没有统一固定 PyTorch/CUDA RNG；通用 `main_ddpir.py --opt ...` 才读取 YAML 的 `seed: 42` 并设置 Python、NumPy、Torch 和 CUDA seed。两条入口因此不是天然逐元素一致。

此外，论文算法伪代码写 `x_T ~ N(0,I)`，task-specific deblur runner 实际使用带测量项的初始化，并根据 `sigma_y` 做 `t_y` 修正。通用 runner 的 deblur 初始化也含测量项，但没有相同的 `t_y` 修正。后续认证应固定：

- `x_init.pt`；
- 每一步的 transition noise；
- measurement noise；
- mask 或 blur kernel。

这样比较的是两仓库算法更新是否一致，而不是比较两个 RNG 或两个退化数据生成器。

## 9. 官方 demo 默认值不能当成论文超参数

固定 commit 的代码存在以下行为：

| 入口 | 表面默认值 | 实际运行行为 | 与论文表格关系 |
| --- | --- | --- | --- |
| `main_ddpir_deblur.py` | noisy FFHQ、NFE=100、`lambda=1`、`zeta=0.1`、Gaussian | 末尾单元素循环实际调用 `lambda=7`、`zeta=0.3`；Gaussian std 仍按图随机 | `lambda/zeta` 恰好匹配 noisy FFHQ Gaussian NFE=100，但 kernel 不匹配论文 |
| `main_ddpir.py --opt configs/deblur.yaml` | YAML 同上，seed=42、batch=16 | 末尾同样把有效值变成 `lambda=7`、`zeta=0.3` | motion NFE=100 应为 `zeta=0.4`，不能只改 `blur_mode` |
| `main_ddpir_inpainting.py` | random mask、noiseless、NFE=20、`lambda=1`、`zeta=1` | 单一 setting | 论文 random/NFE=20 是 `lambda=3`、`zeta=1` |
| `main_ddpir_sisr.py` | noisy、NFE=100、`lambda=1`、`zeta=0.1` | 实际扫描 `lambda=2..12`，固定 `zeta=0.25` | 不是论文中的单一 setting |

结果目录名在参数循环之前生成，可能仍显示表面默认值而不是有效值。后续 manifest 必须记录实际传入每一步的参数，不能仅解析目录名。

当前已经认证的 DeepInv inpainting setting 是“官方 demo random-mask/NFE=20/`lambda=1`/`zeta=1`”代码路径，不是论文表格中的 random-mask/NFE=20/`lambda=3` setting；原认证不应改名冒充论文复现。

## 10. 指标口径

论文报告 PSNR、FID、LPIPS：

- deblur 和 inpainting 的 PSNR border=0；
- ×4 SR 的 PSNR crop border=4，官方脚本还额外计算 Y-channel PSNR；
- 官方 task runner 将输出转成 uint8 后计算 PSNR，而 LPIPS 使用浮点输出；
- FID 是 100 张图像集合级指标，五张图的小规模算法对齐不能与论文 FID 直接比较。

仓库间算法认证仍应额外报告相同 fixture 下 reference/DeepInv 各自的 PSNR、SSIM、LPIPS及其 delta，并报告重建 tensor 与 trajectory probe 的误差；这些是对齐指标，不应伪装成论文表格口径。

## 11. 当前逐任务执行方案

论文完整 deblur 矩阵共有 12 个 setting：

- FFHQ：2 种 blur × 2 种噪声 × 2 种 NFE，共 8 个；
- ImageNet：2 种 blur × noisy `sigma_y=0.05` × 2 种 NFE，共 4 个；
- ImageNet noiseless 没有论文超参数，不自行补值。

第一阶段只做 Gaussian deblur，固定读取 `DiffPIR/testsets/demo_test` 中按文件名排序的五张图：

`69037.png`、`69133.png`、`69367.png`、`69887.png`、`69929.png`。

运行四个 Gaussian setting：

1. `sigma_y=0.05`, NFE=20, (`lambda=8`, `zeta=0.5`)；
2. `sigma_y=0.05`, NFE=100, (`lambda=7`, `zeta=0.3`)；
3. `sigma_y=0`, NFE=20, (`lambda=15`, `zeta=0.5`)；
4. `sigma_y=0`, NFE=100, (`lambda=12`, `zeta=0.4`)。

每个 case 的 `.pt` fixture 至少保存 clean image、kernel、`A(x)`、noise、`y`、`x_init`、完整 transition-noise tape 与 timestep schedule。DeepInv 侧只用 Torch 执行算子与算法；原始仓库的 NumPy/SciPy 只用于产生官方参考 fixture 和输出。

顺序固定为：先运行锁定 commit 的原始仓库并写出 `reference_metrics.json`，再运行 DeepInv，最后生成 `comparison.json` 与 `visualization.png`。五图输入、measurement noise、`x_init`、transition-noise tape、kernel 和 schedule 均冻结为 `.pt`；每个文件及 tensor 内容均记录 SHA256。

第一阶段结束即停止，不自动继续 motion deblur、SR 或 inpainting 论文 setting，等待用户确认。

第二阶段 motion deblur 已完成：仍使用上述五张图，依次运行 noisy/noiseless × 20/100 NFE 四组 setting；motion kernel 固定为 61×61、intensity=0.5，并按官方 `case_index*10` 种子及两次 `Kernel` 构造顺序生成。四组比较均 PASS，证书见 [`certifications/motion_deblur_ffhq5_v1.json`](certifications/motion_deblur_ffhq5_v1.json)。当前停止，等待下一项确认。
