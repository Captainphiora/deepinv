# DiffPIR SR ×4 原始仓库论文门禁报告（2026-09-05）

## 结论

原始仓库五图 sanity 门禁：**PASS**。DeepInv 跨仓库门禁：**WAITING_CONFIRMATION**。

`v1` 的 noiseless/20 异常已经定位为 fixture 随机流错误：初始噪声和第一步
transition noise 由两个同 seed 生成器从相同状态开始，实际张量逐元素相同。`v2`
改用固定但独立的 seed 42/43 后，五图 PSNR 从 19.5041 恢复到 26.9796 dB，LPIPS
从 0.50184 恢复到 0.19569，灾难性退化消失。按要求这里只使用五张 `demo_test`
图片，仍不能与论文 100 图集合指标作严格等价比较；尚未启动 DeepInv 正式运行。

## 参数与来源审计

论文来源为 [正文与附录](https://arxiv.org/abs/2305.08995)，官方实现锁定
[commit `2a989812`](https://github.com/yuanzhi-zhu/DiffPIR/tree/2a9898129a1b274131b98746e5b364bc20adc1e1)。

| 字段 | 论文证据 | 官方代码证据 | 本次执行值 |
| --- | --- | --- | --- |
| 任务 | 正文：bicubic SR ×4 | `main_ddpir_sisr.py`：`sf=4`, `sr_mode='blur'` | bicubic SR ×4 |
| beta/noise schedule | 正文：相同 linear noise schedule | `beta_start=0.0001`, `beta_end=0.02`, 1000 步 | linear，0.0001→0.02，1000 步 |
| timestep subsequence | 正文只说明采用 DDIM quadratic sequence | DiffPIR 为 `sqrt(linspace(0,1000**2,NFE))`；DDIM 原仓库公式不同 | 对齐锁定 DiffPIR commit，见下方实际列表 |
| model output | DiffPIR 预测 clean image | `model_output_type='pred_xstart'` | epsilon checkpoint，经模型得到 clipped `pred_xstart` |
| DDIM/随机项 | 论文公式引入 `zeta` | `eta=0`，transition 使用 `zeta` | `eta=0`；固定且相互独立的 initial/transition noise tape |
| noisy/20 | 附录超参数表 | 官方默认/扫描不是论文单一 setting | `sigma_y=0.05`, `lambda=8`, `zeta=0.4` |
| noisy/100 | 附录超参数表 | 官方默认/扫描不是论文单一 setting | `sigma_y=0.05`, `lambda=8`, `zeta=0.2` |
| noiseless/20 | 附录超参数表 | 需覆盖官方默认值 | `sigma_y=0`, internal `sigma=0.001`, `lambda=9`, `zeta=0.2` |
| noiseless/100 | 附录超参数表 | 需覆盖官方默认值 | `sigma_y=0`, internal `sigma=0.001`, `lambda=6`, `zeta=0.3` |
| proximal | 附录给出近似 bicubic kernel 闭式解 | 25×25 MAT kernel、circular blur 后左上抽取 | 保留官方行为；不等同于测量 resize |
| 指标 | 100 图 PSNR/FID/LPIPS | SR PSNR crop=4，官方 PSNR 经 uint8 | 本报告五图 float RGB PSNR/SSIM crop=4，LPIPS 全图；不算 FID |

checkpoint SHA256：
`81d535743156ec6be34d8668e6920da94f0614074d7793a16c8fa9e306237faa`。
MAT kernel 文件 SHA256：
`6be1ba76a6d6e8bb7fb3e94f109043c7ec250b547ced6c8a31b8cb38bcebd5d5`。

## 实际时间步

linear 指完整 1000 步 beta schedule；quad 指从中选取推理 timestep，二者同时成立。
论文没有给出整数抽取公式，因此这里记录的是锁定 DiffPIR 发布代码的精确行为，不把
它误写成论文唯一公式。论文所引用的 DDIM 原实现使用
`linspace(0, sqrt(0.8*T), NFE)**2`；该差异已单独诊断，未改正式 setting。

NFE=20 实际 model timestep：

```text
[999, 770, 675, 602, 541, 487, 438, 393, 351, 311,
 274, 239, 205, 172, 141, 111, 82, 54, 26, 0]
```

NFE=100 的完整列表记录在
[`PAPER_BENCHMARK_SETTINGS.zh-CN.md`](../PAPER_BENCHMARK_SETTINGS.zh-CN.md#4-20100-nfe-的精确-sampling-schedule)，
首尾为 `999→0`，共 100 个 model timestep；fixture 的 `schedule.pt` 保存实际整数张量，
manifest 保存 tensor SHA256。

## `v1` 原始仓库结果（随机流无效，保留作审计）

输入为 `69037.png`、`69133.png`、`69367.png`、`69887.png`、`69929.png`；run ID 为
`sr4-20260905T151214Z`。四个 run manifest 均只有 `reference`，没有 DeepInv 输出。

| setting | 原始仓库五图 PSNR | SSIM | LPIPS | 论文 100 图 PSNR/LPIPS | 判定 |
| --- | ---: | ---: | ---: | ---: | --- |
| noisy, 20 NFE, `(8,0.4)` | 24.8084 | 0.71036 | 0.29420 | 未报告 | `NOT_COMPARABLE` |
| noisy, 100 NFE, `(8,0.2)` | 25.2035 | 0.75966 | 0.25193 | 26.64 / 0.260 | `NOT_COMPARABLE` |
| noiseless, 20 NFE, `(9,0.2)` | 19.5041 | 0.38863 | 0.50184 | 29.17 / 0.187 | `BLOCKED`：明显退化 |
| noiseless, 100 NFE, `(6,0.3)` | 27.7055 | 0.85548 | 0.17658 | 29.52 / 0.174 | `NOT_COMPARABLE` |

结果文件位于：

```text
reproduction/artifacts/runs/diffpir/<setting-id>/sr4-20260905T151214Z/
├── manifest.json
└── reference_metrics.json
```

四个 `reference_metrics.json` SHA256（按上表顺序）：

```text
b19555308d7f9ea380f5128f835a687d2158ceedd928f1b4588aafbc4ed986c5
5eac8f0262871f3af2f5aa40dd54e492ff3671d3c01e7d1964642671378f219f
3203509bc3dd439c2dcb62a254bf4420053499ae87df4d9ebbfaada7d2b8d49c
e07f9f167f8625f5539c5bbc53362d8379ffcfcb994221e038c8d9e09a5edd82
```

`v1` 只能证明在这组固定张量上的跨实现行为，不能用于论文指标 gate。原因不是“seed
没有固定”，而是两个本应独立的随机流被分别重置到同一个 seed；首个 transition
noise 与 initial noise 的 `torch.equal` 为 `True`，余弦相似度为 1。

## `v2` 原始仓库结果（独立随机流）

`v2` setting 声明
`stream_policy=independent_generators_with_distinct_seeds`，初始流 seed 为 42，transition
流 seed 为 43。四组 fixture 的五个 case 均验证
`initial_noise != transition_noise[0]`。其余参数、输入图片、测量、kernel、checkpoint、
lambda、zeta 和 timestep 完全不变。run ID 为 `sr4-20260905T161302Z`。

| setting | 原始仓库五图 PSNR | SSIM | LPIPS | 论文 100 图 PSNR/LPIPS | 判定 |
| --- | ---: | ---: | ---: | ---: | --- |
| noisy, 20 NFE, `(8,0.4)` | 25.3522 | 0.75217 | 0.27608 | 未报告 | `SANITY_PASS`; `NOT_COMPARABLE` |
| noisy, 100 NFE, `(8,0.2)` | 25.1275 | 0.75261 | 0.25755 | 26.64 / 0.260 | `SANITY_PASS`; `NOT_COMPARABLE` |
| noiseless, 20 NFE, `(9,0.2)` | 26.9796 | 0.83677 | 0.19569 | 29.17 / 0.187 | `SANITY_PASS`; `NOT_COMPARABLE` |
| noiseless, 100 NFE, `(6,0.3)` | 27.5285 | 0.85348 | 0.17395 | 29.52 / 0.174 | `SANITY_PASS`; `NOT_COMPARABLE` |

四个 `reference_metrics.json` SHA256（按上表顺序）：

```text
ccbd8039d13284e85c10a66a2a069eae88e07bc6ccaa7c5fd5bbdb4c1b2a295c
1b42beb7db3185b8146f4a9e16077f593681c7e945ca03c5931fc436232a4afa
7d58b69d89ee8ea39cf185081505353a44b755d703673a668a80363c31ae81a8
27a9011e4340a06c4eca518c2d412d4dfcd2bef6fdf5d229de940c6b17a8136f
```

## 根因验证与排除项

同一 `x_t` 和 timestep 下，reference harness 的
`p_mean_variance(...)["pred_xstart"]` 与官方
`utils_model.model_fn(..., model_out_type="pred_xstart")` 逐元素相同：`max_abs=0`。
因此 harness 没有改变官方 denoiser 语义。

官方测量是 symmetric-boundary antialiased bicubic resize，而 proximal 假设 circular
25×25 kernel blur/decimation；两者并不完全相同。官方 SR 闭式解在 noiseless 极小
`rho` 下还存在 float32 数值消减。一次非正式的数学等价稳定求值诊断没有改善
noiseless/20 NFE（五图 PSNR 18.1532），所以它被排除在正式 setting 和代码之外，
不能冒充官方实现或论文结果。

另外三项首图只读诊断均未改善：历史 `sigma` 下限 0.01 得到 17.7501 dB；让测量与
MAT solver 算子完全一致得到 17.0050 dB；按 DDIM 原仓库的 quadratic 公式运行得到
15.8921 dB。这些诊断不进入 setting，也没有修改 lambda/zeta 或正式算法。

当前停止点严格位于“原始仓库五图 sanity gate”之后、“DeepInv 跨仓库 gate”之前。
若用户确认接受五图 sanity gate（而非要求补跑论文 100 图集合），下一步运行四组 `v2`
DeepInv 输出并生成 `comparison.json`；验收目标是 DeepInv 对锁定 DiffPIR commit 的
张量/轨迹/指标对齐，不声称五图均值等于论文 100 图指标。
