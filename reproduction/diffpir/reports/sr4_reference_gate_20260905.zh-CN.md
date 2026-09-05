# DiffPIR SR ×4 原始仓库论文门禁报告（2026-09-05）

## 结论

门禁状态：**BLOCKED**。

参数和时间步已经对齐论文，但本次只使用五张 `demo_test` 图片，不能与论文 100 图
集合指标作严格等价比较；同时 noiseless/20 NFE 的官方仓库输出出现明显退化，因此
不能继续宣称论文指标已复现，也没有启动 DeepInv 正式五图运行。

## 参数与来源审计

论文来源为 [正文与附录](https://arxiv.org/abs/2305.08995)，官方实现锁定
[commit `2a989812`](https://github.com/yuanzhi-zhu/DiffPIR/tree/2a9898129a1b274131b98746e5b364bc20adc1e1)。

| 字段 | 论文证据 | 官方代码证据 | 本次执行值 |
| --- | --- | --- | --- |
| 任务 | 正文：bicubic SR ×4 | `main_ddpir_sisr.py`：`sf=4`, `sr_mode='blur'` | bicubic SR ×4 |
| beta/noise schedule | 正文：相同 linear noise schedule | `beta_start=0.0001`, `beta_end=0.02`, 1000 步 | linear，0.0001→0.02，1000 步 |
| timestep subsequence | 正文 Accelerated Generation：采用 DDIM quadratic sequence | `sqrt(linspace(0,1000**2,NFE))`，末项减 1 | `skip_type=quad`，见下方实际列表 |
| model output | DiffPIR 预测 clean image | `model_output_type='pred_xstart'` | epsilon checkpoint，经模型得到 clipped `pred_xstart` |
| DDIM/随机项 | 论文公式引入 `zeta` | `eta=0`，transition 使用 `zeta` | `eta=0`，固定 transition-noise tape |
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

NFE=20 实际 model timestep：

```text
[999, 770, 675, 602, 541, 487, 438, 393, 351, 311,
 274, 239, 205, 172, 141, 111, 82, 54, 26, 0]
```

NFE=100 的完整列表记录在
[`PAPER_BENCHMARK_SETTINGS.zh-CN.md`](../PAPER_BENCHMARK_SETTINGS.zh-CN.md#4-20100-nfe-的精确-sampling-schedule)，
首尾为 `999→0`，共 100 个 model timestep；fixture 的 `schedule.pt` 保存实际整数张量，
manifest 保存 tensor SHA256。

## 原始仓库运行结果

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

## 已排查问题与停止点

官方测量是 symmetric-boundary antialiased bicubic resize，而 proximal 假设 circular
25×25 kernel blur/decimation；两者并不完全相同。官方 SR 闭式解在 noiseless 极小
`rho` 下还存在 float32 数值消减。一次非正式的数学等价稳定求值诊断没有改善
noiseless/20 NFE（五图 PSNR 18.1532），所以它被排除在正式 setting 和代码之外，
不能冒充官方实现或论文结果。

当前停止点严格位于“原始仓库论文 gate”之后、“DeepInv 跨仓库 gate”之前。后续只有
两条合法路径：继续查明论文 SR 未公开/不一致的执行细节；或由用户明确接受把目标改为
“对齐固定 commit 的真实可执行行为”，并用不同名称记录该结论。
