# 线性退化算子：实现、伴随与伪逆

本文档规定本项目在线性反问题中使用的数学记号、DeepInv 接口和对齐检查方法。
它描述的是**离散算子**；连续模型相同，不代表离散后的 padding、采样位置和归一化
也相同。当前内容按 DeepInv 提交 `49a21d13fa3e711f73979e6187f96bfc75f67a43`
核对。DeepInv 更新后，应重新检查本文链接的实现。

相关实现：[`forward.py`](../deepinv/physics/forward.py)、
[`inpainting.py`](../deepinv/physics/inpainting.py)、
[`blur.py`](../deepinv/physics/blur.py)、
[`tomography.py`](../deepinv/physics/tomography.py)。跨仓库实验的随机性与文件格式见
[`ALIGNMENT_WORKFLOW.zh-CN.md`](ALIGNMENT_WORKFLOW.zh-CN.md)。

## 1. 统一观测模型与 DeepInv 接口

对本项目中的线性退化，先定义确定性算子

$$
A:\mathcal X\rightarrow\mathcal Y,\qquad z=A x,
$$

再定义观测过程

$$
y=S\bigl(N(Ax)\bigr).
$$

其中 $N$ 是噪声模型，$S$ 是量化、饱和、裁剪等 sensor 非线性。没有显式配置
时，DeepInv 的 `noise_model` 和 `sensor_model` 都是恒等映射。最常见的高斯模型是

$$
y=Ax+n,\qquad n\sim\mathcal N(0,\sigma_y^2I_{\mathcal Y}).
$$

注意：噪声属于测量空间 $\mathcal Y$，所以必须先确定 `A(x)` 的输出表示和 shape，
才能解释 $I_{\mathcal Y}$ 的含义。

| 调用 | 含义 | 是否含噪声/sensor |
|---|---|---|
| `physics.A(x)` | $Ax$ | 否 |
| `physics(x)` | $S(N(Ax))$ | 是 |
| `physics.A_adjoint(y)` | $A^*y$ | 否 |
| `physics.A_dagger(y)` | $A^\dagger y$ 或其数值近似 | 否 |
| `physics.prox_l2(z, y, gamma)` | 二次数据项的近端映射 | 否 |

对齐退化算子本身时使用 `A`；生成完整观测时才使用 `physics(...)`。不要把
`physics(x)` 的随机输出误认为 `A(x)`。

## 2. 转置、伴随、逆和近端不是同一个操作

### 2.1 转置与伴随

实数矩阵的转置写作 $A^\top$。实数或复数 Hilbert 空间中的伴随写作 $A^*$，
由下式唯一确定：

$$
\langle Ax,y\rangle_{\mathcal Y}=\langle x,A^*y\rangle_{\mathcal X}.
$$

实数情况下 $A^*=A^\top$；复数情况下 $A^*=\overline A^\top$，不能漏掉共轭。
DeepInv 统一把它命名为 `A_adjoint`。伴随的方向与前向相反：

$$
A:\mathcal X\to\mathcal Y,
\qquad A^*:\mathcal Y\to\mathcal X.
$$

`A_adjoint` 不是“视觉上看起来像反向操作”即可。必须通过内积等式验证；例如普通
插值放大通常不是下采样的伴随，FBP 也不是 CT 前向投影的伴随。

### 2.2 逆与 Moore--Penrose 伪逆

只有方阵且满秩时才存在满足 $A^{-1}A=AA^{-1}=I$ 的双边逆。成像退化通常丢失
信息，因此一般不存在 $A^{-1}$，应讨论 Moore--Penrose 伪逆 $A^\dagger$：

$$
A^\dagger y
=\operatorname*{argmin}_{x\in\operatorname*{argmin}_u\|Au-y\|_2^2}\|x\|_2^2,
$$

存在多个最小二乘解时取欧氏范数最小者。它满足四个 Penrose 条件，但通常

$$
A^\dagger A\ne I.
$$

更准确地说，

$$
A^\dagger A=P_{(\ker A)^\perp},
\qquad
AA^\dagger=P_{\operatorname{range}(A)}.
$$

因此不能用 `A_dagger(A(x)) == x` 作为有零空间算子的通用测试。正确的基本检查是

$$
AA^\dagger Ax=Ax,
$$

并结合四个 Penrose 条件或与可信的线性代数基准比较。

DeepInv 中：

- `DecomposablePhysics` 用已知 SVD 计算闭式伪逆；小于等于 `1e-5` 的奇异值按零
  处理。
- 普通 `LinearPhysics.A_dagger` 默认通过迭代最小二乘求解，所以结果还依赖 solver、
  `max_iter`、`tol`、dtype 和停止误差；它不一定是机器精度下的精确伪逆。
- CT 的 `fbp=True` 返回的是稳定而快速的 FBP/FDK 近似，不是 Moore--Penrose
  伪逆。

### 2.3 数据一致性梯度与近端映射

对

$$
f(x)=\frac12\|Ax-y\|_2^2,
$$

梯度为

$$
\nabla f(x)=A^*(Ax-y).
$$

这也是扩散/flow inverse solver 中 `A_adjoint` 最常见的用途。DeepInv 的
`prox_l2(z, y, gamma)` 定义为

$$
\operatorname*{argmin}_x
\frac{\gamma}{2}\|Ax-y\|_2^2+\frac12\|x-z\|_2^2,
$$

其正规方程为

$$
(I+\gamma A^*A)x=z+\gamma A^*y.
$$

它是带正则的数据一致性求解，不等于 `A_dagger(y)`。若论文把数据项定义成
`rho * ||Ax-y||^2`、除以噪声方差，或没有 `1/2`，传入 DeepInv 的 `gamma` 也必须
相应换算。

## 3. 四类常用线性算子

### 3.1 Inpainting

DeepInv 使用与输入同 shape 的嵌入式表示：

$$
A=M=\operatorname{diag}(m),\qquad Ax=m\odot x,
$$

其中 $m_i=1$ 表示保留，$m_i=0$ 表示缺失。对复数或非二值 mask，

$$
A^*y=\overline m\odot y.
$$

对二值 mask：

$$
A^*=A,\qquad A^\dagger=A,\qquad A^*A=AA^*=M.
$$

伪逆只把已观测值放回原位置，缺失位置保持零；它不可能恢复缺失内容。对应近端为

$$
x_i=
\begin{cases}
(z_i+\gamma y_i)/(1+\gamma), & m_i=1,\\
z_i, & m_i=0.
\end{cases}
$$

也可以把 inpainting 写成紧凑选择算子
$P_\Omega:\mathbb R^n\to\mathbb R^{|\Omega|}$：
前向只输出观测元素，伴随 $P_\Omega^*$ 做零填充。它和 DeepInv 的 $M$ 表示相同
信息，但 measurement 的 shape 和噪声空间不同，跨仓库时不能直接混用。

DeepInv 特有注意事项：

- `mask` 的 float 值表示**保留概率**，随机生成的保留数量通常不是严格固定值；正式
  对齐应保存并传入同一个 mask tensor。
- `pixelwise=True` 会让同一像素的所有通道共同缺失；现成 tensor mask 则完全按其
  shape 和广播规则执行。
- `Inpainting.noise` 会把噪声结果再次投影到 mask。二值 mask 与加性高斯噪声下，
  `physics(x)` 是 $Mx+Mn$，缺失位置仍为零。若把测量空间定义成与图像等大的
  $\mathbb R^n$，则字面上的 $y=Mx+n$ 会在缺失位置也有噪声；若把测量空间
  定义成紧凑的 $\mathbb R^{|\Omega|}$，再零填充回图像，则 $Mx+Mn$ 是合理表示。
  必须按参考仓库的实际 measurement tensor 选择，不能只凭公式名称判断一致。

实现见 [`Inpainting`](../deepinv/physics/inpainting.py) 和
[`DecomposablePhysics`](../deepinv/physics/forward.py)。

### 3.2 Deblur

空间不变模糊写成

$$
Ax=H x=h*x,
$$

其中 `*` 是卷积。其伴随是边界规则相匹配的转置卷积：

$$
A^*y=H^*y.
$$

在无限域或圆周边界下，$H^*$ 等价于用空间翻转并取共轭的 kernel 卷积。只有
kernel 对称且边界处理兼容时，才可能有 $A^*=A$。

DeepInv 提供两条主要路径：

- `Blur`：执行真卷积，`A_adjoint` 使用对应的 transposed convolution；`valid`、
  `circular`、`replicate`、`reflect` 会改变离散矩阵，不能只对齐 kernel 数值。
  `A_dagger` 使用 `LinearPhysics` 的迭代最小二乘。
- `BlurFFT`：固定为 circular padding，在频域对角化，能直接使用
  `DecomposablePhysics` 的闭式 `A_dagger` 和 `prox_l2`。

圆周卷积下，令 $\widehat h$ 为 kernel 的离散傅里叶响应，则

$$
\widehat{Ax}=\widehat h\,\widehat x,
\qquad
\widehat{A^*y}=\overline{\widehat h}\,\widehat y,
$$

并且

$$
\widehat{A^\dagger y}_k=
\begin{cases}
\widehat y_k/\widehat h_k,&|\widehat h_k|>\varepsilon,\\
0,&\text{otherwise}.
\end{cases}
$$

直接伪逆会严重放大接近零频率响应处的测量噪声；这不是实现错误，而是病态反卷积
的性质。实际重建通常使用正则化近端或带先验算法。

跨仓库至少固定：kernel tensor、kernel 总和、kernel 中心/偶数尺寸相位、卷积还是
cross-correlation、padding、输出裁剪和 FFT normalization。最小诊断输入是单像素
impulse；它能直接暴露翻转、平移和边界差异。

实现见 [`Blur` 和 `BlurFFT`](../deepinv/physics/blur.py)。

### 3.3 Super-resolution

常用超分退化是先低通再抽取：

$$
A=S_rH,
$$

其中 $H$ 是抗混叠模糊，$S_r$ 每隔 $r$ 个像素保留一个样本。伴随必须逆序：

$$
A^*=H^*S_r^*.
$$

$S_r^*$ 在被抽取的位置放回测量值，其余位置补零；在标准欧氏内积下不额外乘
$r$ 或 $r^2$。普通双线性/双三次放大不是 $S_r^*$。由于下采样有大零空间，
通常不存在 $A^{-1}$，`A_dagger` 也只能给出与 measurement 一致的最小范数解。

DeepInv `Downsampling` 的具体顺序是：

1. filter 非空时，按指定 padding 做卷积 $H$；
2. 使用 `x[:, :, ::factor, ::factor]` 抽取，采样相位固定在左上角索引 0；
3. `A_adjoint` 先在同样索引零插值，再做 filter 的转置卷积。

配置了 filter 且 `padding="circular"` 时，`prox_l2` 可使用 FFT 闭式实现；其他情况
使用 `use_fft=False` 回退到迭代求解；
`A_dagger` 继承 `LinearPhysics` 的迭代最小二乘。`filter=None` 是裸 decimation，
不是 bicubic resize。

若参考实现声明 MATLAB bicubic，应优先检查 `DownsamplingMatlab`。它使用 bicubic、
antialiasing 和 reflect padding，并通过 autograd 生成真正的离散伴随；用倒数 scale
再 resize 回去并不是伴随。PIL、OpenCV、PyTorch interpolate、MATLAB `imresize`
即使都叫 bicubic，也可能在坐标映射、边界与 antialiasing 上不同。

跨仓库至少固定：factor、filter/resize 实现及版本、antialiasing、padding、采样相位、
输入输出 shape、值域和 channel 处理。

实现见 [`Downsampling` 和 `DownsamplingMatlab`](../deepinv/physics/blur.py)。

### 3.4 CT

二维 CT 的离散前向是射线积分/Radon 变换：

$$
y=Ax,
$$

DeepInv `Tomography` 的图像 shape 为 `[B,C,H,W]`，sinogram shape 为
`[B,C,num_angles,num_detectors]`。伴随 $A^*$ 是与离散前向严格配对、带正确权重的
**未滤波反投影**，不是 FBP：

$$
A^*:\text{sinogram}\rightarrow\text{image}.
$$

FBP 先对 detector 方向应用 ramp filter，再反投影并乘几何缩放：

$$
x_{\mathrm{FBP}}\approx cA^*Fy.
$$

它是连续逆公式的离散近似，通常不满足四个 Penrose 条件。稀疏角、有限角和离散
插值下，CT 同样可能欠定或病态。

当前 DeepInv `Tomography` 的关键选择是：

- `adjoint_via_backprop=True` 为默认值，使用前向离散实现的 VJP，得到与其配对的
  离散伴随；关闭时使用更省内存的逆 Radon 路径，但会有插值导致的伴随误差。
- `A_dagger(y, fbp=False)` 调用迭代最小二乘；`fbp=True` 明确选择快速 FBP 近似。
- `normalize=None` 会警告后设为 `True`；归一化后前向和伴随都除以估计的谱范数，
  FBP 也做相应补偿。正式 setting 必须显式写 `normalize`，不能依赖默认值。
- 整数 `angles=k` 生成 `[0,180)` 上均匀的 k 个角度；显式角度 tensor 的数值、顺序
  和单位必须保存。
- `circle`、parallel/fan beam、detector 数量/间距和图像像素尺度都是算子的一部分。

`TomographyWithAstra` 还支持 2D/3D parallel、fan 和 cone beam，但它的 ASTRA geometry、
体素/探测器尺度及 FDK weighting 都必须完整保存，不能仅记录“CT”。

实现见 [`Tomography` 和 `TomographyWithAstra`](../deepinv/physics/tomography.py)。

## 4. 算子对齐时必须保存的状态

| 任务 | 最小算子状态 |
|---|---|
| Inpainting | mask tensor、`1=保留` 约定、shape/广播方式、是否 channel-shared |
| Deblur | kernel tensor、卷积/相关、padding、kernel 中心、输出裁剪、FFT 约定 |
| Super-resolution | factor、filter 或 resize 实现、antialiasing、padding、采样相位、输入输出 shape |
| CT | angles tensor 与单位/顺序、geometry、detector/voxel spacing、`circle`、fan/cone 参数、normalize、backend |

共同还要保存 dtype、值域、clean measurement $Ax$、实际 measurement $y$、
measurement noise tensor，以及算子实现所在的仓库提交。正式 fixture 继续使用
`dict[str, torch.Tensor]` 的 `.pt`；字符串配置和 provenance 写 JSON。

相同任务名、相同 seed 或相同输出 shape 都不足以证明算子相同。最可靠的顺序是：

1. 比较固定输入上的 `A(x)`；
2. 做伴随内积测试；
3. 比较 $A^*y$；
4. 再比较 `A_dagger`/`prox_l2`，并明确它们是闭式还是迭代近似；
5. 最后才运行 DPS、DSG、DiffPIR 等重建算法。

## 5. 最小 Torch 校验

以下检查应在计划用于正式实验的 dtype/device 上分别对每个仓库运行。它验证线性与
伴随关系，不要求 CPU 和 GPU 逐元素一致。

```python
import torch


@torch.no_grad()
def check_linear_adjoint(physics, x, *, rtol=1e-5, atol=1e-6):
    x2 = torch.randn_like(x)
    a, b = 0.37, -0.81
    lhs_linear = physics.A(a * x + b * x2)
    rhs_linear = a * physics.A(x) + b * physics.A(x2)
    torch.testing.assert_close(lhs_linear, rhs_linear, rtol=rtol, atol=atol)

    y = torch.randn_like(physics.A(x))
    lhs = torch.vdot(physics.A(x).reshape(-1), y.reshape(-1))
    rhs = torch.vdot(x.reshape(-1), physics.A_adjoint(y).reshape(-1))
    torch.testing.assert_close(lhs, rhs, rtol=rtol, atol=atol)
```

CT 的非-backprop 伴随有已知插值误差，应报告实际 relative adjoint error，而不是盲目
放宽所有算法的全局容差。若此检查失败，先修正/对齐算子；重建指标偶然接近不能替代
算子正确性证明。
