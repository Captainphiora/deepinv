# DeepInv 反问题算法跨仓库对齐工作流

本文档描述当前已经落地的完整工作流。目标是在 DeepInv 中实现扩散模型或
flow matching 的反问题求解算法，并用算法原始仓库作为行为基准完成数值对齐。

## 1. 验收目标

每个算法的正式验收对象只有：

> 在同一依赖环境、同型号 GPU、同一模型权重、同一输入张量和同一随机增量下，
> DeepInv 实现是否与原始仓库实现一致。两个实现可以使用不同物理 GPU。

CPU 与 GPU 之间的数值一致性不属于验收项，也不为此增加测试。GPU UUID 和设备序号
只记录为 provenance，不作为通过条件。若两个仓库无法逐位一致，必须根据实际误差
下限声明显式容差，并定位误差来源；不能通过随意放宽容差掩盖系统性偏差。

工作流为：

```text
固定参考版本 → 参数与时间步审计 → 定义 setting → 生成不可变 fixture
             → 原始仓库论文门禁 → 实现与单元测试 → 单样本跨仓库对齐
             → 完整样本对齐 → 生成认证记录 → 合并长期分支
```

以下原则不可省略：

1. 原始仓库是算法行为基准，不为了提高指标而修改算法公式。
2. 相同 seed 不等于相同随机输入；真正参与计算的随机张量必须提前保存。
3. 先比较原始张量和中间轨迹，再比较 PSNR、SSIM、LPIPS。
4. `noise_level` 始终作为 setting 和 provenance 的一部分保留，即使某个算法当前
   没有直接使用它，或者它的值为零。
5. 所有正式文件和大体积 artifact 都保存在个人目录 `$HOME` 下，不依赖 `/tmp`。
6. 论文复现和跨仓库实现对齐是两个串行 gate：先证明“运行的是论文 setting”，再运行
   原始仓库检查论文指标；原始侧结论未记录前，不开始 DeepInv 正式运行。

### 1.1 三个不可颠倒的 gate

每个“论文 setting 对齐”任务必须依次通过或明确处置以下 gate：

1. **参数 gate**：从论文正文/附录和官方代码取得任务、数据集、退化、测量噪声、
   `lambda`、`zeta`、训练 noise schedule、推理 timestep 选择、NFE、`eta`、初始化、
   clipping、指标口径等；逐项记录来源、论文值、官方代码值和最终执行值。名称相似的
   字段不得合并，例如 linear beta schedule 与 quadratic timestep subsequence 是两个
   独立设置。
2. **原始仓库论文 gate**：只运行锁定 commit 的原始仓库，先生成独立的
   `reference_metrics.json`/实验报告。全量论文数据可按预注册容差给出 `PASS/FAIL`；
   子集只能标记 `NOT_COMPARABLE`，但仍须检查结果是否存在明显异常。若原始仓库不能
   复现论文指标，停止 DeepInv 正式运行并记录差距、已排查项和待确认问题。只有用户
   明确接受“改为对齐官方可执行行为”后，才能绕过论文 gate，且 setting/报告名称必须
   标明它不是论文指标复现。
3. **跨仓库 gate**：原始侧基准获得接受后，DeepInv 才读取同一 fixture、schedule 和
   随机 tape 运行，依次比较 timestep、trajectory、raw reconstruction 和图像指标。

不得先写完或运行 DeepInv，再回头猜论文参数；也不得因为 DeepInv 与一个错误的
reference 数值相近，就宣称论文或算法正确性已经得到验证。

### 1.2 新 agent 接手入口

新 agent 不应先重新实现算法。按以下顺序读取和检查：

1. `$HOME/AGENTS.md`：机器、持久化目录、uv 和 Git 的通用约束。
2. `.agents/skills/align-inverse-solver/SKILL.md`：跨仓库数值对齐的强制约定。
3. 本文档：通用工作流、字段字典和完成条件。
4. `reproduction/LINEAR_DEGRADATION_OPERATORS.zh-CN.md`：线性退化的前向、
   伴随、伪逆、噪声域和跨仓库对齐约定。
5. `reproduction/<algorithm>/README.md`：该算法的参考仓库、兼容说明和命令。
6. `reproduction/<algorithm>/settings/<setting-id>.json`：本次实验的唯一配置。
7. `reproduction/<algorithm>/certifications/<certification-id>.json`：已通过结果和
   artifact 哈希。
8. 最后再读取算法实现、runner 和测试代码。

接手后首先运行以下只读检查：

```bash
git -C "$HOME/deepinv" status --short --branch
git -C "$HOME/deepinv" log --oneline --decorate -5

find "$HOME/deepinv/reproduction/artifacts" -maxdepth 5 -type f \
  \( -name manifest.json -o -name comparison.json \) -print
```

如果 certification 存在但 `comparison.json` 不存在，不代表认证文件损坏。
`comparison.json`、fixture 和原始张量属于 Git 忽略的本地 artifact，新机器或全新
clone 不会自动拥有它们；应根据 certification 中的 setting、run ID、路径和哈希
重新生成，不能伪造同名空文件。

新 agent 开始工作前必须回答以下问题：

- 当前操作的是 `research/reproduction` 还是独立的算法分支/worktree？
- 原始仓库是否是 setting 指定提交上的干净 worktree？
- checkpoint SHA256 是否与 setting 一致？
- 固定 fixture 是否存在，manifest SHA256 是否与 certification 一致？
- 两个 runner 是否会在同一个 uv 环境和同型号/计算能力的 GPU 上运行？
- 本次是复查已有认证、增加新 setting，还是实现新算法？

## 2. 分支与目录

### 2.1 Git 分支

- `research/reproduction`：长期集成分支，保存所有已经完成的算法实现、测试、
  setting 和认证记录。
- `repro/<algorithm>-alignment`：单个算法的短期开发分支，例如
  `repro/dsg-alignment`。
- 原始算法仓库使用独立、干净、detached 的 worktree 固定到参考提交，不能直接用
  存在本地修改的日常开发 checkout 做正式认证。

推荐创建方式：

```bash
git -C "$HOME/deepinv" worktree add \
  -b repro/<algorithm>-alignment \
  "$HOME/deepinv-worktrees/<algorithm>-alignment" \
  research/reproduction

git -C "$HOME/<reference-repository>" worktree add --detach \
  "$HOME/reference-worktrees/<algorithm>-<commit>" \
  <reference-commit>
```

完成认证后，将算法分支推送到远端，再把长期分支 fast-forward 到该提交：

```bash
git -C "$HOME/deepinv-worktrees/<algorithm>-alignment" push \
  -u origin repro/<algorithm>-alignment

git -C "$HOME/deepinv" merge --ff-only repro/<algorithm>-alignment
git -C "$HOME/deepinv" push origin research/reproduction
```

### 2.2 DeepInv 内部布局

```text
deepinv/
├── .venv/                              # DeepInv 独立 uv 环境
├── deepinv/
│   └── sampling/                       # 可复用算法实现及公开导出
├── deepinv/tests/                      # 算法单元测试
└── reproduction/
    ├── ALIGNMENT_WORKFLOW.zh-CN.md      # 本文档
    ├── README.md                       # reproduction 总入口
    ├── dps/
    │   ├── _common.py                  # 当前共享的 artifact/manifest 工具
    │   ├── prepare_inputs.py           # 当前共享的 fixture 生成器
    │   └── compare.py                  # 当前共享的跨仓库比较器
    ├── <algorithm>/
    │   ├── README.md                   # 算法来源、差异和运行命令
    │   ├── run_reference.py            # 调用原始仓库
    │   ├── run_deepinv.py              # 调用 DeepInv 实现
    │   ├── test_<algorithm>_harness.py
    │   ├── settings/
    │   │   └── <setting-id>.json
    │   └── certifications/
    │       └── <certification-id>.json
    └── artifacts/                      # Git 忽略，但保存在 $HOME/deepinv 下
        ├── fixtures/
        └── runs/
```

当前 `_common.py`、`prepare_inputs.py` 和 `compare.py` 最初随 DPS 建立，因此仍位于
`reproduction/dps/`；DSG 已直接复用它们。新增算法优先复用现有工具，不为每个算法
复制一套。若其接口确实无法继续覆盖后续算法，再单独迁移到
`reproduction/common/`，而不是预先增加抽象。

每个 setting 的大文件输出目录固定为：

```text
reproduction/artifacts/runs/<algorithm>/<setting-id>/<run-id>/
```

## 3. 环境管理

DeepInv 使用仓库自己的 uv 环境：

```bash
cd "$HOME/deepinv"
"$HOME/uv-env-tool.sh" --source china --proxy off \
  sync --locked --group dev --extra reproduction
```

- 环境路径：`$HOME/deepinv/.venv`
- uv 共享缓存：`$HOME/.cache/uv`
- Python 和依赖版本由 `pyproject.toml` 与 `uv.lock` 固定。
- 后续下载会自动复用 uv 缓存，不使用 `--no-cache`，也不清理共享缓存。
- 新依赖必须用 `uv add`，不能只执行临时的 `pip install`：

```bash
cd "$HOME/deepinv"
"$HOME/uv-env-tool.sh" --source china --proxy off \
  add --optional reproduction <package>
```

原始仓库 runner 和 DeepInv runner 都由这一个 `.venv` 执行。若原始仓库只能依赖旧
接口，则建立有明确提交记录的最小兼容提交；兼容修改不能改变算法公式和数值语义。
setting 中同时记录上游算法提交和实际执行的兼容提交。

正式运行时 manifest 会记录 Python、Torch、NumPy、CUDA、cuDNN、确定性开关、
GPU 型号、计算能力、设备序号和 UUID。比较器要求软件环境、设备类型、GPU 型号和
计算能力一致，但忽略设备序号与 UUID；后两者只用于定位任务实际落在哪张卡上。

## 4. 数值文件格式

### 4.1 为什么选择 `.pt`

正式数值真值统一使用 CPU 上的 `.pt` 文件，内容限制为
`dict[str, torch.Tensor]`：

- 能原样保存 dtype、shape 和多张量结构；
- 不需要为一个样本维护多份 `.npy` 和额外字段映射；
- 可使用 `torch.load(..., weights_only=True, map_location="cpu")`；
- 与 DeepInv 和原始 PyTorch 仓库之间没有额外转换误差。

`.npy` 只在外部工具明确要求时作为派生导出格式，不作为认证真值。PNG/JPEG 只用于
人工预览，不能重新读回后计算对齐误差或指标。

### 4.2 JSON 的职责

JSON 只保存配置、provenance、哈希和标量指标，不保存大张量。每个 `.pt` 记录：

- 文件 SHA256；
- 基于 tensor key、dtype、shape 和原始字节计算的 tensor-content SHA256；
- 每个张量的名称、dtype 和 shape。

fixture 的 `manifest.json` 还会被整体计算 SHA256。两个 runner 必须把同一个 fixture
manifest 哈希写入 run manifest，因此 transition-noise tape 也被绑定到认证结果。

### 4.3 文件职责总表

| 文件 | Git 跟踪 | 生产者 | 消费者 | 用途 |
|---|---|---|---|---|
| `settings/<id>.json` | 是 | 开发者 | fixture builder、两个 runner、compare | 不可变实验配置 |
| fixture `manifest.json` | 否 | `prepare_inputs.py` | 两个 runner、compare | 固定输入目录及全部哈希 |
| fixture `.pt` | 否 | `prepare_inputs.py` | 两个 runner | 唯一数值输入真值 |
| run `manifest.json` | 否 | 两个 runner | compare、certification | 两次执行的提交、环境和输出索引 |
| `reference.pt` | 否 | reference runner | compare | 原始仓库 raw 输出 |
| `deepinv.pt` | 否 | DeepInv runner | compare | DeepInv raw 输出 |
| `comparison.json` | 否 | `compare.py` | 开发者、certification | 两套实现的独立指标及差值 |
| `certifications/*.json` | 是 | 通过后的人工整理 | 审查者、新 agent | 可提交的精简认证结论 |

“Git 跟踪为否”的文件位于 `reproduction/artifacts/`。它们在当前个人目录中持久化，
但不会被 `git status`、GitHub 或全新 clone 展示。certification 中的相对路径指向这些
本地文件，并用 SHA256 防止把其他运行误认为认证运行。

### 4.4 Setting 字段

| 字段 | 含义 |
|---|---|
| `schema_version` | setting 格式版本；字段语义变化时递增 |
| `id` | setting 的稳定唯一 ID，也是默认目录名的一部分 |
| `algorithm.name` | 算法名，决定 runner 和 artifact 的算法目录 |
| `algorithm.*` | 算法本身参数，例如 DPS `scale` 或 DSG `guidance_scale`、`interval` |
| `task.name` | 反问题类型，例如 `inpainting` |
| `task.image_size` | `[C,H,W]`，不包含 batch 维 |
| `task.batch_size` | 正式认证的 batch 大小 |
| `task.measurement_noise_sigma` | 实际生成 measurement 时使用的噪声强度 |
| `task.noise_level` | 保留并可能传给模型/算法的噪声配置量；不能因为暂未读取而删除 |
| `task.value_range` | 模型和保存 raw tensor 使用的值域 |
| `model.checkpoint_sha256` | 模型权重的内容标识；runner 在加载前强制验证 |
| `model.model_input_type` | 模型接收 timestep、sigma 或其他时间参数化 |
| `model.prediction_type` | 模型预测 epsilon、score、velocity 或 clean sample |
| `model.variance_type` | 固定方差或 learned/learned-range 方差约定 |
| `sampler.*` | 训练步数、采样步数、schedule、respacing、eta、clipping 等 |
| `randomness.*_seed` | 生成固定张量时的 provenance，不代替保存的张量 |
| `randomness.deterministic_algorithms` | 是否要求 Torch 确定性算子 |
| `randomness.allow_tf32` | 是否允许 TF32；两个实现必须一致 |
| `trajectory_probe_steps` | 需要保存和比较的反向循环序号，不是原始 diffusion timestep |
| `thresholds` | raw tensor、轨迹和图像指标的显式通过门限 |
| `reference.repository` | 原始上游仓库 |
| `reference.upstream_commit` | 被视为算法行为基准的原始提交 |
| `reference.compatibility_repository` | 现代环境兼容提交所在仓库；不需要兼容层时可省略 |
| `reference.compatibility_commit` | reference runner 实际执行的提交 |

### 4.5 Fixture manifest 与输入 `.pt`

fixture manifest 顶层字段：

| 字段 | 含义 |
|---|---|
| `schema_version` | fixture manifest 格式版本 |
| `fixture_id` | 不可变 fixture 标识 |
| `created_at` | UTC 创建时间 |
| `source_setting_id` | 生成 fixture 时使用的 setting ID |
| `source_setting_sha256` | 当时 setting 文件的 SHA256 |
| `randomness` | 生成这些张量所用 seed 和确定性配置 |
| `schedule` | `schedule.pt` 的路径、文件哈希、tensor 哈希和张量描述 |
| `schedule_tensor_sha256` | schedule 张量内容的整体哈希 |
| `cases` | 每个固定样本的记录 |

每个 artifact record 通用字段：

| 字段 | 含义 |
|---|---|
| `path` | 相对于所属 fixture/run 目录的路径 |
| `sha256` | `.pt` 文件字节的 SHA256 |
| `tensor_sha256` | 忽略 PyTorch 文件封装、只依据 tensor 内容计算的 SHA256 |
| `tensors` | 每个 tensor 的 dtype 和 shape，不包含数值本身 |

每个 `cases/<case-id>.pt` 包含：

| Tensor | 含义 |
|---|---|
| `ground_truth` | 完成统一预处理后的真值图像 |
| `measurement` | 两个实现实际接收的观测 |
| `mask` | inpainting physics 的固定 mask；其他任务替换为相应算子状态 |
| `measurement_noise` | 本 case 实际使用的测量噪声张量 |
| `x_init` | 两个反向过程共同使用的初始点 |

`schedule.pt` 包含 `timesteps`、`betas`、`alpha_cumprod` 和 `noise_levels`。
随机 sampler 还具有 `noise/<case-id>.pt`，其中 `transition_noise` 的第 0 维按反向
循环消费顺序排列。DSG DDIM-100 `eta=1` 的 shape 是
`[100,1,3,256,256]`。

case manifest 中的 `sources` 记录原图、legacy measurement 和 `x_init` 的来源文件
名及 SHA256，或者记录生成 `x_init` 的 seed。`transition_noise` 字段是对应 noise
artifact record，并额外记录生成 tape 的 seed。

### 4.6 Run manifest 与输出 `.pt`

run manifest 顶层字段：

| 字段 | 含义 |
|---|---|
| `schema_version` | run manifest 格式版本 |
| `setting_id` / `setting_sha256` | 本次执行绑定的 setting |
| `fixture_id` / `fixture_manifest_sha256` | 本次执行绑定的全部固定输入 |
| `run_id` | 此次实验的不可复用标识 |
| `updated_at` | 最后一个实现写入 manifest 的 UTC 时间 |
| `implementations.reference` | 原始仓库运行记录 |
| `implementations.deepinv` | DeepInv 运行记录 |

每个 implementation record 包含：

| 字段 | 含义 |
|---|---|
| `created_at` | 该实现完成时间 |
| `command` | 实际 Python 可执行文件和完整命令参数 |
| `repository` / `revision` | 实际执行仓库及 Git revision |
| `upstream_revision` | 原始算法提交，仅 reference 需要 |
| `compatibility_revision` | 实际兼容提交，仅使用兼容层的 reference 需要 |
| `checkpoint_sha256` | 实际加载的模型权重哈希 |
| `environment` | Python、依赖、CUDA、确定性设置、GPU 型号/计算能力、序号和 UUID |
| `cases` | 此实现的输出 `.pt` 索引和哈希 |

`environment.gpu.uuid` 和 `device` 只记录两个进程实际落在哪张卡上。
`environment.gpu.name` 与 `compute_capability` 才参与环境兼容判断，因此同型号的
不同物理 GPU 可以并行完成 reference 与 DeepInv 运行。

每个输出 `.pt` 的公共 tensor：

| Tensor | 含义 |
|---|---|
| `reconstruction` | 最终未经过 PNG 往返的 raw 重建结果 |
| `trajectory` | 初始点和选定 probes 的 raw 中间状态 |
| `trajectory_steps` | `trajectory` 每一项对应的反向循环序号；`-1` 表示 `x_init` |
| `timesteps` | 每次反向循环实际送入 diffusion schedule 的 timestep |
| `noise_levels` | 每个采样步骤对应的显式噪声水平 |

reference 输出还可包含 `distances`，用于诊断原实现每一步的 measurement distance。
它不是两个实现必须共有的认证字段，不参与当前公共张量比较。

### 4.7 Reference metrics 与论文门禁字段

`reference_metrics.json` 必须同时保存原始仓库指标和本次真正执行的参数快照：

| 字段 | 含义 |
|---|---|
| `implementation` | 固定为 `reference`，避免与 DeepInv 指标混淆 |
| `cases` / `mean` | 原始仓库逐例和均值 PSNR、SSIM、LPIPS |
| `paper_reference` | 论文报告的样本数及指标；未报告的指标必须明确标注 |
| `paper_metrics_comparable` | 当前输入规模与指标口径能否直接用于论文数值 gate |
| `paper_metrics_note` | 不可直接比较时的原因 |
| `parameter_audit.status` | 参数快照是否已写入；`RECORDED` 不等于论文 gate `PASS` |
| `parameter_audit.algorithm/task/model/sampler/randomness/reference` | setting 中的实际执行值快照 |
| `parameter_audit.sampled_timesteps` | 从 fixture 读取的完整实际 model timestep 列表 |
| `parameter_audit.schedule_tensor_sha256` | schedule tensor 内容哈希 |

实验报告在上述机器可读字段之外给出 `PASS`、`FAIL`、`NOT_COMPARABLE` 或 `BLOCKED`
结论。setting ID/SHA 负责防篡改，但不能替代报告中直接展开 `lambda/zeta`、schedule 和
timestep。

### 4.8 Comparison 字段与指标方向

`comparison.json` 顶层字段：

| 字段 | 含义 |
|---|---|
| `schema_version` / `created_at` | 格式版本和 UTC 创建时间 |
| `setting_id` / `fixture_id` / `run_id` | comparison 所属实验 |
| `thresholds` | 从 setting 读取的通过门限快照 |
| `cases` | 每个 case 的两套指标、差值、raw tensor 差异和 probes |
| `mean` | `reference`、`deepinv` 的样本均值、绝对差值和通过状态 |
| `passed` | 所有 case 和 mean gate 是否全部通过 |

每个 case 字段：

| 字段 | 含义 |
|---|---|
| `id` | case ID |
| `reference` | 原始仓库 reconstruction 相对 ground truth 的指标 |
| `deepinv` | DeepInv reconstruction 相对同一 ground truth 的指标 |
| `delta` | 两套实现对应指标的绝对差值，不是重建误差本身 |
| `final_tensor_difference` | `deepinv.reconstruction - reference.reconstruction` 的张量差异 |
| `trajectory` | 每个 probe 上 DeepInv 与 reference 的张量差异 |
| `passed` | 此 case 的所有轨迹和指标差值是否满足门限 |

图像指标方向：PSNR、SSIM 越大通常越好，LPIPS 越小通常越好。`delta` 越接近 0，
只表示两个实现越一致，并不表示重建质量越高。

张量差异字段：

| 字段 | 定义 |
|---|---|
| `mae` | 所有元素绝对差的平均值 |
| `rmse` | 所有元素平方差均值的平方根 |
| `max_abs` | 任意一个元素上的最大绝对差 |
| `relative_l2` | `||deepinv-reference||₂ / max(||reference||₂, 1e-12)` |
| `passed` | 此 probe 是否满足 setting 中的轨迹门限 |

实际 case `00000` 的精简结构如下；完整文件仍保存在 artifact 目录：

```json
{
  "id": "00000",
  "reference": {
    "psnr_db": 36.98387502798004,
    "ssim": 0.9620132446289062,
    "lpips": 0.09253563731908798
  },
  "deepinv": {
    "psnr_db": 36.98387502798004,
    "ssim": 0.9620132446289062,
    "lpips": 0.09253563731908798
  },
  "delta": {
    "psnr_db": 0.0,
    "ssim": 0.0,
    "lpips": 0.0
  },
  "final_tensor_difference": {
    "mae": 0.0,
    "rmse": 0.0,
    "max_abs": 0.0,
    "relative_l2": 0.0
  },
  "trajectory": [
    {
      "step": 0,
      "mae": 0.0,
      "rmse": 0.0,
      "max_abs": 0.0,
      "relative_l2": 0.0,
      "passed": true
    }
  ],
  "passed": true
}
```

### 4.9 Certification 字段

certification 是 `comparison.json` 的可提交摘要，不取代完整 comparison。当前分别
记录两套指标的结构使用 `schema_version=2`。字段包括：

| 字段 | 含义 |
|---|---|
| `status` | 正式结论，只在全部 gate 通过后写 `PASS` |
| `setting` / `fixture` / `run_id` | 被认证实验及其哈希 |
| `implementations` | reference、上游算法、DeepInv、比较器提交和 checkpoint 哈希 |
| `environment` | 正式执行环境、GPU 型号/计算能力及作为 provenance 的 GPU UUID |
| `alignment` | 所有 case 上最坏的 raw、轨迹和指标差值 |
| `cases[].reference` | 原始仓库每例指标 |
| `cases[].deepinv` | DeepInv 每例指标，即使与 reference 相同也单独保存 |
| `cases[].delta` | 每例指标绝对差值 |
| `mean.reference` / `mean.deepinv` / `mean.delta` | 两套均值及差值 |
| `artifacts` | 完整 run manifest/comparison 相对路径和 SHA256 |

新 certification 不允许因为两套指标恰好相同而合并成一份无标签的指标。

## 5. 阶段 A：审计原始实现

写 DeepInv 代码前，先完成以下只读检查：

1. 记录原始仓库 URL、算法提交、兼容提交及二者的关系。
2. 记录模型 checkpoint 路径和 SHA256，不能只记录文件名。
3. 找到论文配置实际对应的运行入口和 setting，不能依据默认参数猜测。
4. 追踪一轮反向更新的完整调用链：
   - 模型输入参数化和输出参数化；
   - beta/sigma/noise schedule；
   - timestep respacing、映射和 rescaling；
   - DDPM、DDIM 或 ODE/SDE 更新；
   - 条件梯度、data fidelity 和物理算子；
   - measurement noise；
   - clipping、归一化和颜色空间；
   - RNG 调用次数和顺序；
   - 输出保存前的转换。
5. 明确算法更新的运算顺序、norm 范围、epsilon、广播方式和 batch 约定。
6. 确认参考 worktree 中算法相关路径为干净状态。

审计结果必须形成逐项参数表，至少包含四列：`字段`、`论文证据`、`官方代码证据`、
`最终执行值`。特别要分别记录：

- beta/sigma/noise schedule 的类型与端点；
- timestep subsequence/respacing 的公式和最终整数列表；
- NFE 与实际模型调用次数；
- 任务、噪声和 NFE 共同决定的算法参数，例如 `lambda`、`zeta`；
- 论文指标的数据集大小、颜色空间、裁边、量化和集合级指标口径。

如果论文与官方默认值不同，必须使用独立 setting ID，并在报告中解释如何让原始
runner 执行论文值。不能把配置文件中的默认值或结果目录名当成实际执行值。

如果现代 PyTorch 无法直接运行旧代码，只允许建立最小兼容补丁。例如把当前 PyTorch
不再接受的多维 `torch.linalg.norm(..., dim=[...])` 改为数学等价的
`torch.linalg.vector_norm(..., dim=(...))`，并同时记录上游提交和兼容提交。

## 6. 阶段 B：定义不可变 setting

每一种算法与实验条件的组合使用独立 setting ID。不要在已认证的 JSON 上直接修改
参数；改变算法、采样器、步数、噪声、模型或预处理时创建新 ID。

setting 至少包含：

```json
{
  "schema_version": 1,
  "id": "<dataset>_<task>_<sampler>_<algorithm>_v1",
  "algorithm": {
    "name": "<algorithm>"
  },
  "task": {
    "name": "inpainting",
    "image_size": [3, 256, 256],
    "batch_size": 1,
    "measurement_noise_sigma": 0.05,
    "noise_level": 0.05,
    "value_range": [-1.0, 1.0]
  },
  "model": {
    "checkpoint_sha256": "<sha256>",
    "model_input_type": "timestep",
    "prediction_type": "epsilon",
    "variance_type": "learned_range"
  },
  "sampler": {
    "name": "ddim",
    "train_steps": 1000,
    "sampling_steps": 100,
    "timestep_respacing": 100,
    "eta": 1.0
  },
  "randomness": {
    "fixture_seed": 42,
    "measurement_seed": 42,
    "x_init_seed": 42,
    "transition_seed": 42,
    "deterministic_algorithms": true,
    "allow_tf32": false
  },
  "trajectory_probe_steps": [0, 1, 2, 50, 98, 99],
  "thresholds": {},
  "reference": {
    "repository": "<upstream-url>",
    "upstream_commit": "<commit>",
    "compatibility_repository": "<compatibility-url>",
    "compatibility_commit": "<commit>"
  }
}
```

`measurement_noise_sigma` 表示测量生成过程真正使用的噪声强度；`noise_level` 是需要
保留并可能传给模型或算法的配置量。即使当前 setting 中两者数值相等，也不能因为
某个实现暂时没有读取 `noise_level` 就删除它。

setting 定稿前必须完成参数 gate。每个 schedule 需要同时保存“生成规则”和“展开后的
实际 timestep”；只记录 `quad`、`linear` 或 `100 steps` 这样的简称不足以复现。

## 7. 阶段 C：生成固定 fixture

fixture 必须在任一实现运行前生成。每个 case 至少保存：

- `ground_truth`：完成 resize、颜色空间和值域转换后的模型真值；
- `measurement`：实际送入算法的观测；
- mask、kernel 或其他 physics 状态；
- `measurement_noise`：本次实际采样的测量噪声张量；
- `x_init`：反向过程的初始点；
- schedule：timesteps、betas、累计 alpha 和显式 noise levels；
- transition-noise tape：每个随机反向步骤真正使用的随机张量。

DDPM、`eta > 0` 的 DDIM，以及随机 SDE/flow solver 必须保存逐步随机 tape。确定性
DDIM、ODE 或确定性 flow solver 仍必须保存 `x_init`。seed 只用于 provenance，不能
替代这些张量。

当前共享 fixture 生成器示例：

```bash
cd "$HOME/deepinv"

.venv/bin/python reproduction/dps/prepare_inputs.py \
  --setting <setting-id> \
  --fixture-id <fixture-id> \
  --images <clean-image-directory> \
  --measurements <measurement-directory> \
  --x-init <x-init-directory> \
  --with-transition-noise \
  --dry-run

# 确认解析结果后，去掉 --dry-run 正式生成。
```

生成器拒绝覆盖已有 fixture。若只想更换第 3 张图的初始点而不改变算法，应新建
fixture ID，并使用类似 `--x-init-seed 00002=43` 的显式 override；原 setting 保持
不变。

### 7.1 原始仓库论文门禁

fixture 生成后，先只运行原始仓库并写 `reference_metrics.json` 和实验报告。报告必须
直接展开以下内容，不能只依赖 setting ID 或 SHA256 间接回溯：

- `lambda`、`zeta`、NFE、`eta`、measurement/algorithm noise；
- noise schedule 及 beta 端点；
- timestep 选择类型、公式和实际 timestep 列表；
- task/operator/kernel/mask、初始化和随机性；
- 输入数量、指标口径、论文指标和原始仓库指标；
- gate 状态及原因。

若只使用论文数据集的子集，论文集合均值不得作为硬阈值，状态写
`NOT_COMPARABLE`；如果结果同时出现明显退化，报告可进一步写 `BLOCKED`，并在用户
确认前不进入阶段 D。正式 DeepInv shell 不得在同一次无人检查的流水线中越过一个
尚未确认的原始仓库 gate。

## 8. 阶段 D：实现 DeepInv 算法

只有阶段 7.1 的原始仓库基准被接受后，才开始正式 DeepInv 实现/运行。实现时按以下
顺序处理：

1. 优先复用 DeepInv 已有 model wrapper、physics、reconstructor 和 sampler。
2. 只增加与原始仓库达到数值等价所必需的兼容层。
3. 保持原始公式的运算顺序、dtype、timestep 语义和 clipping 位置。
4. 不把原始仓库整段复制到 `deepinv/`。
5. 在 `deepinv/sampling/__init__.py` 等既有入口公开算法。
6. 对原始实现只支持的范围作明确说明，例如 DSG 的正式认证范围是 batch size 1。

最小测试集合包括：

1. 单步公式测试：使用小张量独立计算参考公式，并检查更新结果。
2. 一条端到端固定 tape 测试：覆盖真实 `forward()`、timestep/interval 接线和轨迹。
3. 参数边界测试：只覆盖会改变算法语义的关键非法参数。
4. 现有相邻算法的回归测试。

不增加 CPU 对 GPU 的一致性测试。

## 9. 阶段 E：单样本跨仓库对齐

先提交算法、setting 和 harness，使 runner 记录到的 revision 能唯一指向执行代码。
正式 DeepInv runner 会拒绝算法或 harness 相关路径存在未提交修改的运行。

先对一个 case 执行 `--dry-run`，检查：

- setting、fixture 和 case ID；
- 原始仓库提交；
- DeepInv 提交；
- checkpoint SHA256；
- artifact 输出目录。

然后在同型号 GPU 上运行原始仓库和 DeepInv。设备序号可以不同，因此两边可以并行：

```bash
cd "$HOME/deepinv"

.venv/bin/python reproduction/<algorithm>/run_reference.py \
  --setting <setting-id> \
  --fixture-id <fixture-id> \
  --run-id <smoke-run-id> \
  --reference-repo <clean-reference-worktree> \
  --checkpoint <checkpoint> \
  --case 00000 \
  --device cuda:0

.venv/bin/python reproduction/<algorithm>/run_deepinv.py \
  --setting <setting-id> \
  --fixture-id <fixture-id> \
  --run-id <smoke-run-id> \
  --checkpoint <checkpoint> \
  --case 00000 \
  --device cuda:1

.venv/bin/python reproduction/dps/compare.py \
  --setting <setting-id> \
  --fixture-id <fixture-id> \
  --run-id <smoke-run-id> \
  --case 00000 \
  --metric-device cuda:0
```

同一个 `run-id` 下每个实现只能写入一次，comparison 也不能覆盖。重新实验时使用新
run ID，避免认证 provenance 被静默替换。reference 与 DeepInv 并行结束时，run
manifest 通过 AFS 可用的原子目录锁合并写入，不会互相覆盖。
若进程被强制终止并遗留 `manifest.json.lock/`，必须先确认没有 runner 正在写入，再只
删除该 run 目录内的锁目录；不能删除 manifest 或已有 `.pt` 输出。

四卡机器可以同时运行两组 setting，例如：

| GPU | 任务 |
|---|---|
| GPU 0 | reference，setting A |
| GPU 1 | DeepInv，setting A |
| GPU 2 | reference，setting B |
| GPU 3 | DeepInv，setting B |

UUID 不同不会直接失败。如果 setting A 超出张量容差，再把失败 case 的 reference 和
DeepInv 依次放到同一张 GPU 上重跑。只有同卡重跑仍失败，才优先判断为实现偏差。

### 9.1 比较顺序

比较器按以下顺序拒绝不公平实验：

1. setting ID 和 setting SHA256；
2. fixture ID 和整个 fixture manifest SHA256；
3. Python/Torch/NumPy/CUDA/cuDNN、确定性开关、设备类型、GPU 型号和计算能力；
4. checkpoint SHA256；
5. timestep map、显式 noise levels 和 trajectory probe 编号；
6. raw reconstruction 与中间 trajectory 的 MAE、RMSE、`max_abs`、relative L2；
7. 两边相对同一 ground truth 的 PSNR、SSIM、LPIPS 及指标差值。

指标高不代表实现等价。只有 raw tensor 和中间轨迹首先满足门限，图像指标才用于
描述重建质量。

### 9.2 Probe 的定义

probe 是选定的反向采样中间状态检查点，只负责保存和比较，不改变算法计算。
`trajectory_probe_steps` 使用反向循环序号，而不是训练 diffusion timestep：

- `step=-1`：尚未执行反向更新的固定 `x_init`，由 runner 自动保存；
- `step=0`：最高噪声端第一次 sampler + conditioning 更新后的状态；
- `step=k`：第 `k+1` 次更新后的状态；
- 最后一个 step：最终 reconstruction。

`trajectory_steps` 保存上述标签，`trajectory` 以相同顺序保存张量。实际 diffusion
timestep 必须读取同一输出 `.pt` 中的 `timesteps[step]`，不能根据采样步数自行猜测，
因为 respacing 可能采用非等间隔或取整。

当前 DSG DDIM-100 setting 的对应关系为：

| `trajectory_steps` | 状态 | 实际 diffusion timestep |
|---:|---|---:|
| `-1` | `x_init` | 尚未执行 |
| `0` | 第 1 次更新后 | 999 |
| `1` | 第 2 次更新后 | 989 |
| `2` | 第 3 次更新后 | 979 |
| `50` | 第 51 次更新后 | 494 |
| `98` | 第 99 次更新后 | 10 |
| `99` | 第 100 次更新后，即 reconstruction | 0 |

每个 probe 分别计算 DeepInv 与 reference 的 MAE、RMSE、`max_abs` 和 relative L2。
probe 的用途是确定偏差最早出现在哪次更新，而不是替代最终重建指标。

### 9.3 偏差定位

单样本失败时，不立刻运行完整数据，也不先调大容差。若两边使用不同 UUID，先在
同一张 GPU 上重跑失败 case；然后找到第一个偏离的 probe，并按以下类别排查最早
根因：

1. 输入或预处理；
2. checkpoint/model adapter；
3. schedule、respacing、timestep 或模型时间输入；
4. transition noise tape 的顺序或形状；
5. physics、measurement noise 或 `noise_level`；
6. 条件梯度、norm 范围、广播和更新顺序；
7. dtype、设备、TF32 或序列化。

只修复最早的根因，然后从单样本重新运行。

## 10. 阶段 F：完整实验与认证

单样本通过后，用新 `run-id` 去掉 `--case`，对 setting 中规定的完整样本集分别运行
reference、DeepInv 和 compare。正式认证需要同时满足：

- 所有 case 的 raw reconstruction 误差通过；
- 所有 trajectory probes 通过；
- timestep 和 noise levels 完全一致；
- 每例和均值 PSNR、SSIM、LPIPS 差值通过；
- 两个 runner 的软件环境、设备类型、GPU 型号和计算能力一致；
- setting、fixture、checkpoint、代码和参考实现均有哈希或提交记录。

通过后，只把小型认证 JSON 提交到：

```text
reproduction/<algorithm>/certifications/<certification-id>.json
```

认证 JSON 至少保存：

- `PASS` 状态；
- setting ID/SHA256；
- fixture ID/manifest SHA256；
- run ID；
- 原始算法、兼容实现、DeepInv 和比较器提交；
- checkpoint SHA256；
- 环境、GPU 型号/计算能力，以及作为 provenance 的设备序号和 UUID；
- raw tensor、轨迹和指标的最大跨仓库差值；
- 每个 case 及均值的重建指标；
- 完整 run manifest 和 comparison JSON 的相对路径及 SHA256。

fixture、逐步噪声、trajectory 和 reconstruction 等大文件继续保留在 Git 忽略的
`reproduction/artifacts/`，由认证 JSON 中的哈希验证。

## 11. 当前 DSG 已认证实例

当前 DSG 认证使用：

- setting：`ffhq256_inpainting_ddim100_eta1_dsg_v1`
- 任务：FFHQ 256×256 三张图 inpainting
- sampler：DDIM 100/1000，`eta=1`
- DSG：`guidance_scale=0.2`，`interval=1`
- measurement noise 与保留的 `noise_level`：`0.05`
- batch size：1
- 原始算法提交：`b217c7a2463f9ebd68e12fe6b6d91344f195d1b8`
- 兼容执行提交：`99b4e99c57886226fc852218286d951d65edab5e`
- DeepInv 执行提交：`6c6e6729dea4674de9ae73969348d3b9ecb06aa8`
- checkpoint SHA256：
  `81d535743156ec6be34d8668e6920da94f0614074d7793a16c8fa9e306237faa`
- fixture manifest SHA256：
  `5f47c27c60a203ba0ac5d39c64178c12fa0bf3e6797bf9fb692ad26c312c9a8a`
- 设备：本次历史认证恰好在同一张 NVIDIA GeForce RTX 5090 上完成；UUID 仅记录
  provenance，后续认证允许使用不同的同型号 GPU

三张图的 reference 与 DeepInv 指标分别记录如下：

| case | implementation | PSNR (dB) | SSIM | LPIPS |
|---|---|---:|---:|---:|
| `00000` | reference | 36.983875 | 0.962013 | 0.092536 |
| `00000` | DeepInv | 36.983875 | 0.962013 | 0.092536 |
| `00001` | reference | 34.476085 | 0.933090 | 0.100677 |
| `00001` | DeepInv | 34.476085 | 0.933090 | 0.100677 |
| `00002` | reference | 34.555766 | 0.947065 | 0.112467 |
| `00002` | DeepInv | 34.555766 | 0.947065 | 0.112467 |
| mean | reference | 35.338575 | 0.947389 | 0.101893 |
| mean | DeepInv | 35.338575 | 0.947389 | 0.101893 |

六个轨迹 probe 的 MAE、RMSE、`max_abs` 和 relative L2 也全部为 0；PSNR、SSIM、
LPIPS 的跨仓库差值和最终 reconstruction `max_abs` 全部为 0。正式记录位于：

```text
reproduction/dsg/certifications/ddim100_eta1_fixed_v1.json
```

完整 tensor artifact 位于：

```text
reproduction/artifacts/fixtures/ffhq256_inpainting_ddim100_eta1_dsg_v1/
reproduction/artifacts/runs/dsg/ffhq256_inpainting_ddim100_eta1_dsg_v1/
  ddim100-official-fixed-v1/
```

## 12. 完成检查表

一个新算法只有在以下项目全部完成后才算对齐：

- [ ] 原始算法提交、兼容提交和 checkpoint SHA256 已固定。
- [ ] setting 完整记录算法、task、model、sampler、随机性、`noise_level` 和门限。
- [ ] fixture 保存全部真实输入、physics 状态、测量噪声、`x_init` 和随机 tape。
- [ ] 算法实现进入 DeepInv 原有模块并完成公开导出。
- [ ] 单步公式测试和一条固定 tape 的端到端测试通过。
- [ ] 单样本同型号 GPU 跨仓库对齐通过。
- [ ] 完整样本集的 raw tensor、轨迹和图像指标门限通过。
- [ ] run manifest 证明两个实现使用相同 fixture/checkpoint、相同软件环境和同型号
      GPU；设备序号与 UUID 已记录但不作为 gate。
- [ ] 大 artifact 留在 `$HOME` 下的 Git 忽略目录，认证 JSON 已提交。
- [ ] 算法分支已推送，并 fast-forward 合并到 `research/reproduction`。
