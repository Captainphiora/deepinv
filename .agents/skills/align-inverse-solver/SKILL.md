---
name: align-inverse-solver
description: 在 DeepInv 中复现扩散模型或 flow matching 的反问题求解算法，并与其原始仓库进行数值精度对齐。适用于算法移植、固定随机性的跨仓库比较和对齐回归；不适用于没有参考实现的普通 DeepInv 功能开发。
---

# 对齐反问题求解算法

把已记录提交版本的原始仓库视为行为基准。在不改变参考公式、时间调度、张量约定和更新顺序的前提下复用 DeepInv 的现有抽象；只添加实现数值等价所必需的最小兼容层。

## 修改代码前

1. 记录两个仓库的提交、原始命令与配置、模型权重标识、设备、数据类型和依赖环境。
2. 追踪参考实现的完整路径：模型参数化、时间网格、反向更新、条件梯度、物理算子、噪声、预处理和输出转换。
3. 明确预期的等价边界。不得为了提高指标而暗中修改参考算法。

## Python 环境

参考实现和 DeepInv 必须使用各自仓库的独立 uv 环境，不能用 DeepInv 的环境运行
参考算法。每条 Python 命令都在同一条 shell 命令中先显式进入该命令所属项目根目录，
再通过个人目录的 uv wrapper 启动；不得依赖执行工具单独传入的工作目录：

```bash
# DeepInv 实现、公共比较器和可视化
cd /mnt/afs/L202500464/deepinv && \
  /mnt/afs/L202500464/uv-env-tool.sh --proxy off uv run --no-sync python ...

# 参考实现；把路径替换成当前算法的官方仓库
cd /mnt/afs/L202500464/DiffPIR && \
  /mnt/afs/L202500464/uv-env-tool.sh --proxy off uv run --no-sync python ...
```

命令必须解析到所属项目的 `.venv/bin/python*`。若锁定源码位于 detached worktree，
可以使用该算法主 checkout 的 uv 环境执行 worktree 中的源码，但必须分别记录环境
项目路径、实际源码路径和源码 commit。生成官方 measurement 或 kernel 的 fixture
步骤也属于参考侧；统一指标计算属于比较侧，应在 DeepInv 环境中对两边已保存的
tensor 一次性计算。

只有对应项目的锁文件或依赖声明发生变化时才运行 `uv sync`；缺少依赖时必须在依赖
所属项目中用 `uv add` 写入，不能临时 `pip install`，也不能把包加到另一个项目来
迁就当前命令。各项目复用个人目录的 uv 下载缓存，但不得共用 `.venv`。`uv run`
暂时没有输出时先检查其进程或继续等待，不能据此判定缺包、重建环境或切换到系统
Python。

CUDA 正式运行若启用 `torch.use_deterministic_algorithms(True)`，必须在启动 Python
前导出 `CUBLAS_WORKSPACE_CONFIG=:4096:8`；不同 Torch 版本可能对缺少该变量表现为
报错或静默继续，不能依赖后者。

## 确定性产物约定

使用保存于 CPU 的 `.pt` 文件作为唯一数值真值；文件中只存张量和基础容器。PNG 等图像只作为预览，不得作为精度比较依据。

每个样本必须在任一实现运行前保存或加载全部随机输入：

- 原始图像和经过预处理的模型输入；
- measurement 以及所有实际使用的物理参数，包括 mask 和 kernel；
- 实际采样出的测量噪声及其配置项 `noise_level`，即使后者为零也必须保留；
- 初始潜变量或噪声 `x_start`；
- 随机采样器每一步所需的随机增量。

固定 Python、NumPy、Torch CPU 和所有 CUDA 设备的随机种子，但不能把“种子相同”当作“输入相同”：不同仓库可能以不同顺序消耗 RNG。两个实现都应直接接收已保存的张量，不得在某个 runner 内重新生成 mask、measurement、噪声或初始点。

初始噪声、测量噪声和逐步 transition noise 必须使用明确且相互独立的随机流；使用
独立生成器时 seed 也必须不同。fixture 创建后至少检查初始噪声不与首个 transition
noise 逐元素相同，并把 stream policy 与各流 seed 写入 manifest。

在张量旁保存 JSON manifest，记录路径、SHA-256、形状、数据类型、随机种子、完整配置、两个仓库提交、模型权重哈希及运行环境。被 Git 跟踪的配置使用仓库相对路径，并允许在运行时指定 artifact 根目录。

## 对齐循环

1. 先只运行一个样本和一个 setting。
2. 从最早的公共状态开始比较，逐步推进到第一个发生偏差的位置；在该边界保存有明确名称的 `.pt` 快照。
3. 张量比较至少报告 `max_abs`、`mean_abs`、RMSE 和相对 L2。PSNR、SSIM、LPIPS 是结果指标，不能单独证明实现等价。
4. 修改代码前先归类偏差：输入、模型适配、schedule/timestep、RNG、physics/noise、条件更新顺序、精度/设备或序列化。
5. 修复最早的根因，重新运行小样本；通过后再运行完整实验集。

精度认证使用各仓库锁定的独立依赖环境，并在同型号 GPU 上跨仓库比较 DeepInv 与
原始实现；Torch、CUDA、cuDNN、NumPy 等版本差异必须记录但不要求相同。同型号的
不同 GPU 可以并行运行。GPU 序号和 UUID 只记录为 provenance，不作为通过条件。
CPU/GPU 跨设备一致性不属于验收项。若结果超出显式容差，先把失败 case 的两个实现
放到同一张 GPU 上重跑；仍失败时可以增加“同一依赖环境”的诊断运行，以区分硬件、
依赖和实现偏差，但诊断运行不能替代各自独立环境的正式认证，也不得通过放宽容差
掩盖系统性漂移。

## 项目布局

可复用的 DeepInv 实现和单元测试放在包原有目录。跨仓库 runner、配置、manifest、报告及忽略的输出统一放在 `reproduction/`。每次实验使用稳定的实验 ID；每个正式结果对应一个不可变配置。大文件统一放入 Git 忽略的 `reproduction/artifacts/`。

使用长期集成分支 `research/reproduction`，每个算法使用短期分支 `repro/<algorithm>-alignment`。完成一次对齐的提交必须同时包含算法实现、确定性回归测试、精确 setting，以及注明参考提交和实际容差的报告。

## 完成条件

只有在固定输入实验可重复、中间状态满足声明容差、最终张量与指标均已记录、`noise_level` 得到保留，并且另一位执行者仅依赖已跟踪文件即可复现命令（不依赖 `/tmp` 或未记录的本地状态）时，才能宣称对齐完成。
