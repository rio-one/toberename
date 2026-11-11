# OP-Recon 相对 DP-Recon 的改进与实现说明

本文依据本目录中的实际代码编写，说明 OP-Recon 如何在 DP-Recon 基础上改进几何优化、纹理优化与训练射线采样。实现已经完成轻量冒烟测试，尚未进行完整训练、完整扩散链路测试或最终指标评测，因此文中的收益均为设计目标，不代表已经验证优于 DP-Recon 或 ObjectSDF++。

## 1. 方法概览

OP-Recon 的目标是：保留扩散先验对遮挡区域的补全能力，同时减少扩散更新对重建目标的干扰，并增加小物体和实例轮廓获得监督的机会。

| 比较项 | 本项目拷贝的 DP-Recon 原逻辑 | OP-Recon 新增实现 | 预期作用 |
| --- | --- | --- | --- |
| 几何阶段 | 可见性加权后的 SDS 与重建损失相加，再统一反向传播 | 分别计算重建梯度和 SDS 梯度，按参数组削弱冲突分量 | 减少先验补全对已观测表面及几何约束的扰动 |
| 纹理阶段 | 场景 RGB、背景全景与 SDS 损失加权求和 | 以场景 RGB 与背景全景损失为参考，调整纹理 SDS 梯度 | 减少颜色漂移及生成先验与参考外观的冲突 |
| 射线采样 | 在全图随机选择像素 | 全图均匀、前景实例均衡和实例边界混合采样 | 增加小物体和轮廓的监督机会 |
| RGB 损失 | 对随机采样像素直接取平均 | 按混合采样概率进行重要性校正 | 保持 RGB 项对应全图平均拟合目标 |
| 消融与诊断 | 原训练配置和损失日志 | 三个独立模块开关及梯度冲突日志 | 分析各模块贡献和先验保留程度 |

这些改动保留现有 SDF 表示、扩散模型、可见性网格、训练阶段、优化器及学习率调度。没有新增可训练网络或改变检查点参数形状。原来的数据、射线总数、网格分辨率及显存相关配置保持本目录原值；先前搁置的复现条件统一不属于本次改进。

## 2. 为什么在可见性加权之后增加梯度控制

DP-Recon 已经通过可见性网格降低可见区域上的 SDS 权重，提高遮挡区域上的先验作用。OP-Recon 继续保留这一步。

新增机制针对的是参数共享问题：可见区域和遮挡区域共用 SDF 或纹理网络。即使某次 SDS 更新主要来自遮挡区域，它仍可能通过共享参数改变已观测区域。仅从像素可见性判断，无法直接得知当前先验更新是否与重建目标相冲突。

OP-Recon 因而在参数梯度层面增加一道控制：先检测两个目标的梯度方向，再只削弱 SDS 中与参考目标冲突的部分。可见性加权和梯度控制分别作用于像素权重与参数更新方向。

## 3. 公共机制：观测保护的梯度投影

实现文件：[code/utils/op_gradients.py](code/utils/op_gradients.py)。训练接入位于 [code/training/DPRecon_train.py](code/training/DPRecon_train.py) 的 `_op_backward`。

### 3.1 分别计算两类梯度

对于当前参数组，定义参考损失和先验损失的梯度：

$$
g_r = \nabla_\theta L_{\mathrm{ref}},\qquad
g_d = \nabla_\theta L_{\mathrm{prior}}.
$$

其中，先验损失已经包含当前阶段的 SDS 系数和原有可见性加权，不会在投影时重复施加。

若两者内积为负，则沿先验梯度的负方向更新会在一阶近似下增大参考损失，因而视为冲突。对有效梯度组采用：

$$
\widetilde g_d = g_d - \eta
\frac{\min(0,\langle g_d,g_r\rangle)}{\|g_r\|^2}g_r,
\qquad g_{\mathrm{merged}}=g_r+\widetilde g_d.
$$

`train.op_projection_strength` 对应 $\eta$，取值范围为 0 到 1，默认 1。实现用 `eps=1e-12` 判断梯度平方范数是否过小，并在除法中使用下限保护；参考梯度或先验梯度过小时跳过投影。

- 梯度一致：保持原先验梯度。
- 梯度冲突：按强度移除冲突分量，保留其余方向。
- 强度为 1：在参与投影的参数子空间中完整移除反向分量。
- 强度为 0：不改变先验梯度，但仍会分别计算梯度；若要避免额外计算，应关闭对应模块开关。
- SDS 未连接到的参数张量不会因投影获得新的先验更新。
- 冻结参数不参与计算；两类损失均未使用的参数保留 `grad=None`。

这里的“参与投影的参数子空间”由本次 SDS 实际连接到的参数张量组成。实现按组统计内积与范数，没有构建像素与参数的一一对应关系，也没有逐像素冻结几何。

### 3.2 参数分组与更新

几何阶段分别处理 SDF 编码参数和 SDF MLP 参数；使用纯 MLP 时处理其网络参数。纹理阶段分别处理纹理位置编码和纹理 MLP。若还有不属于这些命名分组、且确实被 SDS 连接的可训练参数，公共函数会放入 `other` 组处理。

分组使不同参数块分别判断冲突，避免单一整体内积掩盖局部参数组的相反方向。合并后的梯度写入参数 `.grad`，再由原有 Adam 执行 `step()`。

投影控制的是原始梯度方向。Adam 包含动量和自适应缩放，因此不能据此保证实际有限步更新必然降低重建损失，也不能保证最终几何或渲染指标必然提高。

## 4. 改进一：几何阶段冲突控制

接入位置：`DPReconTrainRunner._op_backward` 的 `geometry` 分支，以及几何训练循环。

### 4.1 与原方法的区别

原逻辑直接对“重建损失 + 加权 SDS”做一次反向传播。新增逻辑先保留两条独立计算图：

$$
L_{\mathrm{ref}}^{g}=L_{\mathrm{reconstruction}},\qquad
L_{\mathrm{prior}}^{g}=\lambda_{\mathrm{SDS}}L_{\mathrm{SDS}}^{g}.
$$

`L_reconstruction` 包括当前启用的 RGB、深度、法线、Eikonal、平滑、实例监督、物体区分及背景相关约束。它是现有重建与正则目标的组合，并非仅有直接观测误差。

[code/model/loss.py](code/model/loss.py) 新增 `reconstruction_loss` 输出，将不含 SDS 的参考损失单独提供给训练器。实现从重建分支直接构建该项，没有通过“总损失减 SDS”来取得参考计算图。

### 4.2 执行流程

1. 先执行原有前向渲染、可见性加权和损失计算。
2. 当 `train.op_geometry_conflict=True` 且 `begin_prior=True` 时，分别计算参考梯度和 SDS 梯度。
3. 在 SDF 编码与 MLP 等参数组内检测冲突，调整先验梯度。
4. 合并梯度后执行原优化器更新，并记录冲突统计。
5. 尚未开始先验阶段，或关闭开关时，继续走原有单次反向传播路径。

目标是兼顾遮挡补全与可见几何保真。若参考几何本身存在偏差，较强保护也可能限制正确补全，因此需要通过消融选择保护强度。

## 5. 改进二：纹理阶段冲突控制

接入位置：`DPReconTrainRunner._op_backward` 的 `texture` 分支，以及纹理优化循环。

### 5.1 参考目标与先验目标

纹理阶段采用：

$$
L_{\mathrm{ref}}^{a} =
\lambda_{\mathrm{scene}} L_{\mathrm{scene\ RGB}} +
\lambda_{\mathrm{pano}} L_{\mathrm{background\ panorama}},
\qquad
L_{\mathrm{prior}}^{a}=\lambda_{\mathrm{SDS\ RGB}}L_{\mathrm{SDS}}^{a}.
$$

先分别求梯度，再对纹理位置编码和纹理 MLP 进行冲突投影。原有权重、整图渲染和冻结几何的训练安排继续保留。

### 5.2 如何改善外观优化

当 SDS 建议的颜色变化与现有外观参考相冲突时，投影削弱这部分变化，同时保留其余先验更新。预期可以减少颜色漂移，并保留未观测区域的纹理补全能力。

这里的参考不仅包含训练 RGB，也包含原方法生成的 NeRF 锚点图像与背景全景监督，其中存在伪监督或生成内容。因此，“观测保护”是机制名称，不能理解为所有参考像素都是真值；错误参考仍可能被保护。

## 6. 改进三：实例与边界均衡采样

实现文件：[code/utils/op_sampling.py](code/utils/op_sampling.py)。数据集接入位于 [code/datasets/scene_dataset.py](code/datasets/scene_dataset.py)。

### 6.1 改变射线预算的分配

均匀随机采样下，小物体获得的射线数与其像素面积成正比。OP-Recon 保持射线总数不变，默认采用：

| 来源 | 比例 | 每步 256 条射线时 | 选择方式 |
| --- | --- | --- | --- |
| 全图均匀 | 50% | 128 | 在当前图像所有像素中均匀抽取 |
| 实例均衡 | 25% | 64 | 先均匀选当前可见前景实例，再均匀选该实例的像素 |
| 边界附近 | 25% | 64 | 在训练实例标签的边界带中均匀抽取 |

比例对应固定数量的来源配额；实例之间为随机均匀选择，不保证每一步每个实例获得完全相同的射线数。实际实例与边界数量由 `int(N * fraction)` 取整，其余配额归均匀采样。

边界由相邻像素的实例标签差异得到，并用默认 `5×5` 核膨胀成边界带。只使用训练实例掩码，在每个视图首次需要时缓存像素池。

没有前景实例或没有边界时，对应配额回退到全图均匀采样。新采样器使用有放回抽样，允许重复射线；这是与原有 `randperm` 无放回抽样的一个区别。

### 6.2 RGB 采样概率校正

混合采样提高了小物体和边界像素的采样概率。如果直接平均 RGB 误差，会改变全图颜色拟合的权重分配。

设全图有 $M$ 个像素，本步总采样数为 $N$，三类实际配额为 $n_u,n_i,n_b$。混合概率为：

$$
q(p)=\frac{n_u}{N}\frac{1}{M}
+\frac{n_i}{N}q_i(p)
+\frac{n_b}{N}q_b(p).
$$

若当前有 $K$ 个可见前景实例，像素 $p$ 属于实例 $k$，且该实例有 $A_k$ 个像素，则 $q_i(p)=1/(K A_k)$；背景像素的实例分支概率为 0。边界带中像素的 $q_b(p)$ 为边界像素总数的倒数，带外为 0。

返回每条射线的权重：

$$
w(p)=\frac{1/M}{q(p)},\qquad
\widehat L_{\mathrm{RGB}}=\frac{1}{N}\sum_{j=1}^{N}w(p_j)\ell_{\mathrm{RGB}}(p_j).
$$

该校正适用于当前固定配额混合、有放回抽样方案，使 RGB 平均损失在采样期望上对应全图均匀平均。实现没有再除以本批权重之和，支持现有的 L1 和 MSE RGB 损失。

数据集通过 `rgb_sampling_weight` 传递权重，由 `MonoSDFLoss.get_rgb_loss` 应用。实例、深度、法线等其他损失保持原有计算和归约方式，因此其采样分布会受到均衡策略影响；不能声称整个总损失都被校正为原来的目标。

### 6.3 作用范围

新采样器用于常规随机射线训练分支；整图读取、评估和纹理阶段的整图渲染不变。关闭采样开关时使用原采样路径。未新增边界平滑损失，也未修改原深度损失中的随机射线重排行为。

## 7. 配置开关和消融

三个现有配置均默认启用新模块：

- [code/confs/replica_grid.conf](code/confs/replica_grid.conf)
- [code/confs/scannetpp_grid.conf](code/confs/scannetpp_grid.conf)
- [code/confs/youtube_grid.conf](code/confs/youtube_grid.conf)

| 配置键 | 默认值 | 含义 |
| --- | --- | --- |
| `train.op_geometry_conflict` | `True` | 几何阶段冲突控制 |
| `train.op_texture_conflict` | `True` | 纹理阶段冲突控制 |
| `train.op_projection_strength` | `1.0` | 投影强度，范围 `[0, 1]` |
| `dataset.op_balanced_sampling` | `True` | 混合采样 |
| `dataset.op_instance_fraction` | `0.25` | 实例采样比例 |
| `dataset.op_boundary_fraction` | `0.25` | 边界采样比例 |
| `dataset.op_boundary_width` | `5` | 边界膨胀核边长，正奇数 |

实例比例与边界比例必须非负且总和小于 1，以保留覆盖全图的均匀分支。没有新增键的旧配置默认关闭三个新模块。

建议按以下设置进行消融；此表为实验建议，本次文档工作没有执行训练：

| 设置 | 几何控制 | 纹理控制 | 混合采样 |
| --- | --- | --- | --- |
| 原逻辑参照 | False | False | False |
| 仅几何保护 | True | False | False |
| 仅纹理保护 | False | True | False |
| 仅采样 | False | False | True |
| 完整 OP-Recon | True | True | True |

三个开关关闭时恢复原训练路径，实验名称仍为 `oprecon_grid_*`。这里指算法路径回退，不承诺浮点逐位一致、随机序列一致或完整训练数值完全复现。

## 8. 代码改动对应关系

| 文件 | 具体职责 |
| --- | --- |
| [code/utils/op_gradients.py](code/utils/op_gradients.py) | 分别求梯度、冲突检测、分组投影、梯度合并与统计 |
| [code/utils/op_sampling.py](code/utils/op_sampling.py) | 当前视图的实例/边界像素池、混合采样与 RGB 重要性权重 |
| [code/datasets/scene_dataset.py](code/datasets/scene_dataset.py) | 配置新采样器，按当前视图抽样并返回 RGB 权重 |
| [code/model/loss.py](code/model/loss.py) | RGB 加权及独立的非 SDS 参考损失输出 |
| [code/training/DPRecon_train.py](code/training/DPRecon_train.py) | 两阶段冲突控制接入、独立开关和日志 |
| `code/confs/*_grid.conf` | 新模块默认参数及独立实验名称 |
| [scripts/oprecon_smoke.py](scripts/oprecon_smoke.py) | 轻量验证脚本 |
| `.gitignore` | 忽略本目录下的冒烟缓存 `.oprecon_cache/` |

## 9. 日志、分布式支持与代价

新增梯度日志如下：

| 名称 | 含义 |
| --- | --- |
| `op_conflict_rate` | 当前步内，有效参数组中内积为负的比例，不是跨整个训练过程的累计比例 |
| `op_gradient_cosine` | 有效参数组中参考梯度与 SDS 梯度的平均余弦相似度 |
| `op_prior_retention` | 投影后 SDS 梯度范数与投影前范数的比值；无有效先验范数时返回 1 |

几何阶段日志同时进入现有损失日志渠道，纹理阶段输出到控制台。

几何受控分支将前向和梯度计算放入 DDP `no_sync()`，多 rank 时由公共函数在投影前分别平均两类梯度。代码包含该同步逻辑，但已有冒烟只验证了单 rank，不能当作多卡训练已经验证。

分别求梯度需要保留计算图并暂存两组梯度，会带来额外时间和显存开销。采样器还会缓存当前视图的像素池。未新增神经网络不代表训练完全没有额外资源代价，实际开销需在完整训练中测量。

## 10. 已有轻量验证及其边界

既有记录：[oprecon_smoke.log](oprecon_smoke.log)。测试使用 `oprecon` conda 环境，已通过：

- 修改文件的 Python 语法和三个配置解析。
- 少量 Replica scan1 真实采样的射线数量、标签范围、RGB 权重和批次拼接。
- 缺少实例/边界时的回退、采样关闭路径和整图路径。
- 冲突、同向、零参考梯度、参数不相交及投影强度为零等情况。
- 单 rank DDP `no_sync` 接线与优化器更新。
- 使用实际 `DPReconLoss`、实际训练器梯度辅助方法和小型网络进行几何更新，包含法线相关高阶梯度。
- 小型纹理网络更新，以及该测试中被冻结的几何参数保持不变。

SDS 使用简化替代损失，没有加载完整预训练扩散模型或完整训练器。没有完整训练、网格导出、最终指标评测或多 rank 测试。因此还不能给出 CD、F-score、PSNR、SSIM、LPIPS 或 MUSIQ 的改进幅度。本次仅补充文档，没有重新运行这些测试，也没有进行哈希验证。

如需手动重新执行轻量测试，在当前项目根目录运行：

```bash
conda activate oprecon
CUDA_VISIBLE_DEVICES=4 PYTHONDONTWRITEBYTECODE=1 python -B scripts/oprecon_smoke.py
```

GPU 编号应替换为当时空闲的设备。脚本将缓存重定向至本项目 `.oprecon_cache/`。

## 11. 手动训练示例

以下命令仅供后续手动使用，本次文档更新没有启动训练。

```bash
conda activate oprecon
cd ~/wangruiyao/OP-Recon/code
CUDA_VISIBLE_DEVICES=4 torchrun --nproc_per_node=1 training/exp_runner.py --conf confs/replica_grid.conf --scan_id 1 --prior_yaml geometry.yaml --none_wandb
```

几何训练完成后，将 `GEOMETRY_RUN` 替换为新生成的 OP-Recon 几何实验目录：

```bash
CUDA_VISIBLE_DEVICES=4 torchrun --nproc_per_node=1 training/exp_runner.py --conf confs/replica_grid.conf --scan_id 1 --prior_yaml texture.yaml --is_continue --ft_folder ../exps/oprecon_grid_replica_1/GEOMETRY_RUN --checkpoint 7500 --none_wandb
```

`7500` 是当前十视图 scan1 训练安排的示例 epoch，应以实际生成的检查点为准。

## 12. 方法表述与后续研究

当前实现可以表述为：在分解式神经重建中，保留原可见性引导，通过分阶段、分参数组的梯度冲突控制协调重建与生成先验，并使用实例与边界混合采样改善监督分配。

梯度投影、重要性采样和实例均衡均有已有研究基础。现阶段不能直接将这些通用技巧声明为全新原理。方法的研究价值应通过以下证据建立：可见性加权后是否仍有显著冲突；投影是否同时保留补全能力并改善几何和外观；混合采样是否在多个场景上稳定增加收益；收益与新增资源开销是否合理。
