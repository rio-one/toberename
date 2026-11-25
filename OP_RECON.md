# OP-Recon v6：统一光线实例归属与先验尺度一致性

本节记录 2026-11-21 部署，优先于后面的 v5/v4/v3/v2 历史说明。本轮按用户确认，以第四版坐标残差 SDF 为基础，仅实施此前方案的第一项“统一场景与实例渲染概率”和第三项“先验渲染尺度一致性”。**第二项“观测约束补全”不实施。**

全部修改与临时测试限定 `/home/seed/wangruiyao/OP-Recon`；全部 Python 执行使用 `/home/seed/miniconda3/envs/oprecon/bin/python`，带 `-B` 和 `PYTHONDONTWRITEBYTECODE=1`。当前仅 Replica 配置启用，限定单卡、batch_size=1；ScanNet++/YouTube 配置本轮未修改。

## 第四版基础与本轮目标

第五版的基础 SDF/门控/粗哈希特征条件输入已通过 `op_completion_conditioned=False` 关闭。保留第四版坐标输入的小型残差、共享现有 VisGrid、门控阈值 0.1/0.5、最大残差 0.2、35000 步后激活和 SDS 参数叶子隔离。scan1 的残差参数仍为1067个，没有增加网格、网络头或模型宽度。

v3 辅助机制保持关闭；v2纹理冲突控制、基础损失修正和均衡采样保留。既有背景过滤/连通块策略及上一轮文件交接修复保留；未新增自由空间分类、观测证据门控、可见性更新或跨视角辅助机制。

## 第一项：共享场景命中贡献与互斥实例归属

场景 SDF 继续采用实例 SDF 的硬最小值；原有场景密度、误差界射线采样、RGB/深度/法线渲染权重和网格导出字段保持一致。改变的是实例归属的生成方式：

```
f_scene(x) = min_i f_i(x)
q_i(x) = softmax_i(-f_i(x) / stopgrad(beta))
O_i(ray) = sum_k w_k(ray) * q_i(x_k)
sum_i O_i(ray) = sum_k w_k(ray)
```

- `w_k` 为原有场景体渲染命中权重；`beta` 复用现有 Laplace 密度厚度参数，不新增可训练温度。
- 只对归属分布使用的温度 stopgrad；实例 SDF 不 detach。beta 仍通过原有场景命中权重接受梯度，避免实例归属通过直接调窄温度降低损失。
- 每个采样点的实例归属和为1，实例贡献相加严格对应同一份场景命中贡献（浮点误差除外），不再分别计算各实例 alpha 后重复使用场景透射率。
- 使用原有实例 one-hot BCE 损失，无新增监督项。训练用的 `object_opacity` 与输出/评估 Mask 用的 `semantic_values` 共用同一结果。
- 不按整条光线重新归一化实例贡献：低场景不透明度仍受实例监督惩罚，不把未命中的光线伪装成高置信度预测。
- 显式单物体渲染 `render_obj_idx` 时，仅将该物体的命中贡献分配给指定实例。
- 背景平滑辅助分支及隐式网络内部历史语义返回接口保持原行为；本轮统一的是主训练/推理光线上的实例概率。

这个改动使实例监督和RGB/深度监督使用相同的命中位置，仍不构成实例不相交的硬约束。原碰撞正则继续存在。由于场景字段仍采用硬最小值，不能宣称本轮改成了密度求和或新的软并集场。

## 第三项：完整坐标链中的SDF/密度尺度一致性

前景先验渲染存在两步坐标变换：

```
u in [-r,r]^3
y = (u + r) / (2*r)          # prior coordinates -> [0,1]
x_scene = center + (y - 0.5) / s
s = shape_init_params / L
a = 2*r*s
```

L 为现有 bbox 去掉每侧0.1 padding 后的最大边长。故 `u=a*(x_scene-center)`，真实先验距离尺度是 `a`。旧实现只把 SDF 乘 s，且直接使用场景密度；这既遗漏了2*r的距离因子，也未调整密度厚度。

启用修正后：

```
f_prior(u) = a * f_scene(x_scene)
sigma_prior(f_prior) = sigma_scene(f_prior/a) / a
dt_prior = a * dt_scene
sigma_prior * dt_prior = sigma_scene * dt_scene
```

对项目的 Laplace 密度，这等价于 `beta_prior=a*beta_scene`。SDF零等值面不因单位换算而改变，法线有限差分与最终字段使用相同单位；没有改成detach空间梯度。

背景先验没有上述物体归一化，使用a=1。尺度每次根据当前对象bbox求取，避免前景/背景或不同尺寸对象切换时使用过期值；所有对象遵循同一公式。此模式限定有界、零SDF偏置的现有配置。

实现通过同一个 `PseudoVolsdfDensity` 包装器处理最终alpha、占据网格更新、采样剪枝及importance proposal的密度查询，没有只在最终图像上做后处理。先验采样数和扩散调用次数未增加。占据网格阈值剪枝可能受数值尺度改变影响；微型光学厚度一致性验证不等于完整训练结果保证。

## 开关、检查点和修改范围

| 配置键 | Replica当前值 | 说明 |
| --- | --- | --- |
| model.op_unified_rendering | True | 第一项：共享命中权重的实例归属 |
| model.op_prior_scale_consistency | True | 第三项：先验距离/密度单位一致 |
| model.implicit_network.op_completion | True | 保留第四版残差核心 |
| model.implicit_network.op_completion_conditioned | False | 关闭第五版条件输入 |
| train.op_v3_enabled | False | 不恢复第三版辅助机制 |
| train.op_geometry_conflict | False | 保留第四版SDS参数隔离 |
| train.op_texture_conflict | True | 保留v2纹理路径 |

两项新开关在未声明的配置中均默认False，可独立关闭做从头训练消融。`op_completion_feature_levels` 为第五版遗留配置，条件输入关闭时不参与计算；本轮没有新增哈希条件。

模型检查点新增一个2字节 `op_rendering_signature` buffer，记录两个开关，不是参数或文件校验哈希。训练器在续训/接入纹理前检查模式是否一致；旧v4/v5检查点缺少该标志，不能用于当前模式续训。旧结果和检查点未被改写。用户应从头训练v6几何，之后以同一配置加载对应v6几何训练纹理。

本轮修改文件：

| 文件 | 作用 |
| --- | --- |
| code/model/op_rendering.py（新增） | 采样点归属与共享权重实例贡献 |
| code/model/network.py | 接入主训练/Mask路径，传递尺度开关，保存模式标志 |
| code/prior.py | 仅按开关启用先验坐标修正 |
| code/threestudio/models/geometry/implicit_sdf.py | 计算包含2*r的真实先验SDF尺度 |
| code/threestudio/models/renderers/neus_volume_renderer.py | 共用密度接口进行单位换算 |
| code/training/DPRecon_train.py | Replica/单卡范围检查、日志、检查点模式兼容检查 |
| code/confs/replica_grid.conf | 启用第一/第三项，关闭第五版条件输入 |
| OP_RECON.md | 本轮实现与验证记录 |

没有修改残差解码器代码、数据、既有实验文件、指标、纹理损失、评估器或其他项目/环境。仓库内此前已有未提交改动均保留。

## 轻量冒烟结果与限制

仅在oprecon环境执行，测试通过标准输入运行：

- 修改代码语法、三份配置解析，确认仅Replica启用、第四版残差参数1067、第五版条件输入与v3关闭。
- 共享贡献守恒、完全重叠实例均分、零不透明度、单实例、单物体渲染、空批次、极端SDF数值以及归属梯度有限；确认温度不直接接受归属分支梯度。
- 使用实际PseudoSDF、坐标归一化函数、ImplicitSDF查询/有限差分法线和密度包装器，检查不同物体尺寸与先验半径下的SDF单位和光学厚度不变性；前景/背景切换、旧模式及尺度梯度通过。
- 实际训练器检查点模式检查接受匹配配置，拒绝缺失或不同模式。
- 空闲GPU0使用实际Replica哈希/SDF/颜色网络、DPReconNetwork.forward、DPReconLoss与缩小的原误差界采样器，16条合成射线、两次微型Adam更新通过；训练概率守恒，推理Mask与单物体渲染也通过。
- 上述GPU测试的先验使用实际尺度包装器和小型可微代理，确认先验梯度只直接更新第四版残差；没有加载或执行预训练扩散模型、真实SDS或完整先验渲染器。
- 通过AST载入实际网络/适配器方法以避开完整训练器初始化，直接复用既有 `code/tmp_build/_hash_encoder.so`，没有运行编译器。测试不代表NCCL/DDP完整训练入口已验证。

不新增网格、射线、网络头、扩散模型调用或独立参考反传。归属计算替换原有逐实例密度/alpha张量运算；尺度换算仅为小量标量与逐点运算。**本轮未测量完整训练峰值显存或耗时，不承诺零开销。**

本轮临时bbox目录和文件已清理，没有新增保留的测试脚本、检查点、日志、缓存、哈希或校验文件；没有安装/升级依赖，没有执行完整训练、完整评估或现有实验重算。冒烟仅证明小规模执行与相关数学性质，不证明指标必然改善。

## 用户后续从头训练

以下命令仅供用户手动执行，本轮没有启动；GPU编号按实际空闲资源选择：

```bash
conda activate oprecon
cd /home/seed/wangruiyao/OP-Recon/code
CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 training/exp_runner.py --conf confs/replica_grid.conf --scan_id 1 --prior_yaml geometry.yaml --description v6_geometry --none_wandb
```

几何完成后，将V6_GEOMETRY_RUN替换为实际目录：

```bash
CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 training/exp_runner.py --conf confs/replica_grid.conf --scan_id 1 --prior_yaml texture.yaml --is_continue --ft_folder ../exps/oprecon_grid_replica_1/V6_GEOMETRY_RUN --checkpoint latest --description v6_texture --none_wandb
```

优先用同一评估口径比较v4、只开第一项、只开第三项及两项共同启用的结果。分别报告物体Acc/Comp/CD/F-score、场景和背景指标、NeRF与网格渲染；背景后处理/导出分辨率差异仍需单独注明。当前尚无v6完整训练结果，不能宣称已优于v4、v5或DP-Recon。


## 2026-11-23 第六版评估命令

本节记录 tmux 0 评估报错后的正确命令。此前把占位符 `DATASET` 原样传入，脚本无法匹配 Replica 实验，出现 `No objects/scenes to average`、`No DATASET experiments found under exps`。本次数据集名称应填写 `replica`，无需为该参数错误重新训练。

第六版已完成纹理训练的目录为：

```text
/home/seed/wangruiyao/OP-Recon/exps/oprecon_grid_replica_1/v6_texture_2026_11_23_07_32_30
```

先进入环境与项目根目录，再逐项执行：

```bash
conda activate oprecon
cd /home/seed/wangruiyao/OP-Recon

# 1. 总场景几何
python -B eval/eval_scene_recon.py --dataset_type replica

# 2. 物体级几何
python -B eval/eval_object_recon.py --dataset_type replica

# 3. 背景几何
python -B eval/eval_bg_recon.py --dataset_type replica

# 4. 参考图渲染指标：PSNR、SSIM、LPIPS，同时输出 NeRF 与 nvdiffrast
python -B eval/eval_rendering.py --dataset_type replica \
  --run_dir /home/seed/wangruiyao/OP-Recon/exps/oprecon_grid_replica_1/v6_texture_2026_11_23_07_32_30

# 5. 无参考 MUSIQ：仅本条命令隐藏 GPU，使用 CPU
CUDA_VISIBLE_DEVICES="" python -B eval/eval_musiq.py   --dataset_type replica \
  --run_dir /home/seed/wangruiyao/OP-Recon/exps/oprecon_grid_replica_1/v6_texture_2026_11_23_07_32_30

# 6. 实例 Mask IoU
python -B eval/eval_mask.py --dataset_type replica
```

渲染与 MUSIQ 使用 `--run_dir` 明确指定第六版纹理目录；MUSIQ 的临时环境变量不影响后续 Mask 命令的 GPU 可见性。当前总场景、物体、背景和 Mask 入口按实验目录名称排序选最后一个，当前会选中上述第六版纹理目录；以后新增或重命名实验目录时，应核对实际选中的输入，不能把这些入口理解为固定绑定本目录，也不要给它们传入尚未支持的 `--run_dir`。

本次仅补充文档，未执行上述评估、未修改训练或评估代码。

---

# OP-Recon v5：局部几何条件化的轻量残差 SDF

本节记录 2026-11-18 本轮改进，优先于后面的 v4/v3/v2 历史说明。用户确认：几何优先，保留 v4 的 Acc/CD/F-score 收益并重点改善完整度；保留轻量残差 SDF 总体结构；新几何从头训练。修改与临时操作范围仅为 `/home/seed/wangruiyao/OP-Recon`，全部 Python 执行仅使用 `/home/seed/miniconda3/envs/oprecon/bin/python`，带 `-B` 和 `PYTHONDONTWRITEBYTECODE=1`。

**仅 Replica 默认启用，仍限定单卡、batch_size=1、Grid_MLP。** 本轮未修改 ScanNet++/YouTube 配置、纹理路径、损失函数、采样预算、导出/评估脚本、背景 TSDF/连通块策略或任何按物体设置的规则。上一轮背景文件交接修复保留。

## 只读诊断与本次选择

诊断读取 v4 几何实验 `exps/oprecon_grid_replica_1/v4_geometry_2026_11_16_23_31_39/checkpoints/ModelParameters/latest.pth`（epoch 7500），不更新权重或检查点。CPU 上抽取每个前景网格 128 个顶点，合计 1280 个表面点；另取其 0.03 尺度随机扰动邻域点 1280 个、均匀体积点 1024 个。对表面与邻域用 eps=0.001 有限差分额外查询附近位置；没有反向传播、GT 匹配、完整指标计算或扩散/渲染调用。

- 表面样本的门控关闭比例约 36.64%，全开比例约 27.81%；这一比例本身不能证明门控过强，可见表面应当受到保护。
- 表面对应实例的未门控残差，幅度超过上限 95% 的比例为 0；邻域约 0.16%。没有证据支持直接把幅度从 0.2 扩大。
- 在 454 个表面门控过渡点上，乘法边界项 `delta * grad(gate)` 的范数中位数约 1.44、90 分位约 17.34；`gate * grad(delta)` 对应约 0.055、0.382。邻域样本也呈现边界项占主导的情况。
- 这些是残差分支的局部有限差分诊断，不是完整合成 SDF 的梯度统计；基础场可能部分抵消该项。结果提示门控边界适应能力值得改进，尚不能证明它是完整度误差的唯一来源。

v4 的小型解码器只看坐标，缺少直接的可见性与局部几何输入。本轮仅改这个核心表达：让同一个残差解码器获得已有观测分支的局部条件。门控阈值、残差幅度、SDS 参数隔离路径均保留；不叠加新辅助损失或放开可见区域。

## 核心表示

保持最终字段：

```
f_i(x) = f_obs_i(x) + g(V(x)) * 0.2 * tanh(R_i(z(x)))
z(x) = [PE_3(x), g(V(x)), tanh(f_obs(x) / 0.2), h_coarse(x)]
```

- `PE_3` 为原有 21 维坐标/位置编码。
- `f_obs(x)` 是本次 forward 已算出的全部实例基础 SDF 通道，采用同一规则，包含背景。
- `h_coarse` 复用同一次基础编码查询的前 4 层哈希特征，当前每层 2 维，共 8 维；没有新的网格或第二次编码查询。
- 解码器仍是一个宽度 32 的共享 MLP，Softplus(beta=10)，末层严格零初始化。scan1 的 C=11 时输入为 41 维，输出仍为 11 个实例通道。
- 可见性门控继续复用原 VisGrid，V<=0.1 时全开，V>=0.5 时为零，中间 smoothstep；最大残差仍为 0.2，单位为归一化场景坐标。
- 条件化使解码器有机会学习随门控状态和局部形状调整修正，不自动保证门控边界项降低，更不保证 Comp/Recall 必然提升；仍需从头训练与消融验证。

基础 SDF、粗哈希特征和门控条件均保留正确的空间导数，没有用 detach 制造与真实字段不一致的法线。几何 SDS 前向继续临时冻结基础编码/MLP、密度和辐射颜色参数叶子，只有残差参数直接接收 SDS 更新。非 SDS 损失继续作用于合成场。原有 Eikonal/法线反传包含条件的空间变化，不新增一套梯度投影或辅助反传。前 35000 步仍不激活残差。

## 资源与开关

v5 条件化解码器参数数为 `992 + 65C`（宽度32、3组位置编码、8维粗特征），scan1 为 **1707**，v4 为1067，增加640个参数。新增 FP32 参数/梯度/Adam 两个矩合计约10 KiB；激活与反传开销另计，不能据此声称零额外显存。

条件在原有最多8192点的残差分块内拼接。没有新增训练射线、扩散模型/调用、可见性网格、跨视角查询或完整训练时长配置。

| 配置键 | Replica 当前值 | 含义 |
| --- | --- | --- |
| model.implicit_network.op_completion | True | 保留残差核心 |
| model.implicit_network.op_completion_conditioned | True | 开启本轮局部条件化 |
| model.implicit_network.op_completion_feature_levels | 4 | 复用基础编码前4层；设0可消融哈希条件 |
| model.implicit_network.op_completion_width | 32 | 保持解码器宽度 |
| model.implicit_network.op_completion_max_residual | 0.2 | 保持幅度上限 |
| model.implicit_network.op_completion_hidden_visibility | 0.1 | 保持全开阈值 |
| model.implicit_network.op_completion_visible_visibility | 0.5 | 保持关闭阈值 |
| train.op_v3_enabled | False | 不恢复v3辅助机制 |
| train.op_geometry_conflict | False | 保持几何SDS参数隔离 |
| train.op_texture_conflict | True | 纹理仍用v2路径 |

新条件化开关在未声明的旧配置中默认 False，保留 v4 的21维解码器布局。当前v5配置不能加载v4检查点继续训练；训练器检查残差首层权重形状并明确拒绝混用。读取既有v4结果应使用其匹配的原 `runconf.conf`。v5几何结束后的纹理训练须加载对应v5几何检查点与同一配置；残差仍随几何一起冻结。

## 修改文件与轻量冒烟

本轮仅更新：

| 文件 | 作用 |
| --- | --- |
| code/model/op_residual.py | 残差接收门控、基础SDF、粗特征条件，保留分块与空间导数 |
| code/model/network.py | 复用同次forward结果接入残差；旧配置默认保留v4布局 |
| code/training/DPRecon_train.py | 打印方法版本，检查v4/v5解码器布局兼容性 |
| code/confs/replica_grid.conf | 仅Replica开启局部条件化 |
| OP_RECON.md | 本轮诊断、方法、资源与验证范围 |

轻量冒烟已通过：

- 修改代码语法、三份配置解析，确认新核心只在Replica启用，v3保持关闭。
- 小型残差的零初始化、可见位置严格零修正、条件输入有效性、幅度界限、空点与分块一致性。
- 包含全部条件的真实空间导数与有限差分对齐；Eikonal二阶反传有限，VisGrid无参数梯度。
- SDS代理前向只产生残差参数梯度，观测分支/条件来源参数无直接先验梯度；观测更新仍可训练这些参数。
- 两次小型CPU优化器更新、内存中的v5保存/恢复、旧残差布局保留、v4/v5不匹配明确拒绝。
- 空闲GPU0上使用实际Replica配置的ObjectImplicitNetworkGrid及项目现有CUDA哈希扩展，32点前向、空间二阶反传和一次Adam更新；实际哈希条件路径的SDS参数隔离也通过。测试从源码AST载入真实网络类以避免初始化扩散/训练器；使用既有 `code/tmp_build/_hash_encoder.so`，未调用编译器或更改环境。
- 8192点残差分支的微型前向/空间导数反传比较：相对各自起始分配的增量峰值，v4约17.88 MiB，v5约19.29 MiB，差约1.41 MiB。条件来源为微型可微函数；该结果不含完整渲染/扩散图，不能当成真实训练峰值或耗时结论。

全部测试经标准输入执行，没有新增测试脚本、磁盘检查点、日志或缓存产物；没有安装/升级依赖、哈希验证、完整训练或完整评估。没有运行NCCL/DDP启动入口或真实SDS，不能把本轮冒烟描述为完整训练入口已验证。既有v4实验、检查点、网格和指标保持原样。

## 后续从头训练

以下仅供用户手动执行，本轮没有启动。GPU编号按空闲资源选择：

```bash
conda activate oprecon
cd /home/seed/wangruiyao/OP-Recon/code
CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 training/exp_runner.py --conf confs/replica_grid.conf --scan_id 1 --prior_yaml geometry.yaml --description v5_geometry --none_wandb
```

几何完成后，替换成实际新几何目录：

```bash
CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 training/exp_runner.py --conf confs/replica_grid.conf --scan_id 1 --prior_yaml texture.yaml --is_continue --ft_folder ../exps/oprecon_grid_replica_1/V5_GEOMETRY_RUN --checkpoint latest --description v5_texture --none_wandb
```

优先比较物体Acc、Comp、CD、F-score/Recall和NeRF渲染，确认是否在保住精度的同时改善完整度。v5尚无训练指标，不能宣称优于v4或DP-Recon。关闭 `op_completion_conditioned` 可回到v4残差输入布局，但这个消融仍需从头训练；它不等于原始DP-Recon。跨版本背景后处理、全场景导出分辨率差异仍应与算法收益分开记录。

---


## 2026-11-20 评估目录选择修复

本次仅修改评估入口与文档：`eval/eval_musiq.py`、`eval/eval_rendering.py`，新增共用的 `eval/render_eval_inputs.py`。此前两个入口按实验名的字母顺序选最后一个，导致 `v5_geometry_resume_2026_11_19_18_31_55` 排在实际更新的 `v4_texture_2026_11_20_09_06_23` 后，错误查找几何目录中不存在的网格渲染图片。

两个入口现在都支持 `--run_dir` 指定包含 `plots/` 的实验目录。未指定时按名称末尾的实际时间戳选择最新目录，无时间戳的目录使用修改时间；最新目录若缺输入会明确报错，不静默退回旧实验。两种渲染的图片及GT视角ID在加载指标模型前检查，打印实际选择路径。MUSIQ仍使用原KonIQ模型及按图像平均的评分方式；仅读取PNG并按视角ID匹配，不跳过缺失视角。

本次只在oprecon环境做了轻量语法、选择/路径/视角检查和替代评分函数的冒烟，真实目录各10张图片通过匹配；未加载TensorFlow/MUSIQ/LPIPS模型、未执行实际指标评估。临时检查目录已清理，无新增保留测试脚本。没有改训练代码、检查点、环境、设备权限或进行哈希验证。

用户手动重跑MUSIQ（CPU；此设置只作用于该命令，不改当前shell后续GPU设置）：

```bash
conda activate oprecon
cd /home/seed/wangruiyao/OP-Recon
CUDA_VISIBLE_DEVICES="" python -B eval/eval_musiq.py --dataset_type replica --run_dir /home/seed/wangruiyao/OP-Recon/exps/oprecon_grid_replica_1/v4_texture_2026_11_20_09_06_23
```

结果默认保存于该纹理实验的 `plots/musiq.txt`，包含实际输入路径。渲染评估如需重跑，可用 `python -B eval/eval_rendering.py --dataset_type replica --run_dir` 加同一路径，一次输出NeRF和nvdiffrast两组指标；该入口默认汇总文件仍位于 `exps/`。

# OP-Recon v4：观测重建与可见性门控残差补全

本节记录 2026-11-13 部署后的 v4 历史实现；当前配置以前面的 v5 说明为准。**新核心仅在 Replica 默认启用，限定单卡、batch_size=1、Grid_MLP；其他数据集配置未改动，没有启用残差 SDF。**

本轮替换几何核心，不继续叠加 v3 辅助机制。Replica 中关闭 v3 总开关和几何梯度投影，不运行联合参考、自适应保护、跨视角损失或可靠性缓存。纹理阶段恢复 v2 的 SDS 单独投影、真实训练图参考、固定强度 0.5；不使用 v3 统一先验分支。v2 的基础损失修正和 75% 均匀 / 12.5% 实例 / 12.5% 边界采样保留。旧实现代码保留用于历史阅读，但本轮 Replica 默认训练不走这些 v3 路径。

## 核心表示与更新

对每个实例（包括背景 ID，采用同一规则），最终隐式场为：

\[
f_i(x)=f_i^{obs}(x)+m(x)\Delta f_i(x).
\]

原哈希网格和 SDF MLP 是观测分支。新增共享的小型坐标解码器输出各实例的有界 SDF 残差：

- 输入：场景坐标以及 3 组 sin/cos 位置编码，共 21 维，不复制哈希网格。
- 解码器：21 → 32 → 实例数，中间 Softplus(beta=10)，末层零初始化。
- 残差：0.2 * tanh(decoder(x))，单位是归一化场景坐标，不是米。0.2 是初始工程设置，尚未通过完整训练标定，不是性能保证。
- 初始残差严格为零，不改变原场；新增模块初始化不消耗原网络后续初始化所用的随机数序列。
- 不增加扩散模型、先验调用次数、训练射线预算或额外跨视角查询。

门控复用 DP 原有 VisGrid。设 V(x) 为其冻结值，t=clamp((V(x)-0.1)/(0.5-0.1),0,1)，则 m=1-t²(3-2t)。V≥0.5 时残差为零；V≤0.1 时允许完整残差；中间平滑过渡。可见自由空间也接受这个门控，不能只保护表面位置。这里的 V 是已有估计，不是实际真值，错误可见性仍可能限制补全。

训练沿用原有流程：前 35000 步残差不参与前向，不建立新增激活图；原 VisGrid 初始化完成后才激活残差，并沿用该网格。没有新增可见性重估或稀疏证据缓存。

RGB、实例监督、单目深度/法线、Eikonal、平滑及物体区分等已有损失作用于最终合成场；原非 SDS 损失可以更新观测分支和残差分支。**几何 SDS 前向临时冻结观测主干、密度和辐射颜色的参数叶子，只允许残差参数接收 SDS 更新，随后在统一 backward 前恢复参数标志。** 冻结的是参数连接，空间导数仍保留，不能直接 detach 整个基础 SDF 输出，否则法线先验会失效。

原有 ray sampler、场景 SDF、实例 SDF、法线、Eikonal 和网格导出均通过同一个合成 forward 查询。SDF 的其余外观特征通道保持原表示。已有几何正则仍让两分支间接耦合，故本方法只声明 SDS 直接参数梯度隔离，不保证实际 Adam 更新后所有可见表面完全不变。

为了让法线/Eikonal 正确包含门控的空间变化，用冻结网格角点值进行可微三线性查询，数值与现有 VisGrid 对齐。没有复制网格；不让梯度更新可见性值。该路径支持空间二阶反传，避免 3-D grid_sample 的二阶反传限制。

## 资源边界

新增参数数为 704+33C，C 为自动读取的实例类别数。Replica scan1 的 C=11 时新增 **1067 个参数**；FP32 参数、梯度、Adam 两个矩合计约 16.7 KiB，不含激活和框架开销。残差参数加入现有 MLP 优化器组，学习率沿用主干 MLP，不使用哈希网格的 20 倍倍率。

新增激活和三线性查询仍消耗少量显存与计算；不能承诺零额外显存或给出完整训练开销百分比。点查询按最多 8192 点分块，限制单次临时张量规模，但这不代表反传时只保存一个分块的全部激活。关闭 v3 的多次参考反传和额外跨视角查询用于抵消开销。**尚未测量真实扩散训练时的峰值显存和每步耗时，只验证轻量可运行性。**

## 配置与检查点

Replica 当前关键设置：

| 配置键 | 值 |
| --- | --- |
| train.op_v3_enabled | False |
| train.op_geometry_conflict | False |
| train.op_texture_conflict | True |
| train.op_texture_strength | 0.5 |
| train.op_observation_reference | True |
| model.implicit_network.op_completion | True |
| model.implicit_network.op_completion_width | 32 |
| model.implicit_network.op_completion_bands | 3 |
| model.implicit_network.op_completion_max_residual | 0.2 |
| model.implicit_network.op_completion_hidden_visibility | 0.1 |
| model.implicit_network.op_completion_visible_visibility | 0.5 |
| model.implicit_network.op_completion_chunk_size | 8192 |

残差核心与 v3/几何投影同时开启会明确报错，避免无意叠加。关闭 op_completion 可消融残差，但此时仍保留 v2 的采样和基础损失修正，不能称为逐项完全恢复原 DP-Recon。

**几何阶段必须从头训练，不能加载 v1-v3 检查点继续训练本结构。** 检查点保存残差权重和激活标志，并继续保存原 VisGrid；恢复时两者都需要。本轮新增了旧检查点/缺失可见性网格的明确错误提示。续接本轮 v4 几何检查点进入纹理阶段时，残差随整个几何一起冻结。手工实例化网络做导出时，须绑定检查点对应的 VisGrid；标准训练器已完成绑定，缺失绑定会报错，不能悄悄退化成错误场。

不能对已有残差检查点强制重建另一套 VisGrid，当前显式拒绝 force_init_visgrid。已有实验、网格和评估文件均未覆盖。

## 本轮修改文件

| 文件 | 本轮作用 |
| --- | --- |
| code/model/op_residual.py（新增） | 小型残差解码器、冻结可见性可微查询、先验参数隔离上下文 |
| code/model/network.py | 合成 SDF、残差加入现有优化器参数列表、几何 SDS 隔离 |
| code/training/DPRecon_train.py | Replica/单卡互斥检查、绑定和激活、检查点兼容性 |
| code/confs/replica_grid.conf | 启用新几何核心、关闭 v3、采用 v2 纹理 |
| OP_RECON.md | 当前方法、训练方式与验证限制 |

未修改其他数据集配置、背景过滤/融合/连通块策略、评估脚本、数据或任何其他项目代码。没有实例名单或单物体参数。

## 轻量冒烟结果与限制

全部 Python 检查均在 /home/seed/miniconda3/envs/oprecon/bin/python 中，以 -B 和 PYTHONDONTWRITEBYTECODE=1 执行；使用空闲 GPU 0。没有安装/升级依赖，没有触碰其他 conda 环境，没有哈希验证或校验文件。

已通过：

- 修改代码的语法、Replica 默认启用和其他数据集默认关闭。
- 可见性查询与原 grid_sample 的数值/一阶空间导数对齐、二阶反传；网格无参数梯度。
- 零残差与原场一致、充分可见区域残差为零、隐藏区域残差有界、空点批次回退。
- SDS 直接参数梯度隔离、观测梯度保留及法线/Eikonal 反传。
- 实际项目哈希 CUDA 扩展及 ObjectImplicitNetworkGrid 的微型前向/Adam 更新；SDF 各查询接口一致；检查点激活和合成场读回。
- 实际 DPReconNetwork.forward、DPReconLoss、32 条合成射线和两次微型 Adam 更新，使用现有 PseudoSDF 接口和简化先验。单 rank DDP 使用 Gloo + CUDA 通过，确认参数共享与梯度接线。
- 实际训练器 _op_backward 的 v2 纹理路径。

NCCL 单 rank 检查在 DDP 构造的参数形状校验阶段持续等待，尚未进入模型前向；限时的最小 Linear/DDP 复查也未通过，相关测试进程已停止。因而**不能把这次冒烟描述成 NCCL 训练入口已通过**。没有修改生产训练入口的 NCCL 后端或系统通信配置；后续人工启动训练时需关注该环境初始化问题。Gloo 仅用于本轮微型验证，不代表默认训练配置改成 Gloo。

本轮未实例化预训练扩散模型，没有执行完整训练器初始化、真实 SDS 链路、完整训练、完整评估或多场景测试。微型解码器/网络的显存峰值不能代表完整训练，不宣称性能已优于 DP-Recon 或旧 OP。测试经标准输入执行，本轮临时目录、通信文件和缓存均已清理，不保留新增测试脚本。

## 用户后续从头训练

以下是原有入口，本轮未执行。GPU 编号仅为示例；NCCL 初始化的上述限制尚待在实际训练时确认。

```bash
conda activate oprecon
cd /home/seed/wangruiyao/OP-Recon/code
CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 training/exp_runner.py --conf confs/replica_grid.conf --scan_id 1 --prior_yaml geometry.yaml --description v4_geometry --none_wandb
```

几何及原有后处理完成后，将目录替换成实际新 v4 几何目录：

```bash
CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 training/exp_runner.py --conf confs/replica_grid.conf --scan_id 1 --prior_yaml texture.yaml --is_continue --ft_folder ../exps/oprecon_grid_replica_1/V4_GEOMETRY_RUN --checkpoint latest --description v4_texture --none_wandb
```

v4 是待验证的方法候选。应优先验证多个 Replica 场景下 Acc/Comp/CD 是否同步改善，再判断渲染收益，不以模块名称或一次烟测代替性能证据。

## 2026-11-18 补记：几何到纹理的网格文件交接修复

本次修复独立于 v4 几何核心，更新文件为 `code/training/DPRecon_train.py` 和本文档。背景 TSDF 过滤算法、最大连通块策略、模型及训练损失未改动。

### 原因与处理

几何实验 `v4_geometry_2026_11_16_23_31_39` 的终端记录显示，TSDF 子进程创建 EGL 离屏渲染器时先出现 `/dev/dri/renderD135`、`/dev/dri/card8` 的 Permission denied，随后 `eglInitialize` 失败（12289），未生成过滤网格。旧代码没有检查 `os.system` 返回值，仍把标准背景网格移动到 `surface_7500_0_ori.ply`；复制不存在的过滤结果又失败，异常分支仅打印“使用原始网格”，没有恢复标准文件名。纹理阶段因此复制到不完整的输入，最终在加载 `surface_7500_0.ply` 时退出。

修复后的交接流程：

- 使用 `subprocess.run(..., check=True)`，子进程解释器沿用 `sys.executable`，即启动训练所用的 oprecon 环境。
- 过滤成功且输出文件存在、非零字节后，先复制原网格为 `_ori.ply` 备份，再用同目录的 `os.replace` 原子替换标准文件；成功的 `_filtered.ply` 作为替换源被移入标准路径。
- 过滤或替换失败时保留标准背景网格，日志显示实际异常及保留路径。此时纹理使用未过滤的原始背景；EGL 权限问题本次未处理，没有改动环境或设备权限。
- 纹理初始化在复制全景图和生成锚点之前检查该 epoch 的全部实例网格文件名（含背景）。缺文件时直接列出完整路径，不自动选择其他 epoch 或以 `_ori` 偷换输入。空网格仍由已有加载检查拒绝。

### 本次已有实验文件恢复

以下两个目录的 `plots/surface_7500_0.ply` 已从各自的 `surface_7500_0_ori.ply` 恢复，`_ori` 均保留；没有覆盖既有标准网格、检查点、渲染结果或指标：

- `exps/oprecon_grid_replica_1/v4_geometry_2026_11_16_23_31_39`
- `exps/oprecon_grid_replica_1/v4_texture_2026_11_18_07_47_07`

原始背景经 trimesh 的轻量加载确认有 737770 个顶点、1469474 个三角面；恢复文件各 27956624 字节。两目录的标准实例网格 0–10 已齐全。失败纹理目录仍是未完成实验；应使用原命令、原几何 `--ft_folder` 重新启动纹理训练，入口会新建实验目录，无需为此次缺文件问题重训几何。

### 轻量冒烟与范围

全部 Python 执行使用 `/home/seed/miniconda3/envs/oprecon/bin/python`，带 `-B` 和 `PYTHONDONTWRITEBYTECODE=1`。从实际训练器 AST 提取本次文件交接代码，在微型临时文件上模拟子进程失败、输出缺失/零字节、成功替换和替换失败；验证标准网格保留、成功备份以及纹理缺文件检查。全部通过，并完成上述真实背景网格的加载检查。

测试通过标准输入执行，无新增保留测试脚本；测试与恢复所用临时文件均在 OP-Recon 内且已清理。没有哈希验证、校验文件、GPU 测试、实际 TSDF/EGL 渲染、完整训练或完整评估。冒烟通过只确认本次文件交接逻辑；完整纹理训练尚待用户重启验证。

---

# OP-Recon v3：可靠观测驱动的轻量先验协调

本节是本轮部署后的当前实现说明。后面的 v2 文档保留为历史记录；发生差异时以本节及实际代码为准。

**本轮新模块仅在 Replica 配置启用，限定单卡、batch_size=1。ScanNet++ 和 YouTube 配置保持本轮开始前的 v2 内容，没有启用 v3。所有实例采用统一规则，无具体物体名单或专用参数。** 用户将从头训练，本轮只做轻量冒烟，没有启动完整训练或完整指标评估。

修改与临时测试范围均为 `/home/seed/wangruiyao/OP-Recon`，Python 执行使用 `/home/seed/miniconda3/envs/oprecon/bin/python`。没有安装或升级依赖，没有操作其他 conda 环境，没有哈希验证或校验文件。按要求，本轮没有修改背景 TSDF 过滤、最大连通块筛选或替换背景网格的逻辑；保留 v2 既有评估与网格导出设置。

## 1. 四项改进与实际作用范围

| 模块 | v2 | v3 |
| --- | --- | --- |
| 统一多来源先验协调 | 仅 SDS 接受投影，合成纹理锚点／生成参考直接进入重建梯度 | 纹理阶段把 SDS、合成锚点及生成全景参考合并为一个先验分支，共同协调 |
| 场景与实例联合参考 | 每参数组只使用一个场景平均参考 | 每步场景参考，每 4 步最多增加一个自动轮换的前景实例参考，联合软约束求解 |
| 自适应保护 | 两阶段固定投影强度 0.5 | 0.5 作为强度上限，根据观测可靠性和冲突余弦计算实际强度 |
| 遮挡感知跨视角表面一致性 | 同一视图内的真实邻域深度 | 在训练视图之间检查有限表面对应，几何阶段增加深度／法线一致性项，两阶段均提供可靠性证据 |

v2 的 75% 均匀／12.5% 实例／12.5% 边界采样、RGB 重要性校正、真实同实例邻域深度及有效法线归约继续保留。没有更换网络结构或扩散模型，没有新增训练权重形状，也没有新增针对特定物体的修复。

### 1.1 统一先验分支与损失守恒

几何阶段仍以带原权重的 SDS 为生成先验；RGB、实例监督、单目深度／法线、Eikonal、平滑和物体区分等非 SDS 项保留在基础训练梯度。新增跨视角几何项也进入基础分支。单目深度／法线本轮没有纳入生成先验投影。

纹理阶段区分第 0 张真实训练图和后续 5 张合成 NeRF 锚点。设原场景 RGB 项的有效 RGB 元素总数为 C，真实图和合成锚点的元素数为 C_o、C_a：

```
observed_contribution = observed_MSE * C_o / C
anchor_contribution   = anchor_MSE   * C_a / C
scene_RGB_loss        = observed_contribution + anchor_contribution
base_loss             = scene_weight * observed_contribution
prior_loss            = scene_weight * anchor_contribution
                        + panorama_weight * generated_panorama_loss
                        + SDS_weight * SDS_loss
```

因此 `base_loss + prior_loss` 与原有加权总损失相同，没有重复加入锚点，也没有把真实图的原始总损失权重意外放大到六倍。参考损失单独按真实图有效像素归约，不作为额外损失再次叠加。

为控制开销，几种纹理伪监督先合并求一次梯度，**本实现不宣称每种先验分别满足保护约束**；不同先验之间仍可能存在抵消。几何阶段目前只有 SDS 属于该受控生成分支。

### 1.2 场景与自动轮换实例参考

`DPReconLoss` 额外提供逐射线的 `RGB + semantic` 观测图。RGB 图沿用混合采样重要性权重，实例语义项按已有计算方式归约。纹理参考为真实训练图的逐像素 RGB MSE。

参考均按 detached 可靠性加权平均；完整基础训练损失继续保留原有有效观测，不因参考可靠性低而删除 RGB 或语义训练信号。纹理参考限制在实际网格覆盖区域。

每 4 步从当前批次中至少有 4 个有效采样像素的前景实例中自动轮换选择一个；其他步仅使用场景参考。所有 ID 使用同一规则。实例候选集随视图变化，不保证每一轮每个实例获得完全相同次数。无候选实例时自然退回场景参考；无有效参考或近零参考梯度时跳过相应协调。

### 1.3 联合软约束，避免顺序投影

对每个现有参数组，在生成先验连接的参数张量子空间中计算参考梯度 g_i、先验梯度 d。令 n_i=g_i/||g_i||，K_ij=<n_i,n_j>，b_i=<n_i,d>。最多两个参考，求解小型非负二次问题：

```
min_{lambda >= 0} 0.5 * lambda^T [K + diag((1-eta_i)/eta_i)] lambda + b^T lambda
adjusted_prior = d + sum_i lambda_i*n_i
update_gradient = base_gradient + adjusted_prior
```

实现枚举最多 4 个活跃集合并检查 KKT 条件，使用数值稳定下限；不依次对实例投影，因此没有该顺序偏差。单参考时约等于移除 eta 比例的冲突分量。软约束允许残余冲突，`op_soft_violation_rate` 非零不代表程序失败。参考冲突与矩阵退化不意味着能够同时满足全部硬约束。

自适应强度为：

```
cos_i = cosine(g_i, d)
eta_i = stage_strength * reliability_i * (0.5 + 0.5*clamp(-cos_i, 0, 1))
```

几何／纹理的 `stage_strength` 均默认 0.5。关闭自适应开关时使用固定 stage_strength。未观测或未检查区域没有被假定为完全可靠；当前实现也没有用 RGB 残差直接判定可靠性。Adam 的动量与自适应缩放仍意味着这一机制不保证实际每步观测损失下降。

## 2. 有界跨视角检查与可靠性

仅使用 `SceneDatasetDN_segs` 已加载的训练图相机、训练实例标签和有效掩码；不加载 eval-novel-views 图像、GT mesh 或真实 GT 深度，也没有额外深度网络。若数据集启用训练视角子集，只从该子集选择目标。

每个训练视角根据相机距离与朝向选最多 3 个相邻训练视角并轮换。几何阶段在第 5000 步后，每 8 步最多选 32 个有效源表面点；先检查投影和同实例标签，再对剩余目标射线查询现有 SDF／体渲染深度。目标查询不执行 RGB 渲染、背景正则、Eikonal 或 SDS，也不构造第二个完整训练前向。

筛选条件包括：

- 源、目标都具有有效表面；SDF 有符号穿越，累计权重大于 0.9。
- 投影深度为正、像素位于目标训练图内、目标掩码有效、两端训练实例标签一致。
- 世界坐标下法线余弦不小于 0.5。
- 投影深度与目标表面深度的相对差不超过 0.05。源点位于目标表面后方且超过容差时作为遮挡拒绝。

仅对通过检查的对应计算相对深度 Smooth-L1 和世界坐标法线余弦项，权重默认 0.05／0.01，并用 2000 步渐入。目标深度／法线 stop-gradient，源侧几何接受梯度，避免为小型参考保留第二套完整计算图。检查无有效对应时损失为零，不构造虚假对应。

本门限是保守重叠约束：它不会主动修复被门限拒绝的严重跨视角差异，也不能监督完全不可见区域。它不是独立真值；仍依赖真实 RGB／实例观测约束共同防止自一致错误。

可靠性证据按像素稀疏保存在内存 CPU 字典中，并用排序索引批量查询。有效对应的证据为 `exp(-relative_error/0.05)*max(normal_cosine,0)`；失败检查使用最低 0.1，尚未检查区域默认 0.5；证据随 2000 步时间常数衰减回默认值。每视角超出 8192 条证据时保留最近 4096 条，避免长期增长。

纹理阶段几何冻结，不增加几何一致性训练损失。利用光栅器已计算的世界坐标位置和法线，每 8 步最多检查 32 个被网格充分覆盖的真实训练像素，只更新保护参考的可靠性。光栅器新增返回这些现有张量，不改变 `comp_rgb`、背景颜色或网格过滤。

若关闭 `op_multiview`，没有几何一致性项和证据缓存，参考可靠性统一为 1；其他新模块可单独保留。

## 3. Replica 配置与消融

只有 `code/confs/replica_grid.conf` 新增并默认启用下列 v3 项。其他数据集配置没有改动，仍保留既有 v2 开关，不能把它们描述为“已经启用 v3”。

| 配置键（均在 train 内） | 默认值 | 含义 |
| --- | --- | --- |
| `op_v3_enabled` | True | 本轮方法总开关；False 回到 v2 训练路径 |
| `op_unified_prior` | True | 纹理多来源伪监督合并协调 |
| `op_joint_reference` | True | 场景＋自动实例双参考；False 仅场景参考 |
| `op_adaptive_protection` | True | 可靠性／冲突驱动的软强度 |
| `op_multiview` | True | 跨训练视角检查、几何项和可靠性证据 |
| `op_instance_interval` | 4 | 实例参考执行间隔 |
| `op_reference_min_pixels` | 4 | 实例参考最低有效像素数 |
| `op_mv_max_points` | 32 | 每次检查的源点数上限 |
| `op_mv_interval` | 8 | 两阶段跨视角检查间隔 |
| `op_mv_warmup` | 5000 | 几何一致性开始步数 |
| `op_mv_ramp` | 2000 | 几何项渐入长度 |
| `op_mv_depth_tolerance` | 0.05 | 相对深度差门限 |
| `op_mv_normal_cos_min` | 0.5 | 世界法线余弦下限 |
| `op_mv_depth_weight` | 0.05 | 相对深度一致性权重 |
| `op_mv_normal_weight` | 0.01 | 法线一致性权重 |
| `op_reference_floor` | 0.5 | 未检查／过期参考的默认可靠性 |
| `op_reference_ttl` | 2000 | 可靠性证据的衰减时间常数 |

v2 的两阶段冲突开关和强度仍有效。v3 使用本节参考构造，旧 `op_observation_reference` 仅控制 v2 的参考选择。关闭某阶段冲突开关时该阶段按原加权总损失正常反传；跨视角几何项若开启仍会训练。全部关闭 v3 时恢复 v2 数值路径，不承诺改变随机数调用后逐位复现旧实验。

建议分别关闭统一先验、实例参考、自适应保护、跨视角，分析各模块作用。跨视角开关同时去掉几何项和可靠性证据，因此这项消融包含两种作用，不应称为单独去掉“深度损失”。新功能对单卡与 Replica 有显式检查，本轮不支持多 rank 或其他数据集的 v3 运行。

## 4. 资源与实现文件

主训练仍为每步 256 条射线；跨视角额外查询最多 32 条目标射线、每 8 步执行一次，相当于调度上平均最多 4 条额外目标射线／步，另有表面法线查询。目标射线采样仍使用原误差有界采样器，不能把射线数比例直接当成总运行开销比例。几何预热前不查询。

场景参考每受控步骤计算，第二个实例参考最多每 4 步增加一次梯度计算；额外梯度张量会增加显存。上限和执行频率用于控制成本，尚未测量完整训练中的实际显存峰值和耗时。单卡仍使用原 `torchrun --nproc_per_node=1` 入口。

| 本轮文件 | 职责 |
| --- | --- |
| `code/utils/op_gradients.py` | 保留 v2 helper，新增最多双参考的联合软协调 |
| `code/utils/op_observations.py`（新增） | 可靠性加权的场景／实例参考与统一轮换 |
| `code/utils/op_multiview.py`（新增） | 坐标投影、训练视角选择、表面对应、几何一致性和稀疏证据 |
| `code/model/loss.py` | 按需提供与原观测损失一致的逐射线图 |
| `code/training/DPRecon_train.py` | 新模块开关、两阶段接线、分支损失拆分与诊断 |
| `code/threestudio/models/renderers/nvdiff_rasterizer.py` | 返回现有世界位置／法线张量供检查复用 |
| `code/confs/replica_grid.conf` | 本轮仅 Replica 的默认参数 |
| `OP_RECON.md` | 当前方法、历史说明及测试边界 |

新增诊断包含 `op_effective_strength`、`op_soft_violation_rate`、`op_reference_count`、`op_instance_id`、`op_reference_confidence`、`op_mv_checked`、`op_mv_pairs`、`op_mv_loss`，保留原冲突余弦和先验范数保留率。它们只解释当前步骤，不是最终性能保证。逐射线图在训练器中消费后移除，不送入标量日志。

## 5. 本轮验证与限制

已在 `oprecon` 环境通过轻量冒烟：

- Replica 新配置与其他配置默认关闭检查；修改文件语法。
- 联合软求解的参考排列不变性、退化／反向参考、零强度、可靠性强度变化、未使用参数及基础梯度保留。
- 场景／实例轮换、空参考回退。
- 带旋转与 skew 相机的反投影／重投影一致性；有界对应点数、源侧几何梯度、遮挡／标签不符／相机后方拒绝、执行间隔与证据过期。
- 少量真实 Replica scan1 输入、实际 `DPReconLoss` 逐射线图与观测均值一致、实际训练器辅助方法的几何／纹理各一次微型 Adam 更新、冻结几何及 v2 回退。
- 纹理真实／锚点分支的加权守恒；原射线采样器与实际体渲染函数对一条解析球射线的目标深度／法线查询。
- 补充通过实际纹理循环代码的完整 v3／模块消融／v2 三条路径、纹理可靠性适配器、单 rank DDP `no_sync` 的双参考更新，以及较大先验梯度尺度下的小型求解稳定性。

测试的 SDS 用简化替代损失，没有加载预训练扩散权重，没有运行完整训练器、完整训练或完整指标评估。微型网络的资源占用不能代表真实训练成本。没有开展多卡测试，也没有宣称结果已优于 v2、DP-Recon 或 ObjectSDF++。

本轮测试通过标准输入执行，临时缓存限定本目录并已删除，不保留新增测试脚本或日志。v1 旧测试文件属于历史产物，没有作为 v3 的验证入口。

## 6. 从头训练建议

用户将在部署后手动从头训练。以下命令未在本轮执行；GPU 4 仅为示例：

```bash
conda activate oprecon
cd /home/seed/wangruiyao/OP-Recon/code
CUDA_VISIBLE_DEVICES=4 torchrun --nproc_per_node=1 training/exp_runner.py --conf confs/replica_grid.conf --scan_id 1 --prior_yaml geometry.yaml --description v3_geometry --none_wandb
```

几何完成及原有后处理完成后，使用新 v3 几何目录进行纹理阶段：

```bash
CUDA_VISIBLE_DEVICES=4 torchrun --nproc_per_node=1 training/exp_runner.py --conf confs/replica_grid.conf --scan_id 1 --prior_yaml texture.yaml --is_continue --ft_folder ../exps/oprecon_grid_replica_1/V3_GEOMETRY_RUN --checkpoint latest --description v3_texture --none_wandb
```

`V3_GEOMETRY_RUN` 替换为实际新目录。多场景测试使用相同配置与规则，仅替换数据 scan_id 及该项目原有必需的数据文件。背景过滤仍沿用现有流程，因此其既有局限仍可能影响网格／纹理结果。发表前仍需统一评估口径，报告多场景、多随机种子、完整性指标、消融及实际开销。

---

# OP-Recon v2 实现与使用说明

本轮在 OP-Recon v1 上继续改进。修改范围为 `/home/seed/wangruiyao/OP-Recon`，Python 执行与轻量测试均使用 `oprecon` conda 环境。本轮没有启动完整训练、完整指标评估或预训练扩散链路，没有安装或升级依赖，没有做哈希验证。

## 方法与默认配置

| 模块 | v1 | v2 默认行为 |
| --- | --- | --- |
| 几何冲突参考 | 所有非 SDS 损失 | 训练 RGB 与实例标签监督 |
| 纹理冲突参考 | 训练图、NeRF 锚点和背景全景 | 真实训练图中被网格覆盖的像素 |
| 投影强度 | 两阶段共用 1.0 | 几何 0.5，纹理 0.5，可分别设置 |
| 射线分配 | 50% 均匀／25% 实例／25% 边界 | 75% 均匀／12.5% 实例／12.5% 边界 |
| 边界分支 | 独立抽像素 | 在同实例边界带内抽真实水平／垂直邻接对 |
| 深度梯度 | 随机射线重排成宽 32 的伪图像 | 每张图单独拟合尺度／偏移，仅计算真实相邻、同实例、有效像素的残差梯度 |
| 法线监督 | 无效预测置零后对所有像素平均 | 排除无效监督，按有效像素归约 |

RGB 仍按采样的实际边缘概率校正到全图均匀平均目标；语义、单目深度、法线等其他项没有做这个概率校正。实例抽样不保证每一步所有实例都有射线。射线总数和训练时长配置没有增加。

### 观测参考与优化梯度分离

记全部非 SDS 训练项为 `L_train`，可靠直接观测项为 `L_obs`，已带原有权重和可见性权重的 SDS 为 `L_sds`。对于每个参数组：

```
g_obs = grad(L_obs)
g_train = grad(L_train)
g_sds = grad(L_sds)
g_sds_adjusted = g_sds - eta * min(dot(g_sds, g_obs), 0) / ||g_obs||² * g_obs
g_update = g_train + g_sds_adjusted
```

零范数跳过投影。只在本次 SDS 连接的参数张量子空间中投影，避免为不相关字段引入先验梯度。几何按 SDF 编码／MLP 分组，纹理按纹理编码／MLP 分组，其余参与参数归到 `other`。

几何的 `L_obs = rgb_weight * rgb_loss + semantic_weight * semantic_loss`。单目深度／法线、Eikonal、平滑、碰撞和背景约束继续参与 `L_train`，不再成为需要保护的参考。这里的“可靠”指直接来自训练图像／实例标签，并没有新增置信度网络或保证标签完全无噪声。

纹理的 `L_obs` 只使用合并批次中的第 0 张真实训练图，且限定在网格覆盖区域。NeRF 锚点和背景全景仍参与纹理训练。场景 RGB MSE 按可见且覆盖的 RGB 元素数归约，避免遮罩后仍按全图面积平均造成监督稀释。这个归约变化会改变相同损失系数的实际作用强度，后续应通过消融观察。

`eta=0.5` 只削弱一半反向分量，不声称完全消除冲突。Adam 的动量和自适应缩放也意味着梯度投影不能保证实际有限步更新一定降低观测误差。

### 真实邻域与成对采样

边界由训练实例标签差异得到，以默认 5×5 核膨胀；在边界带中筛选两端同标签的水平／垂直邻接边，再均匀抽边并加入两个端点。边界预算为奇数时，剩余一个像素在边界带均匀抽样；没有有效边时回退到边界像素抽样，没有边界或实例时对应预算回归全图均匀分支。

令像素数为 M，总射线数 N，均匀、实例、成对边界端点、独立边界像素的实际配额分别为 n_u、n_i、n_p、n_b，则：

```
q(x) = (n_u/N)/M + (n_i/N)*q_instance(x)
       + (n_p/N)*degree(x)/(2*edge_count)
       + (n_b/N)*is_boundary(x)/boundary_pixel_count
RGB_weight(x) = 1 / (M*q(x))
```

实例概率为 `1/(可见前景实例数 × 该实例像素数)`；实际配额取整后计算权重。成对抽样存在相关性，单像素 RGB 平均的重要性校正仍成立。默认每 256 条射线为 192 均匀、32 实例、32 边界端点，不增加额外射线。

新深度项先按每张图的有效采样点拟合单目深度的尺度／偏移，再计算拟合残差的平方误差及真实邻接差分。邻接项去掉重复像素，只对同实例有效端点计算；没有邻接边时该项为零。深度仍是辅助先验，不进入投影保护参考。

## 背景和网格渲染修复

- 两种隐式网络的 `get_specific_outputs(x, idx)` 改为对所选 SDF 通道求法线梯度，同时保留场景 SDF 输出供语义渲染使用。此前背景法线实际上可能来自前景字段。
- 虚拟背景相机的法线变换使用当前背景相机旋转，避免沿用原训练相机旋转。
- 无等值面的物体明确导出空预测并移除对应旧 bbox；不再复用上一物体的网格。下一次提取若没有该物体 bbox，回退到全局网格。纹理阶段遇到空网格给出明确错误。
- 物体与全场景导出都使用 `plot.resolution`，取消隐藏的 256／768 硬编码。三个配置默认统一为 256，以控制资源占用。旧版全场景实际以 768 提取，因此正式比较时必须统一双方导出分辨率。
- 纹理训练打印真实训练视角的网格覆盖率，最终网格视角渲染也打印覆盖率。未覆盖区域继续使用原渲染器背景颜色，没有用 GT 或 NeRF 填洞来改变评测图。

已有 scan1 网格的只读检查发现背景只覆盖很薄的片段，nvdiffrast 图像明显偏白；这些现象与覆盖不足相符，但尚未证明它是唯一原因。本轮没有重写旧检查点、旧网格或旧渲染图；背景修复的效果需要重新训练检验，不能宣称已修复旧结果。

## 评估修复与公平比较

- LPIPS 的所有调用对 `[0,1]` 图像显式使用 `normalize=True`。旧 DP／OP 的 LPIPS 数字需要使用同一修复版评估器重算后再比较。
- 渲染评估按视图 ID 配对，检查缺失／重复 ID 和尺寸；避免整除批次时出现空批次，并按图像数汇总。
- `Prec + Recal = 0` 时 F-score 为 0；空预测的 F-score、Precision、Recall、法线分数为 0，距离为 `inf`，明确表示失败。
- 物体评估按完整 GT 物体集合遍历，并与训练实例映射检查一致性。缺失、空网格、零 F-score 都进入统计，不再过滤。每场景额外输出 `counts.txt` 记录预期对象数、缺失数与零分数。距离遇到空预测会使均值为 `inf`，应与失败数量一起报告，不能静默剔除。
- 子评估使用当前 Python 解释器，失败立即报错，避免读入旧指标文件。
- 保留 `--none_fuse` 的尺度矩阵变换与直接网格评估含义。此参数不会补全缺失表面或自动修复坐标／尺度。

现有图像可用本项目的评估器单独重算；预测目录允许只读指向基线结果，输出应放在 OP-Recon 内。例如：

```bash
conda activate oprecon
cd /home/seed/wangruiyao/OP-Recon
python -B eval/eval_rendering.py --pred_dir PREDICTION_IMAGE_DIRECTORY --target_dir data/replica/scan1/eval-novel-views/rgb --output_txt /home/seed/wangruiyao/OP-Recon/corrected_rendering_metrics.txt
```

上述命令会加载 LPIPS 权重并运行实际评估，仅供后续手动使用，本轮没有执行。NeRF 与 nvdiffrast 对应不同阶段，应分别报告。

## 开关与消融

三个配置：`code/confs/replica_grid.conf`、`scannetpp_grid.conf`、`youtube_grid.conf`。

| 配置键 | 默认值 | 含义 |
| --- | --- | --- |
| `train.op_geometry_conflict` | True | 几何梯度冲突控制 |
| `train.op_texture_conflict` | True | 纹理梯度冲突控制 |
| `train.op_geometry_strength` | 0.5 | 几何投影强度 |
| `train.op_texture_strength` | 0.5 | 纹理投影强度 |
| `train.op_observation_reference` | True | 只保护直接观测；False 使用全部非 SDS 参考 |
| `dataset.op_balanced_sampling` | True | 混合采样 |
| `dataset.op_instance_fraction` | 0.125 | 实例配额 |
| `dataset.op_boundary_fraction` | 0.125 | 边界配额 |
| `dataset.op_boundary_width` | 5 | 边界膨胀核边长 |
| `dataset.op_boundary_pairs` | True | 成对邻接边界抽样；False 独立边界像素 |
| `loss.op_depth_neighbors` | True | 真实像素邻域深度项；False 使用旧深度项 |
| `loss.op_valid_normals` | True | 有效法线掩码归约；False 使用旧归约 |
| `plot.resolution` | 256 | 物体／场景网格提取分辨率 |

没有新增方法键的旧配置关闭相应方法开关。旧 `train.op_projection_strength` 仍可作为两个阶段强度的回退值；显式阶段强度优先。默认新方法不增加网络参数形状，但增加一次独立观测梯度计算与相关显存／时间开销。DDP 受控几何路径保持 `no_sync`，在投影前分别平均完整训练、观测和先验梯度；本轮未测试多 rank。

建议先验证完整 v2 的多个场景和随机种子，再分别去掉几何保护、纹理保护、采样、真实邻域、观测参考做消融。固定射线预算、导出分辨率和统一评估器，逐对象报告 CD、Comp／Recall、F-score，并报告渲染指标、覆盖率和运行开销。收益应在更多对象／场景上稳定出现，不能只依赖单个物体的提升。修复评估得到的变化与重新训练带来的方法收益应分开报告。

## 本轮轻量验证

已在 `oprecon` 中通过：Python 语法与三份配置解析；真实 Replica scan1 的少量采样及批次拼接；采样实际边缘概率、奇数配额与空池回退；邻域深度的顺序不变性／同实例过滤／空掩码／有限梯度；区分全部训练与观测参考的软投影；实际损失和训练器辅助函数的小型几何与纹理各一次 Adam 更新；两种网络的选定 SDF 梯度；低分辨率微型网格及空字段处理；几何零分／空预测统计；渲染批次／视图 ID 与 LPIPS 归一化参数传递。

SDS 与 LPIPS 在冒烟中使用小型替代函数，没有下载或加载扩散／LPIPS 权重，没有做真实 nvdiffrast 光栅器集成或完整训练／评估。补充通过了单 rank DDP `no_sync` 的独立观测梯度路径、缺失 bbox 回退以及两个微型失败物体的命令行评估汇总。所有本轮临时文件、测试产物和缓存已删除，测试代码通过标准输入执行，没有新增保留的测试脚本或日志。仓库原有 v1 `scripts/oprecon_smoke.py` 和日志不是本轮产物，其旧结果不代表 v2 验证，旧脚本接口也不适用于 v2。

本轮通过仅证明小样本执行和相关边界逻辑可用，不证明最终几何或渲染指标提高。梯度投影和均衡采样已有研究基础，论文的新颖性与有效性仍须通过相关工作比较、多场景实验和消融建立。

## 手动训练

以下命令仅供之后手动执行，本轮没有启动：

```bash
conda activate oprecon
cd /home/seed/wangruiyao/OP-Recon/code
CUDA_VISIBLE_DEVICES=4 torchrun --nproc_per_node=1 training/exp_runner.py --conf confs/replica_grid.conf --scan_id 1 --prior_yaml geometry.yaml --none_wandb
```

几何完成后，使用新生成的几何实验目录和实际 epoch 进行纹理阶段：

```bash
CUDA_VISIBLE_DEVICES=4 torchrun --nproc_per_node=1 training/exp_runner.py --conf confs/replica_grid.conf --scan_id 1 --prior_yaml texture.yaml --is_continue --ft_folder ../exps/oprecon_grid_replica_1/GEOMETRY_RUN --checkpoint latest --none_wandb
```

`GEOMETRY_RUN`、`7500` 和 GPU 编号须按实际结果／资源替换。新方法的几何收益需要重新训练，使用旧几何检查点只训练纹理不能验证这些改动。
