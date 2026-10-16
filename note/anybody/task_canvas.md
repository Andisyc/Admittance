# Anybody Encoder–Decoder 接入看板

更新日期：2026-10-16（Asia/Shanghai）。本文件保留技术接口、工程包及历史证据；唯一当前协作状态入口为 [collaboration/overview.md](../../collaboration/overview.md)，本次同步记录见其最新交接。

## 当前游标与授权

- **2026-10-16 架构更正：**用户明确要求将训练实现放回 UniLab，当前按 [ADR-0008](../../docs/sphinx/source/adr/ADR-0008-in-tree-training-runtime.md) 迁入 `src/unilab/rl`，移除外部训练包。下文 S1/S3 中外部 `uni_rl` 的表述是当时的核查基线，已被本次决定替代；不据此回退外部依赖。当前进度及验证以协作总览最新交接为准，S4 尚未开始。
- 用户已确认：在当前 `ADAPT/Admittance` 中接入 Anybody Encoder–Decoder，**以 AMASS 为训练数据来源**，后续使用 UniLab 异步训练架构。不能用已有 G1 NPZ 接入替代 AMASS 数据接入。
- 当前授权：用户已要求将 `Andisyc/Admittance` 的 `main` 更新同步到 `feat/anybody-muse-s2` 并交付该分支。已核实仓库所有者为 `Andisyc`，当前授权账号 `sss99991112` 有推送权限但不是所有者；此前称其为当前账号个人 fork 不准确。仍遵守**不在本地下载或保留训练数据，不在本地训练，后续走远程 SSH**；不执行原始数据下载、优化器或训练。
- 当前进度：**S1、S2 基线和 S3 数据层已完成，阶段验收 3/6**。S3 已实现原始 AMASS 读取、窗口/批次、显式分组划分，并验证 384 条真实动作。网络输入特征/训练标签和异步训练尚待 S4，不把人体数据当作已联接的 524 维机器人观测。
- 交付位置：S2 已在 `Andisyc/Admittance` 的 `feat/anybody-muse-s2` 分支提交至 `52466db7`；误建 fork 已由用户手动删除。S3 代码、测试、远程下载入口和本看板在同一分支交付；原始数据当前不在本地工作区，也不进入 Git。逐文件清单/划分需由远程命令重新生成，不能将历史统计冒充已恢复的清单。
- 下一步：规划 S4 的 AMASS 特征/监督到模型的映射、异步训练接入及远程 SSH 运行条件；先明确训练目标和网络输入输出，不在本地启动训练。
- 当前工程依据：E10 的原项目源码核查和修正后的 S3 工程包。`ANYBODY-IO-v1` 保留为当前 G1 网络基线；AMASS 特征、训练标签和是否修改网络输入输出尚未冻结，训练方式继续留到 S4（E5）。
- 协作入口：合入 main 的协作目录后，统一使用 [协作总览](../../collaboration/overview.md)，本文件仅保留具体工程包/证据，避免两个当前状态入口。远程准备/取批见 [AMASS 数据说明](../../datasets/amass/README.md)；manifest 和 splits 由该命令在服务器生成。没有 SSH 连接信息，未执行远程下载或训练。

## 系统边界与证据基线

终点：以 AMASS 为数据来源，Encoder–Decoder 可在当前 Admittance 通过配置构建，经明确的数据/监督接口和 UniLab 异步训练链路更新，完成保存、重载及确定性推理验证。人体动作表示到机器人控制的后续路线仍待决定，不因本轮讨论自动改为纯人体自编码器。

范围外：外力补偿分支、PCGrad、导纳目标、奖励重设计、全量 AMASS 重定向、长期训练、策略效果结论和硬件部署。冻结 Decoder 的 latent RL 属于待选择的后续训练模式，不自动纳入此次架构接入。

| 对象 | 身份/现状 | 证据等级 |
| --- | --- | --- |
| 当前目标 | `Andisyc/Admittance`，分支 `feat/anybody-muse-s2`；S3 起点为 `52466db7f5d74989b064372ffc2f0e22a4b2a34f`，后续提交身份由 Git 记录 | code-confirmed |
| 模型参考 | `hazel-hammer/Anybody@0cffa139eab0bc145c9216223e249f03c67a324c` | code-confirmed |
| 训练包基线 | 初始通用源码来自 `unilab-rl==1.0.0`，现迁入 `src/unilab/rl`；工作树已移除外部包依赖，最终验收待 CI | implementation-in-progress |
| 训练器依赖 / 网络 owner | 算法、collector、learner、IPC 归当前仓库 `src/unilab/rl`；自定义纯网络归 `src/unilab/models/anybody.py`。不以旁边源码仓库作为运行依赖 | implementation-in-progress |
| 旧 UniLab | `unilab_stand@e1717b9…` 只作历史参考；其 `src/unilab/algos/torch/*` 路径不是当前修改目标 | code-confirmed |
| 既有笔记 | `note/upstream_migration/` 是 AMP 历史规划/执行记录，不授权本主题；不修改其状态 | note-confirmed |

当前 owner 链路：

```text
已实现数据入口：
AMASS 原始人体序列 -> 字段/帧率/模型校验 -> clip 内时序窗口
  -> 人体表示与下游监督适配（待明确，不能直接当作机器人控制输入）

当前已核对的机器人侧链路（尚未接通运行）：
UniLab 机器人动作数据 / task observation terms
  -> ObservationManager（历史与 reset）
  -> UniLab Hydra 配置 / train_appo（组装）
  -> unilab.rl APPORunner -> collector / learner / 权重同步
  -> Encoder -> latent -> Decoder -> 关节动作
  -> UniLab action manager -> unisim

训练状态 -> unilab.rl checkpoint -> UniLab playback / ONNX 组装
```

自定义网络属于当前 Admittance 的 `unilab.models`；算法、collector、learner、IPC 由仓内 `unilab.rl` 管理。纯网络仅依赖 Torch 和标准库。S4 通过仓内可配置 actor/runtime 扩展点接入。环境仍由可 pickle 的 `EnvFactory` 注入，通用训练模块不导入具体 task/环境注册。

现有约束来自 [AGENTS.md](../../AGENTS.md)、[层级边界](../../docs/sphinx/source/en/4-developer_guide/1-architecture/3-layer_boundaries.md)、[观测契约 ADR-0005](../../docs/sphinx/source/adr/ADR-0005-unified-obs-critic-env-and-ipc-contract.md) 及训练所有权更正 ADR-0008。物理接口仍遵循 ADR-0007 的 unisim 边界。

## S1 接口结果：ANYBODY-IO-v1

状态：S1 工程接口已收敛，S2 已据此实现 G1 纯网络基线；它只约束未来机器人侧适配，不能代替 AMASS 数据接口。训练模式不在该接口版本中选择；不声明 Anybody 旧 checkpoint 的兼容性。

| ID | 接口/不变量 | 当前状态 |
| --- | --- | --- |
| I1 | 代码与交付目标均为当前 Admittance；自定义网络在仓内，训练器内置 `unilab.rl` | 最新用户决定及 ADR-0008 替代原外部训练器归属 |
| I2 | 移植 `_MUSETransformerEncoder` 和条件式 MLP Decoder；KP Encoder、teacher 网络及 donor 训练器不属于 S2 | S1 已绑定 |
| I3 | 机器人分支的 goal/proprio/mask 语义见下表；`critic` 不进入模型。AMASS 人体参数是另一个数据层 | G1 基线已绑定；AMASS 特征适配尚未定义 |
| I4 | G1 起始配置为 latent=16、H=5、每帧 proprio=90、原始关节动作=29；网络维度由配置计算 | S2 已实现并通过 CPU 测试 |
| I5 | 网络内部 latent 不等于算法采样动作；先完成网络与观测接入，首个训练方式延后到 S4 | 用户确认延后，E5；不阻塞 S2/S3 |
| I6 | 单一纯模型语义供采集/更新/推理复用；normalizer 不进入纯模型，未来 actor wrapper 对预处理负责；mask 保持 0/1 | S1 已绑定；异步一致性待 S4 |
| I7 | AMASS 入口读取人体模型参数，不要求用户先提供 G1 NPZ；若后续选机器人控制路线，才进入具名重定向及物理观测/监督适配 | 用户明确 AMASS 数据来源；原 G1 NPZ 前置条件已纠正 |

### 精确输入与数据来源

以下 524 维布局只描述当前 S2 的 G1 默认网络。AMASS 的人体姿态、根平移、体型参数或帧率不能按维度凑入其中；AMASS 数据入口完成后仍须单独验证其特征到网络的映射。

`obs` 是 float32 `[..., 524]`，下列区间为 Python 左闭右开切片。全部值须有限。目标机器人/关节顺序取专用配置继承的 [G1 owner YAML](../../src/unilab/conf/appo/task/g1_motion_tracking/mujoco.yaml) 中 `scene.entities.robot.joint_names` 的具名 29 项顺序：左腿 6、右腿 6、腰 3、左臂 7、右臂 7。引用这个顺序，不另写第二份权威列表。参考轨迹及 actions 必须使用同一具名顺序，维度相同不视为顺序相同。

| 切片 | 字段 / 历史 | 定义、单位和权限 | UniLab 对应/缺口（S3） |
| --- | --- | --- | --- |
| `[0:58)` | `delta_command`；当前帧 | `[q_ref-q, dq_ref-dq]`，rad / rad·s⁻¹；动作目标与当前状态 | `MotionCommand.joint_pos/joint_vel` 与 `robot_joint_pos/robot_joint_vel` 已有；需新 observation term，不能用现有绝对 `generated_commands` 顶替 |
| `[58:64)` | `motion_anchor_ori_b`；当前帧 | `R(robot_torso)^T R(ref_torso)` 的前两列按行展开：`[R00,R01,R10,R11,R20,R21]`；无量纲；参考命令＋机器人 anchor 姿态 | 同名 term 及 kernel 已有；本步只核对公式，数值对照留给 S3 |
| `[64:67)` | `motion_anchor_pos_b`；当前帧 | `R(robot_torso)^T (p_ref_torso-p_robot_torso)`，m；含仿真/估计位姿 | 同名 term 已有；不能声称仅凭 IMU/编码器即可部署 |
| `[67:70)` | `base_lin_vel`；当前帧 | 目标契约采用 pelvis **link 原点**线速度，以 pelvis frame 表达，m·s⁻¹；仿真状态/将来估计器输入 | 用公共 `unilab.envs.mdp.base_lin_vel -> EntityData.root_link_lin_vel_b`；不用偏置位置的 IMU velocimeter 冒充 |
| `[70:73)` | `ref_base_lin_vel`；当前帧 | `R(ref_torso)^T v_ref_torso_w`，m·s⁻¹；轨迹参考速度；遵循输入机器人 NPZ 的采样点定义 | anchor quat/lin-vel 的公共属性已有，需新 term；这是 donor 实际公式，尽管 donor 注释写成 robot base frame |
| `[73:218)` | `joint_pos`；5 帧 | 每帧 `q-q_default`，rad；当前机器人状态；不额外减去 episode joint bias | 公共 `unilab.envs.mdp.joint_pos_rel`；与当前 task 的 `motion_joint_pos_rel` 不同，需显式选择 |
| `[218:363)` | `joint_vel`；5 帧 | 每帧 `dq-dq_default`，rad·s⁻¹ | 公共 `joint_vel_rel` 已有 |
| `[363:378)` | `base_ang_vel`；5 帧 | pelvis 角速度，以 pelvis frame 表达，rad·s⁻¹ | 用公共 `unilab.envs.mdp.base_ang_vel -> root_link_ang_vel_b`；当前 task 的 `torso_gyro` 不能直接替代 |
| `[378:523)` | `last_action`；5 帧 | 对应历史观测时刻最近已执行的 29 维 **raw action**；不是力矩或已缩放关节目标 | 公共 `last_action` / action manager 已有 |
| `[523:524)` | `goal_mask`；当前帧 | 原始 float32 0/1；1 表示屏蔽整个 goal token；不是物理传感器 | 需显式 mask observation term；初次网络/观测验收用全可见的 0，不引入随机课程 |

结构与语义差异：Isaac Lab v2.1.0 的 `base_lin_vel` 取 pelvis 质心速度；目标 UniLab 公共接口取 link 原点速度，当前 task 的 velocimeter 又位于偏置 IMU site。此处明确选用目标公开 link 语义、从头训练的输入接口；不宣称三者可直接互换或旧权重等价。若后来要求逐值复现 donor 状态/旧 checkpoint，须回总体规划处理 COM/参考数据及动作缩放，而不能在 S3 静默探测 backend 私有量。

历史/预处理规则：

- goal 与 mask 不滚动历史；四个 proprio 项分别设置 H=5，group-level history override 留空。每项按 oldest→newest 存储，采样间隔沿用 `ctrl_dt=0.02 s`。
- 外部 flat obs 为 term-major；模型先还原 `[...,5,90]`，每帧顺序 `[q_rel(29),dq_rel(29),omega_pelvis(3),last_action(29)]`，Decoder 再按时间帧展开为 450 维。不能将外部连续的 450 维切片直接送入 Decoder。
- 首次观测/部分 reset 的对应行以首个 post-reset 本体样本回填 5 槽；未 reset 行及全局时间指针不被回填影响。复用现有 CircularBuffer，不新增状态 owner。
- 原始 obs 以物理单位产生；S2 纯网络不持有 running statistics。未来 wrapper 如启用经验归一化，只处理连续字段并在采集/更新/推理共享统计，mask 保留原始 0/1。统计更新/冻结属于 S4/S5，尚未配置。不能以 donor normalizer checkpoint 自动适配新输入。
- S2 结构数值对照在同一份已准备好的数值输入上执行；它证明网络映射，不证明源/目标环境产生了相同物理观测。

### 模型公共接口与归属

- 源码根目录：`/Users/sss9999/enbodied_intelligence/ADAPT/Admittance`；当前仅新增纯模型和测试，未安装、发布或切换训练依赖，也未修改 site-packages。
- S2 owner：`src/unilab/models/anybody.py`，由 `src/unilab/models/__init__.py` 导出 `MUSETransformerEncoder`、`MUSEDecoder`、`MUSEEncoderDecoder`。仅依赖 Torch 和标准库；不导入环境、教师、`uni_rl` 或 donor `rsl_rl`。
- `split_obs(obs)` 返回 goal `[...,73]`、按时间排列的 proprio `[...,5,90]` 和 raw mask `[...,1]`；`encode(obs)` 返回原始均值 `mu[...,16]` 和 `z=normalize(mu)`；`decode(z,proprio_history)` 返回 raw action `[...,29]`；`forward(obs)` 组合上述映射。输出诊断需显式返回，不依赖上一次 forward 的隐藏缓存。
- latent 归一化保持 donor 的 `F.normalize(...,p=2,dim=-1,eps=1e-8)`；零向量保持有限，不将其描述成严格单位范数。
- 绑定参数：`d_model=192`、`nhead=4`、`num_layers=2`、`ffn_dim=768`、激活 GELU、dropout=0、latent=16；Decoder hidden=`[1024,512,256,128]`，输入 466，末层线性输出 29。Encoder 使用 `[CLS,goal,proprio_t-4,...,proprio_t]` 共 7 tokens；mask 仅屏蔽 goal。先保留确定性编码，不加入 latent 随机采样、KL 或训练损失。
- raw action 的缩放、默认姿态与 PD 控制继续属于 UniLab action/config。当前目标默认 scale=0.25，donor 使用按 actuator 计算的 scale；S2 不复制该缩放，也不宣称权重兼容。
- S4 再处理 actor wrapper 的概率分布与模型调用；已定位 `APPORunner._build_learner`、`APPOLearner._actor_mean_std` 的 `.mlp` 假设及 collector 的 `forward/get_output_log_prob` 调用。此时只登记缺口，不改训练器。

训练/推理边界：actor 消费 goal、当前/历史本体状态和显式 mask；其来源由 I3 逐项记录。参考轨迹中的未来目标若存在，必须标为命令输入，不能声称它来自部署传感器。critic-only 状态和教师动作不得混入 actor。若选择蒸馏，再登记教师、训练参数、监督标签及部署消费者；若选择残差，再登记零干预基线与可辨识性，不能沿用本轮未定义的目标。

S4 最便宜的否决检查：对固定观测分别走 collector 与 learner 的策略参数计算，比较动作均值及 log-prob；两者不一致就不能进入训练。仅 shape 一致不算通过。该检查不是 S1 的静态验收，当前未执行。

## system-engineering：六步总体路线

| 步骤 | 交付物与 owner | 真实依赖 | 完成标准 | 当前状态 |
| --- | --- | --- | --- | --- |
| S1 接口与边界 | 本看板；Anybody 参考模型、UniLab 配置和 `uni_rl` owner | 用户指定目标仓库 | I2–I4/I6–I7 与源码根目录明确；I5 按用户选择延后，不阻塞纯模型 | 已完成，E5–E8；源码/文档层 |
| S2 纯模型 | Admittance `unilab.models` 中的 Encoder、Decoder、组合模型 | S1 的 I2/I4 | 固定权重/输入与 donor 参考输出对照，梯度及错误输入检查通过 | 已完成；迁入 Admittance 后 39 项网络检查通过 |
| S3 AMASS 数据层 | Admittance 的人体序列读取、clip 内窗口及批次 | E10 格式核查及真实样本 | 原始序列可读、不串 clip、分组不泄漏，可输出含时间/来源的批次 | 已完成：384 条真实动作、108 项聚焦/回归测试，E11；模型联接留至 S4 |
| S4 异步训练接入 | Admittance 自定义 actor/runtime 与 owner YAML；复用外部 `uni_rl` | S2、S3；I5 必须确定 | 同步权重下采集/更新语义一致，授权参数实际更新，生命周期复用 | 未开始 |
| S5 保存与推理 | `uni_rl` checkpoint；UniLab playback、sim2sim、导出 | S4 模型/训练状态确定 | fresh reload 动作一致，恢复训练与错误配置拒绝有效，完整推理导出一致 | 未开始 |
| S6 整链验收 | focused tests、短异步集成及最终回归 | S2–S5 | 短链运行/保存/恢复/退出证据齐全，默认路径回归通过 | 未开始 |

S3 的原始 AMASS 读取可以先于最终训练路线开展；人体特征到当前 G1 网络的映射不能据旧 S1 自动成立。S4 的优化更新须等待数据与监督接口明确。本表描述依赖，不授权自动开启多个 agent。

## engineering-plan：每步工程包

S1–S3 已执行，S4–S6 文件名和命令仍为后续计划。所有自定义实现与测试均归 Admittance；临时环境仅用于测试，不修改外部包的生产源码。

### S1 — 接口与边界收敛

- **Change / Why**：核对 donor `LatentBottleneckMUSETransformer` 与 KP 变体、UniLab `ObservationManager` / task YAML、`uni_rl.APPORunner._build_learner` 与 `APPOLearner._actor_mean_std`，将 I2–I7 收敛为可执行的输入/输出表，定位 `uni_rl` 开发源码。
- **Preserve**：用户当前只要求架构接入；沿用包边界及默认训练路径。
- **Allowed scope**：本看板中的模型选择、参数表、训练/推理输入和 owner 路径；只读相关源码。
- **Forbidden scope**：自行选定新监督目标、创建教师训练任务、改 reward、安装或替换训练包、修改运行代码。
- **Depends on**：I1 已确认。
- **Done when / Check**：逐项核对所选 donor 构造参数、目标 YAML 的观测项及 uni_rl 调用签名；每项都有确定 owner，未决项不再影响 S2。执行结果：已通过静态接口核对，见 E5–E8；训练方式依用户选择留至 S4。
- **Stop**：发现模型变体或训练模式改变用户预期，先解释具体差异；不得靠维度凑齐、默认零填充或将 proposal 改名为 active contract 来绕过。

### S2 — Encoder–Decoder 纯网络

- **Change / Why**：已在 Admittance 的 `unilab.models.anybody` 增加 Encoder、Decoder 和组合接口；移植所选结构、拆分逻辑及 latent 归一化，去除 Isaac/任务环境耦合。实现边界符合 ANYBODY-IO-v1。
- **Preserve**：donor 的输入解析与网络数学行为；模型只依赖 Torch 和明确的包内公共模块。
- **Allowed scope**：新模型模块、必要导出、`tests/models/test_anybody_models.py`，以及独立参考所需的 `tests/fixtures/anybody/generate_reference.py` / `muse_transformer_v1.npz`。后两者仅服务数值对照，不是运行依赖。
- **Forbidden scope**：collector、learner 损失、IPC、机器人动作缩放、奖励、数据下载、复制整套 Anybody `rsl_rl`。
- **Depends on**：I2/I4 和模型 owner 已冻结；当前已满足。S2 使用上述 ANYBODY-IO-v1 纯网络接口，不决定训练模式。
- **Done when / Check**：`uv run pytest tests/models/test_anybody_models.py -q`；对照固定 donor 权重/输入验证数值、批维、非零可学习梯度，并检查错误布局及零 latent 的数值行为。
- **Stop**：必须访问仿真状态或改变上游网络行为才能通过；回到 S1 解释接口问题。

S2 验收结果（2026-10-15，CPU）：

- 新增 5 个文件：模型及 `__init__.py`、聚焦测试、参考生成工具、约 24 KB 的 NPZ 参考。参考生成器只执行固定 donor 的原始 Encoder 和 Decoder 构造/前向代码，不导入移植模型；固定源文件身份、输入和权重种子。常规测试只读 NPZ，不需要 donor 临时目录、网络或 Isaac。
- 纠正后精确命令（cwd=`ADAPT/Admittance`）：`PYTHONPATH=src UV_CACHE_DIR=/private/tmp/anybody-s2-uv-cache uv run --no-project --python /Users/sss9999/enbodied_intelligence/unilab/unilab_stand/.venv/bin/python -- python -m pytest --confcutdir=tests/models tests/models/test_anybody_models.py -q`；结果 **39 passed in 1.21s**。显式 PYTHONPATH 确保读取当前 Admittance 源码。复用既有 Torch 2.7.0 / pytest 9.0.2，未安装依赖；`--confcutdir` 隔离公共 conftest 对未安装 `uni_rl` 的依赖，不修改或伪造训练包。先前正常加载公共 conftest 的尝试因 `ModuleNotFoundError: uni_rl` 失败。
- 对照涵盖 raw mu、归一化 z、历史和动作，阈值 `atol=1e-6, rtol=1e-5`；另验收历史逐元素排序、Decoder 输入、mask 屏蔽/可见语义、潜变量干预、非零有限梯度、单样本/多批维/非连续输入、零 latent、错误布局/配置/dtype/非有限值拒绝。
- 纠正后 Ruff lint / format 通过，mypy 新模型目录通过，Pyright 为 0 errors / 0 warnings。质量命令使用同一 `uv run --no-project --python <既有解释器> -- python -m` 前缀，参数为 `ruff check src/unilab/models tests/models/test_anybody_models.py tests/fixtures/anybody/generate_reference.py`、`ruff format --check` 同路径、`mypy src/unilab/models`、`pyright --pythonpath <既有解释器> src/unilab/models`。
- 完整 gate 尝试：`UV_OFFLINE=1 UV_NO_SYNC=1 UV_CACHE_DIR=/private/tmp/anybody-s2-uv-cache make check` / `make test`。因项目尚未安装 ruff / pytest 分别退出 2，不能声明全仓通过。uv 创建的空 `.venv` 随后清理；未同步依赖或修改锁文件。
- 等级：network-test-confirmed / static-check-confirmed。没有修改既有训练代码、包配置或锁文件；没有运行仿真、优化器训练、GPU 验证、导出或完整 PR gate，不将网络测试升级为策略质量证据。

### S3 — AMASS 数据读取、窗口与批次（已完成）

- **Change / Why**：先建立原始 AMASS 人体序列入口，纠正“已有 G1 NPZ 就是 AMASS 接入”的错误。原项目中未找到可直接移植的原始 AMASS reader；不能复制机器人 loader 后改名充当 AMASS reader。
- **Preserve**：当前 S2 网络数学行为、默认 G1 task、外部 `uni_rl` 依赖方向、序列时间语义。训练目标仍由用户在 S4 选择。
- **Allowed scope / 实际文件**：`src/unilab/data/amass.py` / `__init__.py` 及读取器测试；补充 `amass_download.py`、`scripts/motion/download_amass.py` 和模拟下载测试，供远程服务器显式准备固定版本子集。项目仅保留 `datasets/amass/README.md`；manifest/splits 在远程下载后生成。`.gitignore` 覆盖 raw/archives/cache。未改 S2 网络、G1 task 或项目依赖/锁文件。
- **Forbidden scope**：在本地重新下载训练数据或启动训练、下载整个 AMASS/人体模型、伪造机器人状态/动作标签、自行选择重建或控制损失、改写 S2 输入输出。最新远程数据要求覆盖此前本地下载要求。
- **Depends on**：E10 格式核查和 E11 真实数据验收已完成；远程实际数据准备需要服务器环境/连接条件。原本完整逐文件清单已不在工作区；当前固定源版本及筛选规则见远程准备模块。
- **S3.1 原始读取**：`AmassClip` / `load_amass_clip(path, ...)` 保留人体参数、帧率和来源；检查帧数一致、有限值、合法 fps、已声明格式。不能确定的模型/坐标信息保留为未确认，使用它进行特征转换前必须明确；不能仅凭维度猜测人体模型。
- **S3.2 窗口与批次**：`AmassWindowDataset` 按显式 window_length / stride 在单条 clip 内切片，只缓存一条解码序列，支持 Torch DataLoader。`split_amass_files` 支持先按 clip 分组，或由调用者提供明确 subject 分组后划分。本子集使用源 manifest 的 CMU subject 信息实现受试者隔离；保留原始帧率和精度。短序列在 dataset 中显式拒绝，本次 64 帧示例的两条短文件保留且列入 exclusions，不补零。
- **S3.3 数据验收**：合成带标记帧的样例覆盖字段/帧率/有限值错误、窗口顺序、不跨 clip、划分不泄漏和短序列处理；再读取一份真实 AMASS 文件并构建一批窗口，报告原始与批次形状/时间。核对数据模块不改动当前 S2 模型与默认训练配置。
- **Done when / 实际证据**：读取器及网络/loader/reset 的 108 项回归通过，加入远程准备模拟检查后共 116 项通过。E11 曾逐份验证 384 条真实文件及所有 382 条可用 clip 的首/尾窗口、60/120 Hz 混合时标；该历史数据现不在本地。远程脚本当前仅有模拟执行和源列表元数据验证，未宣称已在服务器下载成功。详细检查限制见 E11/E12。
- **Stop**：数据变体、坐标系或标签来源无法确认，或需要改变网络输出语义时，明确缺口并回到相应接口决定；不将人体数据零填充成 524 维，不从人体关节姿态直接生成未经定义的控制标签。

### S4 — 接入现有异步训练器

- **Change / Why**：在 Admittance 实现自定义 actor/runtime，通过已有可配置类和 `runtime_resolver` 复用外部 `uni_rl`，处理 `_actor_mean_std` 对 `.mlp` 的调用假设。采集、target actor 和更新必须走同一模型定义；接口能否满足训练模式留待 S4 验证。
- **Preserve**：`EnvFactory` 注入、既有 AsyncRunner 生命周期、共享权重协议、V-trace 与 timeout 语义、默认 MLP/APPO/HORA。
- **Allowed scope**：Admittance 内自定义策略/训练器扩展、专用配置和组装；拟新增仓内 `tests/training/test_anybody_appo.py` 与 `tests/scripts/test_anybody_appo.py`。复用现有外部 runner/learner 基类；不复制整个训练包。
- **Forbidden scope**：私建异步同步协议、`uni_rl -> unilab` 反向 import、未选择训练模式便改缓存 schema，或将训练损失塞进 UniLab 入口脚本。
- **Depends on**：S2/S3；I5 和 AMASS 特征/监督接口冻结。若选原项目式控制蒸馏，必须有机器人参考、环境状态及教师动作来源；若选人体重建，须明确人体特征和输出头，不能直接套用当前 G1 的 524→29 接口。两者尚未选择，本轮不据原项目有教师就自动授权教师训练。
- **Done when / Check**：在 Admittance 根目录运行 `uv run pytest tests/training/test_anybody_appo.py tests/scripts/test_anybody_appo.py -q`。同权重下采集/更新均值与 log-prob 一致，E/D 的实际参数变化符合冻结设置，权重同步含模型 buffer。
- **Stop**：相同输入在两条路径含义不同、错误参数被更新、或既有扩展接口不足；回到总体规划说明具体限制，不自行创建外部 fork 或扩写整套 runner。

### S5 — 保存、重载与确定性推理

- **Change / Why**：在 Admittance 的自定义 runtime 中复用外部 `uni_rl` 保存/恢复接口持久化模型和必要训练状态；在 UniLab `visualization/interactive_playback.py`、`scripts/train_appo.py`、`training/onnx_export.py`、`utils/sim2sim.py` 的必要边界接入完整模型推理，替换仅导出 `.mlp` 的假设。
- **Preserve**：旧 checkpoint 与默认 MLP playback 路径；导出/播放不要求教师或 critic-only 输入；跨后端严格校验。
- **Allowed scope**：仓内新模型的结构/layout/归一化元数据及上述调用点；拟新增 Admittance 的 `tests/training/test_anybody_checkpoint.py`、`tests/training/test_anybody_inference.py`。
- **Forbidden scope**：静默部分加载、靠补零适配错误观测、重写已有 checkpoint、引入训练期特权输入到导出图。
- **Depends on**：S4 明确模型/优化器状态；可在 S2/S3 后先准备确定性推理检查。
- **Done when / Check**：在 Admittance 执行 `uv run pytest tests/training/test_anybody_checkpoint.py tests/training/test_anybody_inference.py tests/training/test_sim2sim_resolver.py -q`。独立重载动作一致、恢复后仅授权参数继续变化；错误 layout 在加载前拒绝；ONNX 与完整 PyTorch 推理对照。
- **Stop**：导出绕过 Encoder/归一化/Decoder，或重载只能依赖原进程隐藏状态。

### S6 — 整链验证与验收

- **Change / Why**：拟新增 UniLab `tests/integration/test_anybody_async.py`，串起专用配置、真实 spawn collector、短 rollout/update、checkpoint、fresh resume 和退出清理；补最接近风险的默认路径回归。
- **Preserve**：现有支持范围；通过短链验证不得写成策略已收敛、类人动作已保证或真机可用。
- **Allowed scope**：集成测试及最小验收配置、同一看板的证据和状态。拟新增测试需带 `slow` 标记并限定运行预算。
- **Forbidden scope**：长期训练、全数据集扫描、性能宣传、硬件控制和发布操作。
- **Depends on**：S2–S5 和可复现的开发依赖绑定。当前 sdist 不包含所有历史 AMP 扩展；若既有回归受此影响，记录为基线依赖问题，不自动吸收 AMP 修复。
- **Done when / Check**：后续明确执行短运行时，`uv run pytest tests/integration/test_anybody_async.py -m slow -q`，限制少量环境和两次更新；完成保存/恢复一次、collector 正常退出，无残留共享资源。随后按仓库规则执行 `make check`、`make test`；只有创建/更新 PR 时再满足 `make test-all`。
- **Stop**：包身份不明、进程清理失败、非有限数值或默认路径回归；记录原始失败后只修对应 owner。

## 当前验收视图

| 项目 | Owner / 验收引用 | 状态 | 证据层级 |
| --- | --- | --- | --- |
| 目标仓库确定 | 用户选择 / E2 | 已确认 | human-confirmed |
| 新版职责与依赖检查 | Admittance + published uni_rl / E1 | 已检查 | code-confirmed |
| 总体路线及逐步工程包 | 本看板 / E3 | 已记录为 proposal | note-confirmed |
| S1 接口冻结 | ANYBODY-IO-v1 / E5–E8 | G1 基线已完成；AMASS 入口不受其维度绑定 | code-confirmed + document-confirmed |
| S2 纯网络验收 | Admittance 内本节 S2 验收结果 | 已完成；39 项网络检查通过；全仓 gate 缺依赖 | network-test-confirmed |
| S3 原项目核查 | E10 的版本固定源码链路 | 已完成；发现原项目无 raw AMASS reader | code-confirmed |
| S3 数据层 | E11；384 份文件的哈希/读取和批次证据 | 已完成；当前子集/packed poses 格式 | data-runtime-confirmed |
| S4–S6 | 各工程包的 Done when | 未开始；用户要求远程训练 | unconfirmed |

## 证据记录

### E11 — 2026-10-15：S3 数据实现及项目内真实子集验收

以下是此前验收记录；当前数据存放和交付要求已由 E12 更新。本地原始数据与逐文件清单现已不存在，不将本节当作当前文件清单。

- 用户新增要求：自行联网获取适当规模数据并放进项目，供后续训练；明确本地不训练，后续走远程 SSH。本轮没有优化器更新、训练 rollout、网络结构修改、Git 提交/推送或 SSH 操作。
- 实现：`load_amass_clip` / `AmassClip` 读取 packed poses 原始 NPZ，保留 poses/trans/betas/gender、DMPL、fps、源字段及显式模型声明；无声明保持 unknown，不按 156 维推断。支持两种 fps 键并拒绝冲突。`AmassWindowDataset` 提供 clip 内完整窗口和可复制批次，保留每个样本的时间/文件/帧位置；不同人体参数布局不可混用。`split_amass_files(..., groups=...)` 允许可靠身份分组，避免先切窗再随机划分的泄漏。
- 数据：公开第三方镜像 [realdream-ai/AMASS](https://huggingface.co/datasets/realdream-ai/AMASS/tree/d38490c0e60dcefae81461fc5f4765fe66727831/raw/CMU)，revision `d38490c0e60dcefae81461fc5f4765fe66727831`。下载 384 条 / 99 个 CMU subjects / 513,624,784 bytes（489.83 MiB）/ 384,010 帧，按帧数/fps 约 61.92 分钟；307 条 120 Hz、77 条 60 Hz。每个文件与源 LFS SHA256/size 对照，并由本轮主实现重新读取、重算 SHA256 验证；零失败。原始人体参数和数据模型权重是两回事，本轮未获取 SMPL 权重。
- 可复用划分：seed `20261015`，先按明确 CMU subject 分为 89/10 人，再按 64 帧、stride=16 示例过滤 2 条 20/21 帧短动作。训练 351 条 / 21,017 窗口，验证 31 条 / 1,659 窗口；受试者与 clip 均不重叠。两条短原文件保留，exclusions 写入 splits。64/16 是数据验证设置，不是 S4 已确定的训练超参数。
- 真实数据检查：所有 382 条可用 clip 的首/尾窗口身份、边界、fps 和有限值通过；训练/验证均用 DataLoader 取 batch，poses `[16,64,156]`、trans `[16,64,3]`、DMPL `[16,64,8]`。另对真实 60/120 Hz 各一条构建混合 batch，逐值核对原始 frame_times，未默认重采样或 float32 转换。
- 聚焦测试命令（cwd=Admittance）：`PYTHONPATH=src UV_CACHE_DIR=/private/tmp/anybody-s3-uv-cache uv run --no-project --python /private/tmp/anybody-s3-venv/bin/python -- python -m pytest tests/data/test_amass.py tests/models/test_anybody_models.py tests/envs/test_motion_loader.py tests/managers/test_observation_partial_reset.py -q` → **108 passed in 1.16s**。包含字段/错误/时间/分组/可变帧率/默认 Torch 批次与序列化，以及已有网络、机器人 loader 和部分 reset 回归。首轮 92 项通过后补充受试者隔离，再执行此最终集合；没有以测试触发训练。
- 静态质量：新 data 模块和测试的 Ruff lint/format、mypy、Pyright 均通过。全仓 `ruff check src tests` 和测试 F401/F821/F811/F841 检查通过；全仓 `mypy src/unilab` 为 157 个源码文件无问题。
- 全仓限制：`ruff format --check src tests` 提示两个未改的 AMP 文件需格式化；没有自动格式化无关文件。`make -o format check` 在临时环境因缺 mypy 命令入口退出 2，随后通过同环境 `python -m mypy` 验证全仓。全仓 Pyright 为 3 errors / 12 warnings，错误位于未改的 `base/entity.py`（unisim sensor-history 接口）和 `scripts/train_appo.py`（外部 APPORuntime.runner_kwargs），当前依赖与仓库既有接口不一致；新 data 模块无错误。为遵守不在本地训练的要求，未跑含训练 smoke 的全量 pytest，不宣称完整 make/PR gate 通过。
- 临时测试环境复用既有 Torch 2.7.0 / NumPy 2.4.4，并隔离安装锁定的外部 runtime；它不是已满足项目 Torch 下限的正式训练环境。正式远程训练依赖配置仍归 S4，本轮未改变 pyproject/uv.lock。
- 当时交付状态：代码/测试、`.gitignore`、本协作看板、数据 README/manifest/splits 尚未提交；raw 当时位于项目并受 gitignore 保护。后续用户明确禁止本地保留/下载数据，当前交付按 E12 改为远程准备入口，不同步本地 raw。
- 结论：S3 **数据层完成**；人体数据到网络特征、重建/控制标签及异步优化仍由 S4 决定，不声称策略已训练或机器人可执行。

### E12 — 2026-10-15：改为远程准备数据，并交付用户 GitHub

- 用户明确要求不在本地下载数据，并提交自己的 GitHub。检查时 `datasets/amass/` 已不存在；没有执行本轮原始数据删除或重新下载，未找到 manifest/splits 备份，不声称恢复原清单。
- 新增 `unilab.data.amass_download.prepare_amass_subset`，由 `scripts/motion/download_amass.py` 仅负责 CLI 组装。固定 E11 的第三方源 revision；按 subject 轮选 384 条，单条不超过 4 MiB、总计不超过 512 MiB。下载逐份校验 SHA256/size，并调用已有 reader；已验证文件可跳过，损坏既有文件报错，失败清理 `.part`。远程生成 manifest 及 subject 隔离 splits，64/16 仍只是预览设置。
- 最小工程包：仅数据 owner、CLI、相关测试和说明；保留 S2 网络/默认配置/外部训练器依赖。验收为模拟网络下来源选择、分页、哈希失败/清理、重跑跳过、受试者隔离；本地不调用真实原始动作下载。
- 聚焦命令：`PYTHONPATH=src UV_CACHE_DIR=/private/tmp/anybody-s3-uv-cache uv run --no-project --python /private/tmp/anybody-s3-venv/bin/python -- python -m pytest tests/data tests/models/test_anybody_models.py tests/envs/test_motion_loader.py tests/managers/test_observation_partial_reset.py -q` → **116 passed in 1.14s**。新增下载测试只使用生成字节和模拟 HTTP；未运行优化器或训练 smoke。
- 源接口单独读取公开文件列表元数据，确认分页、LFS `oid` 和大小字段；新规则选出 384 条 / 93 个 subjects / 534,899,312 bytes（510.12 MiB），未获取 NPZ。这是新准备规则的元数据结果，并非 E11 丢失清单的精确恢复；不把此检查记作服务器真实下载验收。新数据 owner 的 Ruff lint/format、mypy 和 Pyright 全部通过；全仓依赖/格式限制仍见 E11。
- Git 交付目标仅为 `origin=https://github.com/Andisyc/Admittance` 的现有 `feat/anybody-muse-s2` 分支。原始数据不提交；无 SSH 信息，因此尚未在远程执行准备命令。下一步仍为 S4 特征/监督和远程运行规划，不自动开始训练。

### E10 — 2026-10-15：AMASS 目标纠正与原项目链路核查

- 此历史核查回合只读取固定版本 `hazel-hammer/Anybody@0cffa139eab0bc145c9216223e249f03c67a324c` 并更新看板，当时未改生产代码或运行训练。之后的 S3 数据实现与当前状态以 E11 为准。
- 原项目链路：**人体动作经外部重定向 → 机器人 CSV → 机器人 NPZ → 仿真参考/机器人观测 → 教师动作监督 → Encoder/Decoder 学生控制策略**。README 的公开 recipe 使用 Bones Studio 数据；所选启动脚本的默认目录是 `SONIC_npzs/g1/npz_by_motion/train`。二者均不能证明当前版本提供原始 AMASS 解析或 AMASS 人体重建训练。

| 核查对象 | 源码事实及证据 | 对本项目的影响 |
| --- | --- | --- |
| 数据说明 | [README 数据准备](https://github.com/hazel-hammer/Anybody/blob/0cffa139eab0bc145c9216223e249f03c67a324c/README.md#data-preparation) 明确先重定向，再转换 CSV 为 NPZ；训练步骤先教师后学生 | 原始 AMASS 读取不能假定已包含在待移植模块中 |
| CSV 转换 | [csv_to_npz.py](https://github.com/hazel-hammer/Anybody/blob/0cffa139eab0bc145c9216223e249f03c67a324c/scripts/csv_to_npz.py#L181) 使用机器人关节名，写入机器人状态后导出 joint/body 数组；[motion_csv_loader.py](https://github.com/hazel-hammer/Anybody/blob/0cffa139eab0bc145c9216223e249f03c67a324c/scripts/motion_csv_loader.py#L20) 输入已含 root 和机器人 DOF | CSV→NPZ 是导出/重采样，不是 raw AMASS→G1 的重定向算法；脚本写状态不运行物理积分，不证明轨迹动力学可行 |
| 运行时数据 | [commands.py 的 MotionLoader](https://github.com/hazel-hammer/Anybody/blob/0cffa139eab0bc145c9216223e249f03c67a324c/source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/commands.py#L391) 与 MultiMotionLoader 读取 fps、joint_pos/joint_vel、body_pos/quat/lin_vel/ang_vel_w | 读取的是预处理后的机器人轨迹；这些字段不等于 AMASS 人体参数 |
| 模型输入 | [MUSETransformerDistillationTrackingEnvCfg](https://github.com/hazel-hammer/Anybody/blob/0cffa139eab0bc145c9216223e249f03c67a324c/source/whole_body_tracking/whole_body_tracking/tasks/tracking/tracking_env_cfg.py#L970) 组合参考误差、实际机器人状态、5 帧历史和 last_action | 原始 AMASS 文件本身不提供机器人实际状态和控制历史；不能直接送入当前 524 维输入 |
| 训练监督 | [muse_distillation.py 的 act/update](https://github.com/hazel-hammer/Anybody/blob/0cffa139eab0bc145c9216223e249f03c67a324c/source/rsl_rl/rsl_rl/algorithms/muse_distillation.py#L235) 在线收集教师动作，在第 288 行计算学生动作与教师动作的行为损失；[启动脚本](https://github.com/hazel-hammer/Anybody/blob/0cffa139eab0bc145c9216223e249f03c67a324c/run/train/MUSE/run_muse_transformer_distillation.sh#L29) 传入 motion 与 teacher_checkpoint | 原项目的 Encoder/Decoder 是控制策略蒸馏，不是人体姿态重建自编码器。本项目是否复用此监督仍待 S4 决定 |
| 名称相似的离线 reader | [motion_data_loader.py](https://github.com/hazel-hammer/Anybody/blob/0cffa139eab0bc145c9216223e249f03c67a324c/source/whole_body_tracking/whole_body_tracking/utils/motion_data_loader.py#L148) 返回 305 维输入及 3 维参考 anchor 速度；消费者是 [train_ref_vel_estimator.py](https://github.com/hazel-hammer/Anybody/blob/0cffa139eab0bc145c9216223e249f03c67a324c/scripts/train_ref_vel_estimator.py#L296) | 它是参考速度估计器的数据工具，不是 MUSE 或 AMASS 训练 reader。其拼接各文件、过滤帧后统一切窗口的逻辑也不能直接作为新 reader 的序列采样实现 |
| SMPL 分支 | `robots/smpl.py` 与 `config/humanoid/flat_env_cfg.py` 定义的是仿真 SMPL humanoid，并引用已导出的 walk.npz | 有 SMPL 仿真模型不等于已实现原始 AMASS 数据读取 |

- 检索边界：对当前固定 checkout 执行 `git grep -n -i amass -- .` 无匹配；另外检索原始数据常见字段 `mocap_framerate`、`mocap_frame_rate`、`dmpls`、`betas`、`poses`/`trans` 读取形式，并逐项读取上述数据/模型/训练入口。未找到 raw AMASS parser 是本版本的源码检查结果，不是对其他分支或外部预处理工具的断言。[AMASS 官方说明](https://amass.is.tue.mpg.de/) 确认其发布人体模型参数，不能与机器人 joint/body NPZ 混同。
- 当时缺口：未取得真实 AMASS 文件，且网络输入输出/监督尚未定义为人体预训练接口。样本缺口现已由 E11 的真实子集解决；网络/监督接口仍归 S4。已有 G1 轨迹和 donor fixture 不作为 AMASS 数据证据。
- 文档核验：通过标准库 AST 检查确认固定 donor 的机器人 NPZ 字段、教师动作监督和速度估计器消费者；验证 13 个版本固定源码/本地链接。`git diff --check` 通过，工作区只有本看板变化；没有把静态源码核查记成模型或数据运行测试。
- 等级：code-confirmed / document-confirmed。该回合确定的下一实施包（AMASS 原始序列读取/采样）已在 E11 完成；没有据此自动选择人体重建、教师蒸馏或 latent RL。

### E1 — 基线源码与包边界
- 初始 Admittance 为 `70afeb9ac9708ec1320e3d62bdc7923d73abf957`，工作树干净。锁定的 unilab-rl 1.0.0 sdist SHA-256 为 `b2ab788a99054b2a75183b08c76c9cb5e70af5a6ed749316106a082ded9fbe16`；已确认 runner/learner/collector 在外部包，环境由可 pickle 的 factory 注入。

### E2 — 目标仓库
- 用户选择当前 ADAPT/Admittance；旧 unilab_stand 仅作参考。用户后来再次纠正代码交付目标，S2 最终提交位于本仓 `52466db7`，误建个人训练包 fork 已由用户手动删除。

### E3 / E4 — 看板与文档检查
- 本文件是唯一当前协作入口。初版 owner/链接和空白检查通过；这些属于文档证据，不是模型、环境或训练验收。详细历史见 Git。

### E5 — 训练选择延后
- 用户明确选择“先完成网络与观测接入，训练方式到 S4 再定”。后续明确 AMASS 是训练数据来源；没有据人体重建/机器人控制的解释讨论自动选择其中一条路线。

### E6 — G1 模型/观测语义
- 对照 donor 模型、配置、观测函数与目标公开 entity/manager 接口，形成上方 ANYBODY-IO-v1。已分清参考/实际状态、pelvis/torso、term-major/frame-major、raw/scaled action，以及 donor COM 与目标 link 原点语义差异。该接口只证明 G1 基线的工程定义，不证明原始 AMASS 可直接作为输入。

### E7 — 训练包源码参考
- 为核查已克隆 unilab_rl tag v1.0.0，commit `ce0050ecc77ec17de84b6b581148c67e1ae03caa`，当时 src/uni_rl 与 sdist 无差异。该目录当前仅作参考；自定义代码和交付归 Admittance，不再作为另一开发仓库。

### E8 / E9 — S1 静态验收
- 通过 donor 配置 AST 和布局算术检查：goal=73、每帧 proprio=90、H=5、obs=524、Decoder 输入=466；链接及看板状态检查通过。S2 的后续 39 项网络测试及依赖限制见 S2 节，不能将早期静态验收当作 AMASS 或运行时验收。
