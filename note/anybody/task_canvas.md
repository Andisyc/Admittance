# Anybody Encoder–Decoder 接入看板

更新日期：2026-10-15（Asia/Shanghai）。唯一当前看板：本文件。

## 当前游标与授权

- 用户已确认：在当前 `ADAPT/Admittance` 中接入 Anybody Encoder–Decoder，并以多步、从粗到细的方式规划异步训练链路。
- 当前授权：用户已要求完成 S2，并将本次代码与进度记录提交到 GitHub；不接入训练器或切换 Admittance 运行依赖。
- 当前进度：**S1、S2 已完成，阶段验收 2/6**。S2 已通过 CPU 网络数值、梯度、错误输入及包边界检查；实际环境和异步训练链路尚未接通。
- GitHub 交付：网络源码与测试已推送至 [sss99991112/unilab_rl 的 S2 提交](https://github.com/sss99991112/unilab_rl/commit/253c1231b006cef16bc655baa0a77415b4219a53)，分支 `feat/anybody-muse-s2`。上游无写权限，因此创建个人 fork；本看板随 Admittance 同名功能分支提交。推送前重跑 39 项模型检查及 1 项包边界检查，合计 `40 passed in 0.44s`。GitHub 提交不代表 S3–S6 链路验收。
- 下一步：规划 S3 观测与机器人动作数据适配；本轮止于 S2。S2/S3 不选择训练目标，训练方式由用户明确延后到 S4（E5）。
- 当前工程依据：用户接受的 S1 范围及下列 `ANYBODY-IO-v1` 接口记录。尚无 active training/method contract；训练部分继续保持 proposal，不将接口记录升级为策略或运行证据。
- 维护方式：只更新本文件的当前状态、工程包和检查项；证据按 E 编号追加。无需另建 registry、JSON 清单、重复计划或独立日志文件。

## 系统边界与证据基线

终点：Encoder–Decoder 可在当前 Admittance 通过配置构建，接收明确语义的观测，经现有异步训练链路更新，并完成保存、重载及确定性推理验证。

范围外：外力补偿分支、PCGrad、导纳目标、奖励重设计、全量 AMASS 重定向、长期训练、策略效果结论和硬件部署。冻结 Decoder 的 latent RL 属于待选择的后续训练模式，不自动纳入此次架构接入。

| 对象 | 身份/现状 | 证据等级 |
| --- | --- | --- |
| 当前目标 | `Andisyc/Admittance@70afeb9ac9708ec1320e3d62bdc7923d73abf957`；创建本文件前工作树干净 | code-confirmed |
| 模型参考 | `hazel-hammer/Anybody@0cffa139eab0bc145c9216223e249f03c67a324c` | code-confirmed |
| 训练包基线 | 当前 `pyproject.toml` / `uv.lock` 固定 `unilab-rl==1.0.0`；已读取同版本 PyPI sdist，未安装 | code-confirmed |
| 训练源码 owner | `../unilab_rl`，基线 tag `v1.0.0` / commit `ce0050ecc77ec17de84b6b581148c67e1ae03caa`；检出时与 sdist 无差异。S2 commit `253c1231b006cef16bc655baa0a77415b4219a53` 已推送个人 fork；未接为 Admittance 运行依赖 | code-confirmed |
| 旧 UniLab | `unilab_stand@e1717b9…` 只作历史参考；其 `src/unilab/algos/torch/*` 路径不是当前修改目标 | code-confirmed |
| 既有笔记 | `note/upstream_migration/` 是 AMP 历史规划/执行记录，不授权本主题；不修改其状态 | note-confirmed |

当前 owner 链路：

```text
UniLab 机器人动作数据 / task observation terms
  -> ObservationManager（历史与 reset）
  -> UniLab Hydra 配置 / train_appo（组装）
  -> uni_rl APPORunner -> collector / learner / 权重同步
  -> Encoder -> latent -> Decoder -> 关节动作
  -> UniLab action manager -> unisim

训练状态 -> uni_rl checkpoint -> UniLab playback / ONNX 组装
```

模型位置在 S1 绑定到 package-neutral 的算法/模型 owner；算法、collector、learner、IPC 不迁回 UniLab。`uni_rl` 不导入 `unilab` / `unisim`，环境由可 pickle 的 `EnvFactory` 注入。

现有约束来自 [AGENTS.md](../../AGENTS.md)、[层级边界](../../docs/sphinx/source/en/4-developer_guide/1-architecture/3-layer_boundaries.md) 及 [观测契约 ADR-0005](../../docs/sphinx/source/adr/ADR-0005-unified-obs-critic-env-and-ipc-contract.md)。ADR-0007 中尚有拆分前的 runner 归属表述；当前实际归属按 AGENTS、层级页、依赖及入口 import 共同确认，不能据旧表述恢复仓内算法实现。

## S1 接口结果：ANYBODY-IO-v1

状态：S1 工程接口已收敛，可供 S2/S3 使用；训练模式不在该接口版本中选择。模型结构参考 MUSE-Transformer，观测/动作遵循目标 UniLab 的明确约定；不声明 Anybody 旧 checkpoint 的兼容性。

| ID | 接口/不变量 | 当前状态 |
| --- | --- | --- |
| I1 | 目标是当前 Admittance；算法变更遵循外部 `uni_rl` owner | 用户确认 / 源码确认 |
| I2 | 移植 `_MUSETransformerEncoder` 和条件式 MLP Decoder；KP Encoder、teacher 网络及 donor 训练器不属于 S2 | S1 已绑定 |
| I3 | goal/proprio/mask 的顺序、公式、单位、历史与 reset 见下表；`critic` 不进入模型 | S1 已绑定；S3 尚未实现 |
| I4 | G1 起始配置为 latent=16、H=5、每帧 proprio=90、原始关节动作=29；网络维度由配置计算 | S2 已实现并通过 CPU 测试 |
| I5 | 网络内部 latent 不等于算法采样动作；先完成网络与观测接入，首个训练方式延后到 S4 | 用户确认延后，E5；不阻塞 S2/S3 |
| I6 | 单一纯模型语义供采集/更新/推理复用；normalizer 不进入纯模型，未来 actor wrapper 对预处理负责；mask 保持 0/1 | S1 已绑定；异步一致性待 S4 |
| I7 | 仅接受已按目标机器人顺序导出的轨迹；不把原始 AMASS 当机器人 NPZ；不推断 donor 关节/刚体顺序等于目标顺序 | S1 已绑定；实际数据核验待 S3 |

### 精确输入与数据来源

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

- 源码根目录：`/Users/sss9999/enbodied_intelligence/ADAPT/unilab_rl`，以 `v1.0.0` 为基线新增 S2 模型与测试；未安装、未发布、未切换 Admittance 的依赖，也未修改 site-packages。
- S2 owner：已新增 `src/uni_rl/algos/anybody/models.py` 及包初始化文件；类名 `MUSETransformerEncoder`、`MUSEDecoder`、`MUSEEncoderDecoder`。仅依赖 Torch 和标准库；不携带 IsaacLab、UniLab、教师或 donor `rsl_rl`。
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
| S2 纯模型 | package-neutral Encoder、Decoder、组合模型 | S1 的 I2/I4 | 固定权重/输入与 donor 参考输出对照，梯度及错误输入检查通过 | 已完成；39 项网络检查＋1 项包边界检查通过 |
| S3 观测与数据 | UniLab task terms、ObservationManager 配置、MotionLoader 适配 | S1 的 I3/I7；联接依赖 S2 | 真实 task 布局可被模型解析；历史/部分 reset/动作映射正确 | 未开始 |
| S4 异步训练接入 | `uni_rl` 模型接口/训练器；UniLab runtime resolver 与 owner YAML | S2、S3；I5 必须确定 | 同步权重下采集/更新语义一致，授权参数实际更新，生命周期复用 | 未开始 |
| S5 保存与推理 | `uni_rl` checkpoint；UniLab playback、sim2sim、导出 | S4 模型/训练状态确定 | fresh reload 动作一致，恢复训练与错误配置拒绝有效，完整推理导出一致 | 未开始 |
| S6 整链验收 | focused tests、短异步集成及最终回归 | S2–S5 | 短链运行/保存/恢复/退出证据齐全，默认路径回归通过 | 未开始 |

S1 冻结后，S2 纯模型与 S3 数据/观测准备可以独立开展；S4 必须等待二者接口验收。本表描述依赖，不授权自动开启多个 agent。

## engineering-plan：每步工程包

下列新增文件名是拟定 owner，不表示文件已存在。验收命令也是未来执行计划，本轮均未运行。`uni_rl` 命令必须在 S1 绑定的开发源码根目录执行；不得在临时 sdist 或 site-packages 内直接改生产代码。

### S1 — 接口与边界收敛

- **Change / Why**：核对 donor `LatentBottleneckMUSETransformer` 与 KP 变体、UniLab `ObservationManager` / task YAML、`uni_rl.APPORunner._build_learner` 与 `APPOLearner._actor_mean_std`，将 I2–I7 收敛为可执行的输入/输出表，定位 `uni_rl` 开发源码。
- **Preserve**：用户当前只要求架构接入；沿用包边界及默认训练路径。
- **Allowed scope**：本看板中的模型选择、参数表、训练/推理输入和 owner 路径；只读相关源码。
- **Forbidden scope**：自行选定新监督目标、创建教师训练任务、改 reward、安装或替换训练包、修改运行代码。
- **Depends on**：I1 已确认。
- **Done when / Check**：逐项核对所选 donor 构造参数、目标 YAML 的观测项及 uni_rl 调用签名；每项都有确定 owner，未决项不再影响 S2。执行结果：已通过静态接口核对，见 E5–E8；训练方式依用户选择留至 S4。
- **Stop**：发现模型变体或训练模式改变用户预期，先解释具体差异；不得靠维度凑齐、默认零填充或将 proposal 改名为 active contract 来绕过。

### S2 — Encoder–Decoder 纯网络

- **Change / Why**：已在 `uni_rl.algos.anybody.models` 增加 Encoder、Decoder 和组合接口；移植所选结构、拆分逻辑及 latent 归一化，去除 Isaac/任务环境耦合。实现边界符合 ANYBODY-IO-v1。
- **Preserve**：donor 的输入解析与网络数学行为；模型只依赖 Torch 和明确的包内公共模块。
- **Allowed scope**：新模型模块、必要导出、`tests/algos/test_anybody_models.py`，以及独立参考所需的 `tests/fixtures/anybody/generate_reference.py` / `muse_transformer_v1.npz`。后两者仅服务数值对照，不是运行依赖。
- **Forbidden scope**：collector、learner 损失、IPC、机器人动作缩放、奖励、数据下载、复制整套 Anybody `rsl_rl`。
- **Depends on**：I2/I4 和模型 owner 已冻结；当前已满足。S2 使用上述 ANYBODY-IO-v1 纯网络接口，不决定训练模式。
- **Done when / Check**：`uv run pytest tests/algos/test_anybody_models.py -q`；对照固定 donor 权重/输入验证数值、批维、非零可学习梯度，并检查错误布局及零 latent 的数值行为。
- **Stop**：必须访问仿真状态或改变上游网络行为才能通过；回到 S1 解释接口问题。

S2 验收结果（2026-10-15，CPU）：

- 新增 5 个文件：模型及 `__init__.py`、聚焦测试、参考生成工具、约 24 KB 的 NPZ 参考。参考生成器只执行固定 donor 的原始 Encoder 和 Decoder 构造/前向代码，不导入移植模型；固定源文件身份、输入和权重种子。常规测试只读 NPZ，不需要 donor 临时目录、网络或 Isaac。
- 精确测试命令（cwd=`ADAPT/unilab_rl`）：`PYTHONPATH=src UV_CACHE_DIR=/private/tmp/anybody-s2-uv-cache uv run --no-project --python /Users/sss9999/enbodied_intelligence/unilab/unilab_stand/.venv/bin/python -- python -m pytest tests/algos/test_anybody_models.py tests/test_smoke.py::test_no_unilab_dependency -q`；结果 **40 passed in 0.45s**，退出码 0。复用既有 Torch 2.7.0 / pytest 9.0.2，没有安装依赖；显式 PYTHONPATH 确保读取本次源码。
- 对照涵盖 raw mu、归一化 z、历史和动作，阈值 `atol=1e-6, rtol=1e-5`；另验收历史逐元素排序、Decoder 输入、mask 屏蔽/可见语义、潜变量干预、非零有限梯度、单样本/多批维/非连续输入、零 latent、错误布局/配置/dtype/非有限值拒绝。
- Ruff lint 通过；首次 format check 提示 models.py 需格式化，限定该文件执行 formatter 后，4 个 Python 文件 format check 全通过。mypy 新模型目录通过；Pyright 使用显式既有解释器检查，结果 0 errors / 0 warnings（另提示本仓没有 `.venv`，未因此创建环境）。
- 上述质量命令均由同一个 `UV_CACHE_DIR=... uv run --no-project --python <既有解释器> -- python -m` 前缀运行，参数分别为 `ruff check src/uni_rl/algos/anybody tests/algos/test_anybody_models.py tests/fixtures/anybody/generate_reference.py`、`ruff format --check` 同一路径、`mypy src/uni_rl/algos/anybody`、`pyright --pythonpath <既有解释器> src/uni_rl/algos/anybody`。
- 等级：network-test-confirmed / static-check-confirmed。没有修改既有 tracked 训练代码、包配置或锁文件；没有运行仿真、优化器训练、GPU 验证、导出或完整仓库/PR gate，不将网络测试升级为策略质量证据。

### S3 — 观测历史与机器人数据适配

- **Change / Why**：在当前 UniLab 的 `src/unilab/conf/appo/task/` 增加专用 owner YAML，复用 `ObservationManager` 历史能力；必要语义适配放在 `src/unilab/tasks/motion_tracking/common/manager_terms.py` 和 `motion_loader.py` 的最小边界。
- **Preserve**：`obs`/`critic` 分离、现有 task 默认配置、单位与控制周期、clip 内采样和部分 reset 隔离。
- **Allowed scope**：专用配置、确有缺失的 observation/data adapter；拟新增 `tests/tasks/test_anybody_observations.py`。
- **Forbidden scope**：新增第二套历史缓冲、热路径解析 XML、仅按数组下标猜测关节/刚体映射、全量 AMASS 转换器、改 backend 或 reward。
- **Depends on**：I3/I7；模型联接验证依赖 S2。
- **Done when / Check**：`uv run pytest tests/tasks/test_anybody_observations.py tests/managers/test_observation_partial_reset.py tests/envs/test_motion_loader.py -q`；用有可辨识关节/帧值的样本检查时序、部分 reset 不串行、映射和无效元数据拒绝。
- **Stop**：只有原始 AMASS 且缺少重定向数据，或字段坐标/来源无法确定；保留已验证布局，不将未知值补零。

### S4 — 接入现有异步训练器

- **Change / Why**：修改 `uni_rl` 的模型调用边界或专用 learner/runtime，解决 `_actor_mean_std` 对 `.mlp` 的直接依赖；通过 UniLab `runtime_resolver` / owner YAML 组装。采集、target actor 和更新必须走同一模型定义。
- **Preserve**：`EnvFactory` 注入、既有 AsyncRunner 生命周期、共享权重协议、V-trace 与 timeout 语义、默认 MLP/APPO/HORA。
- **Allowed scope**：`uni_rl.algos.appo` 中必要模型调用点及新模型 runtime；UniLab 专用配置/组装；拟新增 `tests/algos/test_anybody_appo.py`（uni_rl）与 `tests/scripts/test_anybody_appo.py`（UniLab）。
- **Forbidden scope**：私建异步同步协议、`uni_rl -> unilab` 反向 import、未选择训练模式便改缓存 schema，或将训练损失塞进 UniLab 入口脚本。
- **Depends on**：S2/S3；I5 冻结。关节空间 APPO 可保持动作缓存语义；若选蒸馏则另列监督更新工程包；若选 latent RL 则显式区分采样动作/log-prob 与解码关节动作，并先回到总体路线补齐其 owner。
- **Done when / Check**：在 uni_rl 根目录运行 `uv run pytest tests/algos/test_anybody_appo.py -q`；在 UniLab 根目录运行 `uv run pytest tests/scripts/test_anybody_appo.py -q`。同权重下采集/更新均值与 log-prob 一致，E/D 的实际参数变化符合冻结设置，权重同步含模型 buffer。
- **Stop**：相同输入在两条路径含义不同、错误参数被更新、或需要未列出的协议/损失 owner；不得扩写整套 runner 规避问题。

### S5 — 保存、重载与确定性推理

- **Change / Why**：在 `uni_rl` 保存/恢复 owner 持久化模型及必要训练状态；在 UniLab `visualization/interactive_playback.py`、`scripts/train_appo.py`、`training/onnx_export.py`、`utils/sim2sim.py` 的必要边界接入完整模型推理，替换仅导出 `.mlp` 的假设。
- **Preserve**：旧 checkpoint 与默认 MLP playback 路径；导出/播放不要求教师或 critic-only 输入；跨后端严格校验。
- **Allowed scope**：新模型的结构/layout/归一化元数据及上述调用点；拟新增 `tests/algos/test_anybody_checkpoint.py`（uni_rl）、`tests/training/test_anybody_inference.py`（UniLab）。
- **Forbidden scope**：静默部分加载、靠补零适配错误观测、重写已有 checkpoint、引入训练期特权输入到导出图。
- **Depends on**：S4 明确模型/优化器状态；可在 S2/S3 后先准备确定性推理检查。
- **Done when / Check**：uni_rl：`uv run pytest tests/algos/test_anybody_checkpoint.py -q`；UniLab：`uv run pytest tests/training/test_anybody_inference.py tests/training/test_sim2sim_resolver.py -q`。独立重载动作一致、恢复后仅授权参数继续变化；错误 layout 在加载前拒绝；ONNX 与完整 PyTorch 推理对照。
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
| S1 接口冻结 | ANYBODY-IO-v1 / E5–E8 | 已完成；I5 经用户决定延后 | code-confirmed + document-confirmed |
| S2 纯网络验收 | 本节 S2 验收结果 | 已完成；40 项检查通过 | network-test-confirmed |
| S3–S6 实现与验收 | 各工程包的 Done when | 未开始 | unconfirmed |

## 证据追加区

### E1 — 2026-10-15：基线源码与包边界

- 方法：在目标根目录运行 `git rev-parse HEAD`、`git remote -v`、`git status --short`；读取 `AGENTS.md`、`pyproject.toml`、`uv.lock`、训练入口、观测 manager、motion loader、playback、相关 ADR 和历史迁移笔记。
- 观测：目标 HEAD 为 `70afeb9ac9708ec1320e3d62bdc7923d73abf957`；初始工作树干净；当前没有仓内 `src/unilab/algos/torch/appo`。此前针对旧仓的路径不适用。
- 包读取：下载 `uv.lock` 指定的 `unilab_rl-1.0.0.tar.gz` 到 `/private/tmp/unilab_rl_feasibility_1.0.0.tar.gz`，执行 `shasum -a 256` 并展开只读检查；结果 `b2ab788a99054b2a75183b08c76c9cb5e70af5a6ed749316106a082ded9fbe16` 与锁文件一致。
- 观测：实际发布包保留 `.mlp` 更新调用、可配置 actor class 和 runtime resolver；collector 缓存动作并直接传给环境。Admittance 入口通过可 pickle 的 registry env factory 注入环境。
- 等级：code-confirmed。未安装依赖、未导入运行包、未执行模型或仿真测试；历史 AMP 已运行的记录不能证明当前锁定包包含该扩展。
- 检查异常：初次沿用旧路径的 `sed/rg` 返回文件不存在；随后根据当前树重新定位 `src/unilab/scripts/`、`src/unilab/conf/`、`tasks/` 和外部包，未据旧路径制定修改。

### E2 — 2026-10-15：用户确认目标

- 用户明确选择“当前 ADAPT/Admittance 仓库（推荐）”。旧 `unilab_stand` 不再作为实施目标。
- 等级：human-confirmed；不等同于选择训练模式或授权长期训练。

### E3 — 2026-10-15：规划落盘

- 新增本看板，集中记录范围、六步路线、逐步工程包、状态及追加证据；没有替换本主题旧版本或修改历史笔记。
- 等级：note-confirmed。所有新增模型/测试/配置路径均为拟定，尚无实现或测试通过记录。
- 文档检查记录见 E4；不能将此条作为运行验收证据。

### E4 — 2026-10-15：文档验证

- `git diff --no-index --check /dev/null note/anybody/task_canvas.md`：诊断输出为空；原始退出码 1 表示新增文件有差异，不能误记为命令退出码 0。后续用显式检查“诊断为空且原始退出码不大于 1”的文档检查确认，无空白错误。
- 逐项 `test -f` 检查 13 个既有 owner/引用路径：全部存在；S1–S6 工程包均包含范围、依赖、完成条件和停止条件。拟新增文件已明确标注，不充当既有路径证据。
- 初次路径检查采用 zsh 特殊变量 `path` 作循环变量，13 项存在性检查通过后，末尾 `git` 因 PATH 被改变返回 127；该子 shell 结束后已改用普通变量 `owner_file` 重跑，13 项及 git 状态读取均通过。组合命令尾部的 no-index diff 返回 1，含义如上。失败未覆盖，未修改系统或仓库环境配置。
- `git status --short`：只有本主题新增 `note/anybody/`；tracked 生产代码 diff 为空。
- 等级：document-confirmed。模型测试、仿真、异步训练、保存/恢复、导出均未执行。

### E5 — 2026-10-15：S1 执行授权及训练范围收敛

- 用户要求“开始执行第一步”，随后明确选择“先完成网络与观测接入，训练方式到 S4 再定”。
- 因此更新 S1 完成条件：I5 的训练模式延后是用户已决定的范围，不再把选定 APPO/教师作为 S2/S3 的前置条件；S4 仍须明确训练目标、采样空间和可更新参数。
- 等级：human-confirmed。没有借此授权 S2 编码、修改算法、启动教师训练或长期训练。

### E6 — 2026-10-15：模型与观测公式核对

- 只读 donor：`latent_bottleneck_muse_transformer.py` 的构造函数、`split_obs`、`encode`、`_shape_proprio_for_decoder`、`_decode`；`G1FlatMUSETransformerDistillationRunnerCfg`；`MUSETransformerDistillationTrackingEnvCfg`；observations 中 `delta_command_real`、`motion_anchor_ori_b`、`motion_anchor_pos_b`、`ref_base_lin_vel_b`。
- 对照只读目标：`conf/appo/task/g1_motion_tracking/mujoco.yaml`、`MotionCommand` 公开状态属性、motion manager terms、`envs/mdp/observations.py`、`EntityData`、ObservationManager、CircularBuffer。结果是上方逐字段公式和 S3 缺口表，而非已接通的代码。
- 已分清 absolute command 与 delta command、pelvis 与 torso 角速度、term-major 外部历史与 frame-major Decoder 历史、关节默认值 bias、raw action 与缩放动作。
- 关键注释差异：donor `ref_base_lin_vel_b` 实际是 `quat_rotate_inverse(command.anchor_quat_w, command.anchor_lin_vel_w)`，使用参考 torso 坐标系；按实际代码记录，不按其 robot-base 注释猜测。
- 官方来源：[IsaacLab v2.1.0 observations](https://github.com/isaac-sim/IsaacLab/blob/v2.1.0/source/isaaclab/isaaclab/envs/mdp/observations.py)、[ArticulationData](https://github.com/isaac-sim/IsaacLab/blob/v2.1.0/source/isaaclab/isaaclab/assets/articulation/articulation_data.py)。核对 `root_lin_vel_b/root_lin_vel_w` 后，已明确 donor pelvis COM 与目标 link 原点的差异；S1 选择目标公开 link 输入，不新增 backend COM 能力，不宣称 donor 观测/权重等价。
- donor 主文件：[模型](https://github.com/hazel-hammer/Anybody/blob/0cffa139eab0bc145c9216223e249f03c67a324c/source/rsl_rl/rsl_rl/modules/latent_bottleneck_muse_transformer.py)、[配置](https://github.com/hazel-hammer/Anybody/blob/0cffa139eab0bc145c9216223e249f03c67a324c/source/whole_body_tracking/whole_body_tracking/tasks/tracking/config/g1/agents/rsl_rl_ppo_cfg.py)。
- 等级：code-confirmed。没有运行 Isaac/UniLab，没有验证数据集元数据或跨后端物理等价。

### E7 — 2026-10-15：绑定可修改的训练源码 owner

- `git ls-remote https://github.com/unilabsim/unilab_rl.git refs/tags/v1.0.0 refs/tags/v1.0.0^{}`：tag 指向 `ce0050ecc77ec17de84b6b581148c67e1ae03caa`。
- `git clone --branch v1.0.0 --depth 1 https://github.com/unilabsim/unilab_rl.git /Users/sss9999/enbodied_intelligence/ADAPT/unilab_rl`：成功；检出 tag 的 detached HEAD，仅作为版本固定的源码工作目录。未创建分支/commit，未安装依赖或改变 Admittance 的锁文件。
- 已读取该仓 `AGENTS.md`：模型/算法 owner 在 `uni_rl.algos`，禁止反向依赖 UniLab/unisim，未来按该仓校验规则执行。
- 在新源码根目录执行 `git rev-parse HEAD`、`git status --short` 及 `git diff --no-index --stat /private/tmp/unilab_rl_feasibility_1.0.0/unilab_rl-1.0.0/src/uni_rl src/uni_rl`：commit 如上，状态为空，`src/uni_rl` 对照无差异，组合命令退出码 0。
- 等级：code-confirmed。源码存在不等于当前 Python 会导入它；S4 前仍需选择开发依赖绑定与版本交付方式。

### E8 — 2026-10-15：静态参数与布局检查

- 执行方式：`UV_CACHE_DIR=/private/tmp/anybody-s1-uv-cache uv run --no-project --no-sync --python /opt/homebrew/Caskroom/miniforge/base/bin/python3 -`，通过 stdin 输入纯标准库脚本；用 `ast.parse` 读取 donor 配置类的 policy 构造参数，以 `ast.literal_eval` 提取常量，未 import donor/UniLab/Torch。
- 检查：`sum(goal_term_sizes)=73`、`sum(proprio_term_sizes)=90`、`H*P=450`、总输入 524、Decoder 输入 466；逐项累加 ANYBODY-IO-v1 十个字段区间，末端恰为 524。实际输出 `Static config/layout check: PASS; no model or simulator imported.`，退出码 0。
- uv 提示 `--no-sync` 在 `--no-project` 下无效；未发生项目依赖同步，使用的是显式指定的既有解释器。
- 等级：static-check-confirmed。这里只验证配置常量和切片算术；不是模型数值、梯度、历史运行或训练验收。S2–S6 仍未开始。

### E9 — 2026-10-15：S1 看板一致性检查

- 用同一既有解释器经 `uv run --no-project` 执行标准库检查：Markdown 的 4 个本地链接存在；S1 状态为 1/6，S2–S6 行仍为“未开始”；E1–E8 编号各出现一次；新源码目录尚无 `uni_rl.algos.anybody` 实现；两个仓库均无 tracked 生产代码差异。
- 文档空白检查使用 `git diff --no-index --check /dev/null note/anybody/task_canvas.md`，接受新增文件的原始状态码 1，但要求诊断为空。全部检查通过，脚本退出码 0。
- 等级：document-confirmed。S1 完成限定为接口、源码及文档验收；S2 模型数值/梯度、S3 真实观测、S4 异步更新均未运行。
