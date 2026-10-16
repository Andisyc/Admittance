---
orphan: true
---

# ADR-0009 Anybody Encoder–Decoder 直接强化学习

- Status: Accepted for implementation; real-data and remote validation pending
- Date: 2026-10-17
- Owners: UniLab 数据、动作跟踪与仓内 RL 维护者
- Supersedes: None
- Superseded by: None

## Context

用户明确选择直接用强化学习更新现有 MUSE Encoder–Decoder，不采用教师蒸馏。仓内 APPO 已在采集端和学习端支持配置化 Actor。本地真实 BVH 实验验证了机器人轨迹/FK/NPZ 边界，但不证明 AMASS 人体重建或仿真跟踪效果。

## Decision

- 训练继续由仓内 `unilab.rl` 负责，复用 APPO 采集器、ring、V-trace 和权重同步；新增兼容 Actor，其 `.mlp` 为纯 `MUSEEncoderDecoder`，不分叉异步运行时。
- Encoder 和 Decoder 共同接受 RL 梯度。归一化的确定性潜码进入 Decoder，探索分布为 29 维原始关节动作上的独立 Gaussian；本阶段不是潜变量动作 RL，也不证明未访问潜码具有可复用动作先验。
- 新增独立 G1 配置：当前目标 73 维，按 term 拼接的最旧到最新 5 帧 [29,29,3,29] 历史，以及原始 mask。历史/reset 仍归 ObservationManager。初始直接跟踪方案使所有目标可见，关闭观测归一化和编译，Critic 独立使用 MLP，不消费教师标签。
- 保留目标 G1 的动作尺度、默认偏置及已有任务配置；概率描述环境处理前的原始动作，不引入 donor 的 effort/stiffness 动作尺度。
- 离线人体重建必须显式声明 packed SMPL-H 并提供匹配的 dense NPZ 人体/DMPL 权重，保留所有 beta/DMPL 系数。未知模型、维度不兼容或缺权重必须报错，禁止自动猜测、补零或静默丢弃动态形变。
- 命名世界系人体姿态可以桥接到 GMR 名为 SMPL-X 的共同刚体匹配表；这不等同于把 SMPL-H 参数当作 SMPL-X 参数。真实形体、frame offset 与数据质量仍待验证。
- 只转移根姿态和命名关节，再用本仓目标 XML 重新计算世界系刚体原点位置/速度；NPZ 包含 world0 和名称。使用明确时间网格，不以整数跳帧冒充目标频率；GMR 附加刚体不能决定目标索引。
- 网络结构与策略 I/O 纳入 sim2sim 契约。完整导出留至 S5，本 profile 显式关闭未验收导出及训练后自动播放；已有默认行为保持不变。

## Alternatives Considered

教师蒸馏未获用户选择；移入 donor 同步训练器会丢失异步架构要求；离线人体姿态自编码器会改变本次控制策略目标。

## Validation boundary

远端运行前检查人体重建数学、时间/四元数/限位、NPZ 布局、历史/reset、采集与学习端的均值/log-prob、E/D 梯度连通和真实权重传输，不执行本地 optimizer 更新或 rollout。缺少真实权重与远端接入仍是验收前提；不能从这些检查推断动力学可行性、策略质量或完整训练恢复成功。

## Evidence In Repo

`src/unilab/data/`、`src/unilab/rl/algos/anybody/`、`src/unilab/conf/appo/task/anybody_g1/mujoco.yaml`、`tests/rl/test_anybody_actor.py`、`tests/data/test_human_motion.py`、`tests/data/test_tracking_export.py`、`tests/models/test_anybody_observations.py`。

## Related Documents

- {doc}`仓内训练所有权 </adr/ADR-0008-in-tree-training-runtime>`
- {doc}`配置所有权 </adr/ADR-0003-task-owner-and-config-compose-contract>`
- 当前进展只记在仓库 `collaboration/overview.md`。
