# AMASS 数据：仅在远程训练服务器准备

本地不保留训练 NPZ，也不启动训练。GitHub 只提交读取器、测试、准备命令和说明。

2026-10-16 用户另行授权 Mac **单段转换验证**：一份真实样本保留在项目忽略目录 `.tmp/mac-gmr-probe/amass/`，不是本目录的正式训练集。该例外不授权本地批量下载或训练；运行结果与缺少匹配人体权重的限制见 [Mac 验证交接](../../collaboration/handoffs/20261016T221954+0800-assistant.md)。

在**远程服务器**的 Admittance 根目录、已配置好的项目环境中运行：

```bash
PYTHONPATH=src uv run python scripts/motion/download_amass.py --output-dir datasets/amass
```

命令只准备数据：固定公开第三方镜像
[realdream-ai/AMASS](https://huggingface.co/datasets/realdream-ai/AMASS/tree/d38490c0e60dcefae81461fc5f4765fe66727831/raw/CMU)
的 revision `d38490c0e60dcefae81461fc5f4765fe66727831`，按 subject 轮选 384 条、每条不超过 4 MiB、总量不超过 512 MiB。
同一 subject 内按源路径 SHA256 排序。每份文件检查源 LFS SHA256、大小及 AMASS 字段，已有文件通过检查则跳过；失败可重跑，损坏的既有文件会明确报错。
网络访问使用 Python 标准代理环境变量，例如 `HTTPS_PROXY`，不更改系统设置。
本轮仅查询源列表得到新规则预计选择 384 条、93 个 subjects、534,899,312 bytes（约 510.12 MiB）；这不是此前丢失清单的同一子集，实际下载/帧数由远程生成的 manifest 确认。

远程生成：

- `raw/CMU/`：原始人体动作，受 gitignore 保护，不随 Git 上传。
- `manifest.json`：实际下载的来源、哈希、大小、帧率、帧数和模型声明。
- `splits.json`：按 CMU subject 隔离的训练/验证文件，seed=20261015；划分后过滤不足 64 帧的文件。

64 帧、stride=16 仅用于数据预览，并非已确定的训练超参数。保留原始帧率和 float 精度，不默认重采样或推断人体模型。
人体参数到 Encoder/Decoder 的特征、标签及训练目标仍属于 S4，本命令不连接优化器。

来源是第三方镜像，使用时遵循 [AMASS 原许可](https://amass.is.tue.mpg.de/license.html)；本命令不获取 SMPL 模型权重。
此前曾在本地验证 384 条、489.83 MiB、384,010 帧真实数据（见协作看板 E11）；该目录及逐文件清单现已不在工作区。
此处提供重新准备的固定版本规则，**不声称恢复了丢失的完整清单**。本轮未在本地执行该下载命令，也尚未在 SSH 服务器执行。

远程读取示例（只取批次）：

```python
import json
from pathlib import Path
from torch.utils.data import DataLoader
from unilab.data import AmassWindowDataset

root = Path("datasets/amass")
split = json.loads((root / "splits.json").read_text())
dataset = AmassWindowDataset(
    [root / p for p in split["train"]],
    window_length=split["window_length"], stride=split["stride"],
)
batch = next(iter(DataLoader(dataset, batch_size=16, num_workers=0)))
print(batch["poses"].shape, batch["fps"])
```

## S4：显式人体模型与 G1 动作转换

正式入口为 `scripts/motion/amass_to_g1.py`；它只转换，不训练。当前支持显式声明的 packed **SMPL-H**（156 维姿态），不是把任意 AMASS 文件强行当 SMPL-X。人体模型需提供匹配性别、含全部 beta 维度的 dense NPZ（`v_template`、`shapedirs`、`J_regressor`、`kintree_table`）；非零 DMPL 还必须提供匹配的 `eigvec` NPZ。读取时不启用 pickle，不截断 beta、不补关节、不默默丢弃 DMPL。模型身份和配套文件需由来源记录确认，156 维本身不构成身份声明。

转换需要可选的离线预处理环境：MuJoCo 3.11、SciPy、Mink、DAQP，以及官方 [GMR 固定版本](https://github.com/YanjieZe/GMR/tree/bb1bbe40774794fceb2a7c579a3464a28e68c844)。按官方说明安装 GMR 及其 G1 资源；若源码独立放置，可将该源码根目录加入 `PYTHONPATH`。这些是离线工具依赖，未引入外部训练运行时或更改项目锁。当前适配器以关键字向 Mink 传递限位，不需要修改 GMR 安装包。

取得匹配权重后，在准备数据的服务器运行（路径和身高应使用真实资料）：

```bash
PYTHONPATH=src:/path/to/GMR uv run python scripts/motion/amass_to_g1.py \
  --input datasets/amass/raw/CMU/01/01_08_poses.npz \
  --output datasets/amass/g1/CMU/01/01_08.npz \
  --model-type smplh --source-up-axis z \
  --body-model datasets/amass/body_models/smplh/male/model.npz \
  --dmpl-model datasets/amass/body_models/dmpls/male/model.npz \
  --human-height 1.75 --fps 50
```

`1.75` 仅为命令示例，不代表已经测量该 subject 的身高。转换器先在时间网格重采样，再做离线重定向；只接收已检查运动学一致的 G1 根与命名 29 关节，用本仓模型重新计算含 world0 的刚体布局和世界系速度。NPZ 保存关节/刚体名、输入及模型哈希、原始/目标帧率和明确的坐标约定；拒绝覆盖已有产物。`g1/` 和 `body_models/` 已加入 gitignore。不会自动生成跨 subject 的机器人训练划分；批量处理应沿用原 `splits.json` 的 subject 隔离。

## Encoder–Decoder 直接 RL 配置

用户已选择直接 RL。独立 owner 配置为 `src/unilab/conf/appo/task/anybody_g1/mujoco.yaml`：Encoder 从 524 维观测生成 16 维潜码，Decoder 接收潜码和 450 维本体历史，输出 29 维动作；Critic 独立使用 MLP。输入的 5 帧历史由现有 ObservationManager 管理，mask 保持原始零、所有目标可见、关闭观测归一化。Gaussian 探索发生在原始关节动作上，现有 G1 action scale/default offset 不变。它复用仓内 APPO 采集/传输/V-trace 更新/权重同步，不要求 Teacher。

该配置的 `motion_file` 是必填值，拒绝悄悄使用旧示例。以下仅为**第三步远端短训练验收的待执行示例**，本轮未执行：

```bash
uv run train --algo appo --task anybody_g1 --sim mujoco \
  env.commands.motion.params.motion_file=/absolute/path/to/converted_g1.npz \
  algo.num_envs=8 algo.steps_per_env=16 algo.max_iterations=2 \
  training.no_play=true training.export_policy=false
```

设备应在远端根据实际 CPU/GPU 配置 `training.device` 与 `training.collector_device`，不以此示例推断吞吐或策略质量。新 profile 默认关闭训练后自动播放及未验收的 ONNX 导出；完整保存/恢复与推理仍归后续验收。直接 RL 从头训练并不自动获得覆盖任意潜码的动作先验。

当前证据：正式数据 owner 已用真实 LAFAN1 骨架通过 30→50 Hz 转换和现有 loader 读取；独立数学/接口测试覆盖人体形变和坐标规则。**真实 SMPL-H/DMPL 权重仍未取得，真实 AMASS 重建、完整仿真和参数更新尚未验证。**不能把合成权重测试或 LAFAN1 结果当作 AMASS 整链验收。
