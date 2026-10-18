# AMASS 数据：仅在远程训练服务器准备

本地不保留训练 NPZ，也不启动训练。GitHub 只提交读取器、测试、准备命令和说明。

当前正式训练的[进度、温度、服务与 checkpoint 检查命令](../../note/anybody/remote_training/README.md)，以及已部署温控脚本的版本记录，见该实验说明；CPU 达到90°C暂停，降至75°C自动继续。运行状态以远端日志为准。

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

这是原始人体参数转换路线。当前正式训练另采用下文已重定向的 G1 AMASS 数据，因此不必先取得人体模型权重；两种来源不能混淆。

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

该配置的 `motion_file` 是必填值，拒绝悄悄使用旧示例。用户已授权跳过独立短训练，直接远端正式训练。使用下文生成的训练集配置时，入口为：

```bash
uv run train --algo appo --task anybody_g1 --sim mujoco \
  training.device=cuda:0 training.collector_device=cuda:0 \
  training.no_play=true training.export_policy=false \
  --config-dir /absolute/project/datasets/amass/g1/xwbc/run-config \
  --config-name formal
```

设备应根据远端硬件配置；以上为单张 CUDA GPU 示例，不推断吞吐或策略质量。正式 profile 默认 1024 环境、24 steps/env、5000 次迭代，每 500 次保存。新 profile 默认关闭训练后自动播放及未验收的 ONNX 导出；完整保存/恢复与推理仍归后续验收。直接 RL 从头训练并不自动获得覆盖任意潜码的动作先验。

原始人体路线证据：正式数据 owner 已用真实 LAFAN1 骨架通过 30→50 Hz 转换和现有 loader 读取；独立数学/接口测试覆盖人体形变和坐标规则。**真实 SMPL-H/DMPL 权重仍未取得，原始 AMASS 人体重建未验证。**预重定向 G1 AMASS 路线已进入正式异步训练，运行及中断事实见 `collaboration/overview.md`；不能把合成权重测试或 LAFAN1 结果当作原始 AMASS 重建验收。

## 正式训练数据：已经重定向的 G1 AMASS

来源为发布者的 [X-WBC 数据集](https://huggingface.co/datasets/TonyZhang1999/xwbc-retarget-data)，固定 revision `b4e169f460bd68b27e04e75c99be6e0affbfe046`，文件 `amass/unitree_g1_29dof.tar.gz`：8,099,086,834 bytes，SHA256 `2ed1c6270a6c2d060300cdb8c607a31d2b118f131d6555e6f9dede4321bd1738`。发布者标注每个机器人约 1.3 万段 AMASS；实际归档条数由准备程序统计。它不是原始 SMPL-H 参数，也不是 LAFAN1 替代集。

仅在远端准备完整归档：

```bash
mkdir -p datasets/amass/archives/xwbc
aria2c --continue=true --auto-file-renaming=false \
  --checksum=sha-256=2ed1c6270a6c2d060300cdb8c607a31d2b118f131d6555e6f9dede4321bd1738 \
  --dir=datasets/amass/archives/xwbc --out=unitree_g1_29dof.tar.gz \
  https://huggingface.co/datasets/TonyZhang1999/xwbc-retarget-data/resolve/b4e169f460bd68b27e04e75c99be6e0affbfe046/amass/unitree_g1_29dof.tar.gz
uv pip install --python .venv/bin/python --no-deps scipy==1.16.3
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 uv run --no-sync python scripts/motion/prepare_xwbc.py \
  --archive datasets/amass/archives/xwbc/unitree_g1_29dof.tar.gz \
  --source-dir datasets/amass/raw/xwbc \
  --output-dir datasets/amass/g1/xwbc --workers 8
```

完整归档保留用于后续扩展，首轮沿用此前的 CMU 范围：提取其中全部 CMU 片段，按源文件中的 CMU subject 编号隔离训练/验证。原始 reader 的 384 条/512 MiB 下载上限不限制这条预重定向数据路线。

适配器 `unilab.data.robot_motion` 按 `joint_names` 映射，不复用源 32 刚体索引。显式声明根为源 body 0、米/Z-up/wxyz；每段在至多九个源帧做目标非 world 刚体的注入式姿态匹配，要求误差不超过 0.1 mm/0.1 mrad。允许 float32 四元数范数误差不超过 1e-5 并明确归一化，其余非法数据拒绝。根/关节轨迹在明确时间网格上重采样至 50 Hz，再由目标模型 FK/Jacobian 重算 31 刚体及速度；不继承源速度，不裁剪越界关节。

输出 `manifest.json` 记录通过、拒绝及短片段排除原因、文件数、时长与实现身份；`splits.json` 记录 subject 隔离。`run-config/formal.yaml` 只注入训练动作列表，复用原有 Hydra task、网络和优化器配置，避免把数千条路径塞进命令行。读取原始人体权重的路线仍可独立使用，预重定向数据无需这些权重。上述数据、模型与缓存均留在 gitignore 目录，不上传 GitHub；AMASS 来源与各子集许可和引用要求仍适用。
