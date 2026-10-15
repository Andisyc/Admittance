# AMASS 数据：仅在远程训练服务器准备

本地不保留训练 NPZ，也不启动训练。GitHub 只提交读取器、测试、准备命令和说明。

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
