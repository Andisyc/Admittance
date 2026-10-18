# 当前远程实验：温控与查看命令

本目录保留 2026-10-18 正式训练使用的监督脚本，供代码审查和实验复现。它是本次 Linux/systemd/cgroup v2 运行记录，不是新的 UniLab 训练入口或通用部署工具；不会因拉取仓库自动安装服务或改变正在运行的训练。

`thermal_supervisor.py` 来自已部署的 `.tmp/anybody-safe-supervisor.py`，只整理导入与格式，非导入 AST 等价。部署原文 SHA256 为 `f9f2176dc0285c5e9b2e4f5c118a642272ab3bd0a5e9e2b854225be5d6378e44`。本次单次切换脚本含已完成操作的固定 run 路径，不作为可重复执行的入口发布；完整命令与配置保留在远端对应 run。

当前启动记录见[温控续训交接](../../../collaboration/handoffs/20261018T182125+0800-assistant.md)，最新协作状态见[总览](../../../collaboration/overview.md)。数据、checkpoint、缓存、虚拟环境和临时日志都不上传 GitHub。以下是远端目录，不是本机路径。

## 在 Electerm 中查看

登录训练服务器后先执行；`run` 指向当前续训目录，`unit` 是其后台服务：

```bash
cd /ssd1/cyx/liujun/Admittance
run=$(cat .tmp/latest-formal-run.txt)
unit=$(cat "$run/unit.txt")
```

实时进度与降温事件：

```bash
tail -f "$run/progress.log"
```

`new_updates=.../2300 from_checkpoint=2700` 表示从累计 2700 次保存点继续的本次更新数。`cooling` 是正常降温暂停；`training` 是继续训练。`THERMAL_PAUSED`/`THERMAL_RESUMED` 记录状态切换。按 **Ctrl+C** 只退出日志查看，不停止训练。

每 5 秒刷新一次温度、进度、暂停/恢复次数及错误原因：

```bash
watch -n 5 cat "$run/status.json"
```

查看 `resources.cpu_max_temperature_c`、`thermal.paused`、`thermal.pause_count`、`thermal.resume_count` 和 `stop_reason`。当前配置为 CPU **≥90°C 暂停、≤75°C 自动继续**，暂停保留同一进程内模型、优化器与采集状态。显存不会因暂停自动释放；温度下降来自暂停训练计算，脚本不控制风扇或硬件电源。

确认服务存活和查看保存模型：

```bash
systemctl --user status "$unit" --no-pager
ls -lh "$run"/model_*.pt
```

首次保存前 `ls` 报文件不存在是正常情况。每 100 次本次迭代保存；当前 run 的 `model_100.pt` 对应累计 2800 次。检查详细训练报错可用 `tail -n 80 "$run/console.log"`。服务 active 表示监督服务仍活着；实际在训练还是降温应同时看 `status.json`，`guard_stopped`/`failed` 需要检查 `stop_reason` 和日志。

## 监督脚本的范围

脚本读取现有 run 的 `launch_plan.json`，先验证内核内存限制和继承的 CPU affinity，再完整加载检查点检查有限性；通过既有 `uv run --no-sync train --algo appo --task anybody_g1 --sim mujoco` 与独立 Hydra 配置恢复仓内 APPO。它本身不创建资源限制，必须由已配置的 systemd 服务启动；脱离原 cgroup 直接执行会在预检拒绝运行。不要与现有服务并行启动第二份训练。

当前资源配置是 6 个性能核 CPU 2/4/6/8/10/12、MemoryHigh=24 GiB、MemoryMax=28 GiB、MemorySwapMax=512 MiB。用户 manager 仅下放 memory/pids，不能把 systemd 接受 CPUQuota 属性当作配额生效证据。

CPU 温度采用本机 coretemp 的最高读数；监控进程留在训练进程组外，整个采集/学习组通过 SIGSTOP/SIGCONT 暂停/继续。降温期间停滞计时不触发误退出，恢复后重新计算活动时间。其他内存/GPU/磁盘/非有限值/传感器故障仍停止，不盲目重启。5 秒采样的软件温控不保证消除先前未确诊的驱动/硬件卡死。

已完成的证据：无计算负载的真实进程组验证 89°C 继续、90°C 暂停、82°C 保持暂停、75°C 恢复；中间 82/75 配置在正式训练发生过真实自动循环。最终 90/75 配置已核实运行中阈值并继续产生有限更新；截至交接验收，尚未自然触发 90°C，不能将人工温度输入测试当作实际升温试验。
