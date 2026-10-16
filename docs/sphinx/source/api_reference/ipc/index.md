# Shared-Memory Runtime

The in-tree `unilab.rl.ipc` package hosts the async
runner, shared-memory buffers, replay pipelines, inference slot, DP launcher /
sync, and weight sync.

| Submodule | Role |
|---|---|
| `unilab.rl.ipc.async_runner` | The high-level orchestration loop |
| `unilab.rl.ipc.shared_buffer` | NumPy-backed shared-memory ring/buffer |
| `unilab.rl.ipc.rollout_ring_buffer` | Rollout window used by on-policy collectors |
| `unilab.rl.ipc.replay_buffer` | Bounded shared ingress for off-policy transitions |
| `unilab.rl.ipc.replay_pipelines.*` | Authoritative CUDA/MPS replay ring, device gather, and native H2D |
| `unilab.rl.ipc.inference_slot` | Fixed shared observation/action slot for learner-owned off-policy inference |
| `unilab.rl.ipc.weight_sync` | Push learner weights to on-policy collector workers |
