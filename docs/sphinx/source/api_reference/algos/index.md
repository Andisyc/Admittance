# Learning Algorithms

The RL algorithm layer is maintained in this repository under `src/unilab/rl`:

- `unilab.rl.algos.rsl_rl` / `unilab.rl.algos.rsl_rl_ppo` / `unilab.rl.algos.rsl_rl_runtime` — PPO (RSL-RL) integration
- `unilab.rl.algos.appo` — APPO runner, learner, staging, worker
- `unilab.rl.algos.fast_sac` / `unilab.rl.algos.fast_td3` / `unilab.rl.algos.flash_sac` — off-policy learners and runners
- `unilab.rl.offpolicy` — generic off-policy runner, worker, thread budget
- `unilab.rl.algos.him_ppo` — HIM-PPO
- `unilab.rl.algos.hora` — HORA models, trainers, and distillation
- `unilab.rl.algos.common` — shared actor factory, networks, normalization, compile helpers

UniLab keeps the training *entrypoints* (`src/unilab/scripts/train_*.py`),
which inject environments into unilab.rl runners through
`unilab.rl.env_contract.EnvFactory`; see `src/unilab/base/env_factory.py` for the
registry-backed adapter.

All trainers conform to a single runner contract — see
{doc}`../../en/4-developer_guide/2-contracts/5-runner_lifecycle`.
