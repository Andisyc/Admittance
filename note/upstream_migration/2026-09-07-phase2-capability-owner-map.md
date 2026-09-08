# UniLab Upstream Migration Phase 2: Capability and Owner Map

Status: current analysis artifact; production construction not admitted  
Date: 2026-09-07  
Migration mode: `BRANCH_ISOLATED`  
Donor: `Andisyc/UniLab@205cbf59765242344cbce6b1b838ffbc7afb2052`  
Target: `unilabsim/UniLab@d9afc9656e63c82a64263f5a4dbd938ba1652af2`  
Target branch: `dev/upstream-20260907-integration`  
Rollback: `origin/backup/pre-upstream-20260907@205cbf59765242344cbce6b1b838ffbc7afb2052`

Companion target sources are also immutable for this analysis:

- `unilabsim/unilab_rl@ce0050ecc77ec17de84b6b581148c67e1ae03caa`
  (`v1.0.0`), inspected at
  `/private/tmp/unilab-phase2-unilab-rl-v1.0.0`.
- `unilabsim/unisim@f69e7b2025a301e6e0ce57021d6baaa232741e1e`
  (`v1.1.3`), inspected at
  `/private/tmp/unilab-phase2-unisim-v1.1.3`.
- Recursive source comparisons, excluding only `__pycache__` and `*.pyc`,
  produced no differences between those tagged checkouts and the installed
  `.venv` packages. These detached checkouts prove source identity; they are
  not construction branches.

## Decision

The migration is not a directory transplant. The current architecture has three
separate production owners:

```text
unisim public backend contract and adapters
        ^
UniLab assets, scenes, env/task semantics, registry, Hydra and playback assembly
        ^ EnvFactory / EnvProtocol injection
uni_rl learner, runner, collector, rollout storage, IPC and training checkpoint
```

The first target remains fixed-forward, walk-only AMP on MuJoCo G1. It retains
the existing asynchronous APPO architecture. Standing, running, fall recovery,
gait-phase/contact scheduling, height control and synchronous PPO are excluded.
Distillation and DAgger are mapped here so their lifecycle constraints are not
lost, but their implementation remains a separate later migration unit.

## Authority and Evidence Boundary

- Semantic donor authority: `AMP-WALK-METHOD-v003` and
  `AMP-WALK-TRAIN-v004` at the immutable donor commit.
- Distillation lifecycle evidence is preserved from `note/distill_native_corruption/`
  at the donor commit. It does not prove a native-corruption root cause.
- Target repository discipline is active in `AGENTS.md`, the developer guide,
  package-boundary tests and the installed package contracts.
- Target baseline evidence: `make check` passed; `make test` passed with
  1588 passed, 21 skipped and 1 xfailed on target `d9afc96`.
- Independent bounded migration review passed after closing source-identity,
  owner-boundary and research-tensor-contract findings. Its machine-checkable
  receipt is `manifests/phase2-migration-review.json`.
- This artifact establishes ownership and scope only. It does not claim module
  correctness, runtime connectivity, training readiness or policy quality.

## Capability Inventory and Target Ownership

| ID | Donor semantic object | Donor evidence | Target owner and public boundary | State / persistence risk | Disposition |
| --- | --- | --- | --- | --- | --- |
| C1 | Fixed-forward AMP task | `src/unilab/envs/locomotion/g1/amp_walk.py::{G1AMPWalkCfg,G1AMPWalkEnv}` | UniLab: `ManagerBasedRlEnv` owns lifecycle and manager execution; `src/unilab/tasks/locomotion/g1/` owns G1-specific terms, task config and registration | Reset, terminal observation and obs-group schema | Migrate |
| C2 | 195D AMP state and 390D transition | `src/unilab/algos/torch/amp/spec.py`; `amp_walk.py::_compute_amp_observation` | State production in UniLab; transition consumption in `uni_rl` | Shape, body order, terminal identity and normalization | Migrate with a frozen Test Card |
| C3 | ReGait expert transitions | `amp/motion_dataset.py::WalkMotionDataset.from_manifest`; `regait_v013_a022_full_horizon_313` | UniLab asset hub/provenance loader supplies package-neutral arrays to `uni_rl` | SHA, FPS, frame adjacency, no-wrap sampling and body order | Migrate; do not infer generator rationale |
| C4 | Native four-sample sensor history | old `SimBackend::{configure_sensor_history,get_sensor_history}` and MuJoCo implementation | `unisim.backend.base.SimBackend` plus MuJoCo adapter | Oldest-to-newest layout, reset/step state, one native `nsteps=4` transaction | Migrate first in `unisim` |
| C5 | Symmetric self-collision reward | `amp_walk.py::{compute_self_collision_cost,_reward_self_collisions}`; `amp_walk_task.xml` | XML sensor and G1-specific reward term in the UniLab task package; reward execution in `ManagerBasedRlEnv`; history API in `unisim` | `nsample=4`, 10 N threshold, scale `-0.1` | Migrate without right-hand special cases |
| C6 | Generic async APPO lifecycle | old APPORunner/worker/staging plus IPC | Existing `uni_rl.algos.appo`, `uni_rl.ipc`, `uni_rl.logging` | Spawn, ring ownership, error channel, close/terminate and weight versions | Reuse current owner; never copy into UniLab |
| C7 | AMP APPO extension | `src/unilab/algos/torch/amp/{runner,learner,worker,transition}.py` | New AMP extension in `uni_rl`, selected through existing `runtime_resolver` | Extra rollout fields, terminal next state and update order | Adapt current `uni_rl`; no synchronous fallback |
| C8 | Discriminator/replay/normalizer | old `amp/{learner,discriminator,replay,normalizer}.py` | `uni_rl` AMP learner and checkpoint schema | Optimizer, replay RNG/ring, normalizer and discriminator version | New versioned state schema |
| C9 | APPO checkpoint path and playback | old AMP runner/checkpoint plus generic APPO play | `uni_rl` owns training payload; UniLab owns path resolution, sim2sim guard and playback assembly | Old/new reader compatibility and actor-only playback | Defer compatibility decision to integration unit |
| C10 | Hydra activation and entrypoint | old `conf/appo/task/g1_amp_walk/mujoco.yaml` | `src/unilab/conf/appo/task/g1_amp_walk/{base,mujoco}.yaml`; generic `train_appo.py` | Complete owner config and fail-closed runtime selection | Migrate after owners exist |
| C11 | Student-state DAgger and generic distillation | old `src/unilab/algos/torch/distill/`, `scripts/train_distill.py`, `conf/distill/` | Algorithm/runtime/checkpoint in `uni_rl`; thin task/config composition in UniLab | Dataset lineage, iteration barrier, teacher/student versions and atomic writes | Separate Phase 2B; not part of AMP walk delivery |
| C12 | Persistent distillation worker lifecycle | old `distill/{async_runtime,persistent_runtime,persistent_resources,g1_persistent_worker}.py` | Future `uni_rl` runtime extension | Spawn ownership, shared weights, request/result versions, reset and cleanup | Preserve as opt-in; do not promote by default |

## Frozen AMP Parameter Inventory

These values are donor facts that must be confirmed by executable Test Cards
before implementation, not silently retuned during the port:

| Parameter | Frozen value / rule |
| --- | --- |
| Command | fixed nonzero forward `[1.0, 0.0, 0.0]` |
| AMP state | 195 values from 13 canonical bodies |
| AMP transition | concatenated current/next state, 390 values |
| Policy rollout additions | `amp_state` and `amp_next_state`, shape `(N,T,195)` |
| Terminal rows | use true `final_observation["amp"]`; never the autoreset observation |
| Style timing | discriminator `D_k` scores policy transitions before actor/critic update; discriminator then advances to `D_(k+1)` |
| Physics batching | retain one native backend step with four physics substeps |
| Sensor history | four samples, oldest to newest |
| Self-collision | symmetric whole-body/pelvis-subtree sensor, 10 N hit threshold, reward scale `-0.1` |
| Expert sampling | adjacent transitions, no end-to-start wrap |
| Numeric representation | AMP state, next state and transition are canonical `float32` at the rollout/learner boundary; no implicit mixed precision |
| Axis semantics | rollout fields are `(num_envs, time, feature)` before the learner's explicit flattening; current and next feature order is stable |
| Device transfer | collectors publish CPU/shared-memory arrays; learner staging performs the explicit copy to the learner device |
| Validity and masks | every collected step has a valid AMP pair; terminal rows substitute the true final observation before flattening; missing rows are not zero-filled or masked away |
| Finite-value behavior | reject non-finite expert or policy AMP rows at the nearest producer/ingestion boundary; do not sanitize silently |
| Gradient ownership | style rewards are scored with frozen `D_k` and detached from discriminator gradients during actor/critic update; discriminator, normalizer and replay mutation are learner-only |
| Empty/invalid batches | fail closed with an owner-level diagnostic; do not perform a no-op update that appears successful |
| Non-scope | standing, running, recovery, gait control, height control, right-hand-only reward |

The inventory above is complete for migration planning. U0 Test Cards must
bind each rule to the exact target producer, consumer, dtype/device assertion
and failure behavior before construction is admitted.

## Target Gaps Proven by Source Inspection

1. `unisim-core==1.1.3` has current sensor getters and `BackendSensorView`, but
   no public sensor-history contract. UniLab must not reconstruct history from
   private MuJoCo state.
2. `unilab-rl==1.0.0` contains the generic asynchronous APPO runtime, but its
   rollout ring, collector, learner and checkpoint flow do not expose the extra
   AMP fields/state required by C7-C8.
3. `ManagerBasedRlEnv` owns observation/reward/reset lifecycle and manager
   execution, while `src/unilab/tasks/locomotion/g1/` owns task-specific term
   implementations, task config and registration. The current observation
   route exposes policy and optional critic groups, so these owners jointly
   need a deliberate AMP group and matching terminal-observation contract.
4. The target asset hub resolves individual motion paths, but has no migrated
   AMP manifest validator or ReGait provenance contract.
5. Current HORA distillation is task-specific. It is not a substitute for the
   donor's generic G1 DAgger/distillation workflow.

## Implementation Units

### U0: Engineering admission

Owner: workflow governance and module alignment.

- Rebind the accepted AMP contract to the target identities without changing
  its walk-only semantics.
- Create confirmed Test Cards for AMP state/transition, terminal next-state,
  sensor-history order, self-collision reward, update order and checkpoint
  compatibility behavior.
- Create construction branches and a checkout-bound affected-module registry
  for all three repositories. The immutable release source identities used by
  this analysis are already bound above.
- Obtain an identity-bound mature-method alignment receipt because the port
  changes the training extension architecture, even though scientific intent
  is preserved.
- Pass an independent plan/migration review.

Stop: no production RED test or code edit until these receipts are current.

### U1: UniSim sensor-history contract

Owner: `unisim`.

- Add a fail-closed public history view/contract to `SimBackend`.
- Implement MuJoCo native-history materialization and oldest-to-newest reads.
- Preserve the single batched backend call and the default no-history fast path.
- Prove ON/OFF physics equivalence, reset/step ordering and unsupported-backend
  rejection.

### U2: UniLab AMP task and expert-data boundary

Owner: UniLab.

- Add G1-specific AMP observation/reward terms, task config, registration and
  the task-level XML fragment under `src/unilab/tasks/locomotion/g1/`.
- Extend the `ManagerBasedRlEnv` lifecycle/public observation route for the AMP
  group and true terminal-observation propagation without embedding G1 rules
  in the generic env owner.
- Add manifest-aware ReGait resolution through the asset hub cold path.
- Add the symmetric self-collision reward using only the public `unisim` API.
- Keep generic gait-phase rewards and default-pose authority absent.

### U3: uni_rl AMP asynchronous extension

Owner: `uni_rl`.

- Generalize rollout field/checkpoint extension points only as far as AMP needs.
- Add AMP collector payload, transition resolution, learner, discriminator,
  replay and normalizer.
- Preserve generic APPO behavior when the AMP runtime is not selected.
- Prove `D_k` reward/update ordering and deterministic resource cleanup.

### U4: UniLab composition, checkpoint and playback

Owner: UniLab composition root plus `uni_rl` checkpoint owner.

- Add owner YAML and runtime resolver selection without adding AMP branches to
  the generic training script.
- Integrate checkpoint path resolution, sim2sim dimension guards and actor-only
  playback.
- Define and test the old/new checkpoint matrix. Until then, old training
  checkpoints are fail-closed for resume; this does not decide whether a later
  converter will be supplied.

### U5: Offline and bounded runtime closure

Owner: module-alignment and formal-runtime-audit.

- Close owner tests, official config composition, spawned collector/learner
  connectivity, save/reload and cleanup.
- Only then admit one bounded asynchronous runtime probe.
- Training duration and policy quality remain separate decisions.

### U6: Distillation/DAgger migration (separate later unit)

Owner: future `uni_rl` algorithm/runtime and UniLab composition.

- Preserve iteration barriers, artifact lineage, atomic writes, process reset,
  error propagation and cleanup from the donor design.
- Keep persistent mode opt-in. Do not claim the old native-corruption root cause
  was fixed merely because the architecture changed.
- Do not begin this unit until AMP rollout and checkpoint schemas are stable.

## Construction Admission

`code-construction`: **NOT ADMITTED** for production migration in this phase.

The first unclosed receipt is U0: target-bound semantic/Test Card and
mature-method alignment evidence do not yet exist. Immutable source identity
is established for all three repositories, but construction branches for
`unisim` and `uni_rl` are not. Creating those branches, implementing U1,
committing or pushing are outside this phase's current Git authority.

The next safe action is to execute U0 and prepare a reviewed U1 engineering
unit. No simulator, training, remote execution, commit, push or `main` update is
admitted by this document.

## Unresolved but Non-Blocking for U0/U1

- Whether legacy AMP checkpoints need conversion for new training resume, or
  only actor-only playback compatibility.
- Where the private ReGait blobs will be hosted in the new asset workflow.
- ReGait generation/selection rationale; the donor only proves artifact
  identity and consumption.
- The exact first invalid native operation in the historical distillation
  corruption incidents.
