# Phase 3: AMP Walk on the Split UniLab Async Architecture

Status: `REVIEW_READY / USER_CONFIRMATION_PENDING`  
Plan ID: `UNILAB-AMP-PHASE3-PLAN-v001`, revision 2  
Date: 2026-09-07  
Planning authority: read source and write planning artifacts only  
Production, branch, commit, push, simulation and training authority: not granted

## 1. Reviewable outcome

Deliver one fixed-forward, walk-only G1 AMP training path on the current split
architecture:

```text
mujoco-uni-runtime -> native mjSTATE_HISTORY export in one batched step
unisim  -> public four-substep sensor history
UniLab  -> G1 task semantics, 195D AMP observation and expert-data provenance
uni_rl  -> async APPO collector/IPC/learner/checkpoint AMP extension
UniLab  -> Hydra composition, sim2sim guard and actor-only playback assembly
```

The implementation must preserve the single native
`BatchEnvPool.step(..., nstep=4)` transaction and the existing asynchronous APPO
collector/learner lifecycle. It must be trainable from a fresh run, save and
resume its own versioned checkpoint, and play the new actor through the normal
MuJoCo playback path. This phase does not claim that the resulting gait is
high-quality or that a long training run has succeeded.

## 2. Bound authority and identities

| Role | Immutable identity |
|---|---|
| Semantic donor | `Andisyc/UniLab@205cbf59765242344cbce6b1b838ffbc7afb2052` |
| Target UniLab | `unilabsim/UniLab@d9afc9656e63c82a64263f5a4dbd938ba1652af2`, current branch `dev/upstream-20260907-integration` |
| Target unisim | `unilabsim/unisim@f69e7b2025a301e6e0ce57021d6baaa232741e1e`, tag `v1.1.3` |
| Target uni_rl | `unilabsim/unilab_rl@ce0050ecc77ec17de84b6b581148c67e1ae03caa`, tag `v1.0.0` |
| Target native runtime baseline | PyPI sdist `mujoco-uni-runtime==0.5.0`, SHA-256 `a0c774da3607b6111d7c91155867377f9eb6d91ab6e0bbb97f15247df1ead9d7`, whose installed source is pinned by target `uv.lock` |
| Method authority | donor `note/amp/contracts/active/method/AMP-WALK-METHOD-v003.md` |
| Training authority | donor `note/amp/contracts/active/training/AMP-WALK-TRAIN-v004.md` |
| Owner map | `../2026-09-07-phase2-capability-owner-map.md`, SHA-256 `171b98332ed7994bac5d3ec7bf8c2e2edf957115b3a4aebad9cf23a2e4f70500` |
| Repository rules | root `AGENTS.md`, SHA-256 `94e37c5d77cac8d72771199a851c63ab6e13f6336dcb0463ff768f5585cdc273` |

The donor contracts are semantic evidence, not permission to restore the donor
package layout. The target owner boundaries are authoritative:
`mujoco-uni-runtime` owns batched native MuJoCo transport, `unisim` owns the
backend capability contract, UniLab owns environments/adapters/configuration,
and `uni_rl` owns algorithm runners, learners, collectors, IPC and training
logs.

Before construction, the maintainer must provide a checkout of the repository
that publishes the bound runtime sdist and record its native return-schema
change, the `unisim` public sensor-history contract and the new `uni_rl`
IPC/checkpoint schema as scope decisions in the appropriate issue/ADR. Proposed
construction branches, after separate Git authorization, are
`dev/amp-walk-async-phase3` in each of the four affected repositories.

## 3. Frozen behavior

### Requested

- Fixed command `[1.0, 0.0, 0.0]`; no zero-command standing case.
- AMP observation is 195 float32 values from the 13 canonical bodies in the
  frozen order; a transition is current plus true-next state, 390 values.
- Task reward contains forward locomotion and viability, with no default-pose
  authority. AMP supplies style reward.
- Symmetric pelvis-subtree self-collision uses four native substep samples,
  oldest to newest, threshold `> 10 N`, scale `-0.1`.
- AMP style reward is
  `amp_coef * clamp(1 - 0.25 * (D_k(x) - 1)^2, min=0)`.
- The learner computes all rollout style rewards once with frozen `D_k`, feeds
  them into V-trace/APPO, updates actor/critic, and only then updates the
  discriminator to `D_(k+1)`.
- Discriminator, AMP normalizer and replay are learner-only mutable state.
- `amp_state` and `amp_next_state` cross IPC as `(N,T,195)` float32 shared-memory
  fields and move to the learner device only during staging/learning.
- On a true terminal, next AMP state comes from
  `final_observation["amp"]`; autoreset observation remains the next collector
  carry. Missing, malformed or non-finite AMP data fails before publication.
- New AMP checkpoints are CPU-detached, versioned and atomically replaced.
  Full resume validates the entire identity before mutating the learner.
- Actor-only playback restores no discriminator, replay, normalizer or optimizer.

### Preserved

- Generic APPO, HORA, G1 walk, task registry, CLI route and playback behavior.
- `EnvFactory` remains pickleable; `uni_rl` imports neither UniLab nor unisim.
- `NpEnv.reset -> (obs_dict, info_dict)` and dict-valued `NpEnvState.obs`.
- Final-sensor APIs continue to mean the final native substep.
- No-history backends incur no history allocation or step-path work.
- Reset and interval perturbation reinitialize only selected environment rows.
- Existing runner/collector parent-child cleanup and error-pipe behavior.
- Asset/XML parsing remains cold path only.

### Forbidden

- Standing, running, fall recovery, gait-phase/contact scheduling, gait control,
  height control, right-hand-specific shaping, motion reset or distillation.
- Synchronous PPO fallback or a Python loop of four `nstep=1` calls.
- Backend-private probing from UniLab or uni_rl.
- Task-name branches in generic training, environment-factory or CLI code.
- A reverse `uni_rl -> unilab` or `uni_rl -> unisim` import.
- Silent zero-fill, invalid-batch no-op, partial checkpoint restore or mixed
  donor/target checkpoint identity.
- Resume or playback of historical `model_1850.pt` and `model_2000.pt`; they are
  evidence only. A future converter is a separate scope decision.

## 4. Current runtime flow

1. `unilab.cli.build_route/build_command` selects the existing APPO entrypoint.
2. `unilab.scripts.train_appo.main` composes Hydra, ensures registries, resolves
   the APPO runtime, builds a pickleable registry environment factory and
   constructs the runner.
3. `uni_rl.algos.appo.runtime.resolve_appo_runtime` selects `APPORunner` or a
   registered custom runner while retaining the default playback function.
4. `APPORunner.learn` owns shared memory, spawn collectors, learner updates,
   weight publication, checkpoint writes and final cleanup.
5. `appo_collector_fn` owns one environment and the current observation;
   `RolloutRingBuffer` currently has fixed float32 fields.
6. `ManagerBasedRlEnv` computes rewards and observation groups after one
   `NpEnv.step -> backend.step(ctrl, sim_substeps)` call. `NpEnv._reset_done_envs`
   already copies every public observation key into `final_observation` before
   replacing terminated rows.
7. `MuJoCoBackend.step` calls `BatchEnvPool.step` exactly once and refreshes its
   final physics and sensor arrays; `set_state` and interval root-velocity events
   mutate selected rows through `pool.reset`.
8. In runtime 0.5.0, native `_unsafe_step` advances all `nstep` rounds but
   `BatchEnvPool.step` exports only final FULLPHYSICS state and optional final
   `sensordata`; it does not export `mjSTATE_HISTORY`. The installed MuJoCo
   3.11.0 model/data expose `sensor_history`, `history` and `mjSTATE_HISTORY`, so
   the missing boundary is the runtime transport rather than the simulator.

## 5. Accepted Test Cards

These cards define the target behavior for this plan. Confirmation of the exact
plan hash confirms all six cards; construction must stop if a RED test reveals
that a card is ambiguous or incompatible with an owner contract.

### TC-AMP-01 — Native sensor history

| Field | Contract |
|---|---|
| Input | Sensor names and `history_length=4` bound before stepping |
| Output | Stable read view `(num_envs,4,total_width)`, numeric, finite, oldest-to-newest |
| Reset | All four samples for reset rows equal the post-reset sensor value; other rows unchanged |
| Step | One native `BatchEnvPool.step(..., nstep=4)`; history reconstructed from the returned native FULLPHYSICS history, never four Python steps |
| Failure | Unknown sensor, invalid depth, unavailable native history, bad shape/dtype or unsupported adapter raises with backend/name context |
| Non-effect | No binding means no history allocation/work and unchanged final sensor result |

### TC-AMP-02 — AMP state and expert support

| Field | Contract |
|---|---|
| State | `(N,195)` float32 finite tensor from the frozen 13-body order and torso-relative representation |
| Expert | Manifest pins files, hashes, FPS, schema and body order; returns package-neutral adjacent transitions |
| Adjacency | No clip-boundary wrap; each pair is frame `t -> t+1` from one clip |
| Lifecycle | Manifest and arrays load once before runner/spawn; reset/step never parse assets |
| Failure | Missing/hash/schema/FPS/body-order/non-finite mismatch fails before runtime allocation |

### TC-AMP-03 — True terminal transition and IPC

| Field | Contract |
|---|---|
| Normal row | `amp_next_state = next state.obs["amp"]` |
| Terminal row | `amp_next_state = final_observation["amp"]`; collector carry still uses autoreset `state.obs["amp"]` |
| IPC | `amp_state`, `amp_next_state`: float32 `(N,T,195)` shared-memory fields |
| Publication | Validate group existence, dtype, shape and finiteness before `signal_write_done()` |
| Failure | Invalid row publishes no slot/version and child closes env, attachments and error channel in `finally` |

### TC-AMP-04 — Symmetric self-collision

| Field | Contract |
|---|---|
| Sensor | AMP task XML fragment only; symmetric pelvis-subtree contact-force coverage |
| Cost | Count four native samples whose force magnitude is strictly greater than 10 N; reward scale `-0.1` |
| Symmetry | No left/right or right-hand-specific term, target or distance heuristic |
| Failure | Missing history capability/sensor or wrong sample depth fails closed during task construction |

### TC-AMP-05 — Learner order and ownership

| Field | Contract |
|---|---|
| Order | validate batch -> freeze normalizer and `D_k` -> detached style reward -> APPO/V-trace actor/critic update -> replay/normalizer/discriminator update to `D_(k+1)` |
| Ownership | Discriminator parameters are absent from the base actor/critic optimizer; collector never owns AMP mutable training state |
| Invalid batch | No optimizer, replay, normalizer, RNG, counter or checkpoint mutation |
| Health | Log policy/expert logits, style reward, discriminator losses and saturation metrics without changing the update schedule |

### TC-AMP-06 — Checkpoint, lifecycle and playback

| Field | Contract |
|---|---|
| Save | Versioned schema; actor/critic/base optimizer/target actor/APPO counter plus discriminator/optimizer, AMP normalizer, replay ring/RNG, expert provenance and dimensions; CPU detached; temp file then atomic replace |
| Resume | Complete preflight before any `load_state_dict`; new-schema identity only; exact method fingerprint includes fixed command and task-reward schema, 13-body order/195D layout, `amp_coef`, discriminator topology/loss/optimizer/cadence, replay capacity/sampling, normalizer epsilon/clip/update rule, expert hashes/FPS/adjacency and APPO actor/critic/update identities |
| Playback | New-schema actor extraction only; compare only actor topology/parameters, policy observation mapping, action dimension and sim2sim/runtime identity before actor mutation; discriminator/replay/normalizer/expert identities are intentionally irrelevant to actor-only play and no AMP training objects are constructed |
| Legacy | `model_1850.pt`, `model_2000.pt` and generic/donor AMP checkpoints are rejected for both resume and this phase's playback |
| Cleanup | Normal, constructor-failure, collector-failure and interrupt paths close child envs, shared memory, queues/pipes, trackers and parent runner; terminate is fallback after bounded join |

## 6. Function-level change map

### B0 — mujoco-uni-runtime native history transport

**P3-C00 — optional native history output**

- Baseline: PyPI sdist `mujoco-uni-runtime==0.5.0`, SHA-256
  `a0c774da3607b6111d7c91155867377f9eb6d91ab6e0bbb97f15247df1ead9d7`.
- Owner/files: `mujoco_uni/runtime/batch.py` and
  `mujoco_uni/native/batch_env.cc`; `mujoco_uni/batch_env.py` remains only the
  compatibility re-export.
- Symbols: public `BatchEnvPool.step`, native `_unsafe_step`,
  `_unsafe_step_threaded`, `_unsafe_step_callback_round`,
  `_unsafe_step_callback_round_threaded`, `BatchEnvPool::step`,
  `BatchEnvPool::StepWithControlCallback`, and the pybind `step` declaration.
- Change: add keyword `return_history: bool = False`. Default false preserves
  the current state-only or `(state, final_sensordata)` returns. True requires
  `return_sensor=True` and `nstep >= max(model.sensor_history[:,0])`, allocates
  `(nbatch, mj_stateSize(model, mjSTATE_HISTORY))`, calls
  `mj_getState(..., mjSTATE_HISTORY)` after the last native round for every
  environment, and returns `(state, final_sensordata, raw_history)`. Direct
  stepping fills every requested history slot before export. Callback stepping
  additionally owns a native `(nbatch, nhistory)` carry: initialize each row,
  restore both FULLPHYSICS and HISTORY before that environment's next round,
  and store both immediately after `mj_step`. Capture HISTORY before any
  optional `post_step_forward_sensor` refresh; that refresh may replace only
  the separately returned final/current `sensordata`, never the stepped history
  snapshot. This prevents worker reuse from mixing history across environments
  or callback rounds. Do not introduce a Python substep callback.
- Tests: default return/API snapshot; raw history against direct
  `mj_getState(..., mjSTATE_HISTORY)` oracle; contact sensor `nsample=4`
  oldest-to-newest decode; `nbatch > nthread` callback carry/no-cross-talk;
  direct/callback equality; invalid flag or insufficient-`nstep` combination;
  thread/chunk variants; and a dispatch spy proving one native call.
- Release: public return-shape change requires its owner issue/ADR, changelog,
  cross-platform wheels and a version compatible with `>=0.6.0,<0.7`; B1 must
  pin the exact released artifact hash.

### B1 — unisim public history contract

**P3-C01 — backend-neutral view**

- Owner/file: `unisim/src/unisim/backend/base.py`
- Symbols: `SensorHistoryReadFn`, `BackendSensorHistoryView`,
  `SimBackend.bind_sensor_history`, `SimBackend._bind_sensor_history_reader`.
- Change: add a separate immutable view with construction-time metadata checks
  and read-time `(num_envs, history_length, total_width)`, numeric and finite
  checks. The default adapter hook fails closed. Keep the existing final-sensor
  API untouched. Follow the current `BackendSensorView` import convention; do
  not add only the new sibling as a top-level export.
- Tests: contract metadata, shape, dtype, finite values, unknown names, invalid
  depth and unsupported backend.

**P3-C02 — MuJoCo native implementation**

- Owner/file: `unisim/src/unisim/backend/mujoco/backend.py`
- Symbols: `_bind_sensor_history_reader`, `_refresh_sensor_history`,
  `_reset_sensor_history_rows`; lazy `_sensor_history_data` and view cache.
- Change: call the B0 runtime once with `return_sensor=True,
  return_history=True`, decode the returned raw `mjSTATE_HISTORY` using public
  `sensor_history` metadata and the donor's proven state-offset/order logic,
  then refresh stable named views. Cover both direct and pre-step-control native
  calls. Reinitialize selected rows after both `set_state` and interval
  root-velocity reset. A bound view rejects any step whose `nsteps` is smaller
  than its requested depth, preventing pre-reset history leakage.
- Dependency files: `unisim/pyproject.toml` and `unisim/uv.lock` update the
  exact 0.5.0 pin to the released B0 compatibility range/hash; no editable or
  untracked runtime source is accepted.
- Tests: oldest-to-newest values, single native dispatch, pre-step-control,
  partial reset, interval reset, ON/OFF physics identity and disabled fast path.

### B2 — UniLab AMP task and expert boundary

**P3-C03 — generic extra observation outputs**

- Owner/file: `src/unilab/envs/manager_based_rl_env.py`
- Symbols: `ManagerBasedRlEnvCfg.additional_observation_groups`,
  `ManagerBasedRlEnv._build_observation_output_mapping`, and reuse from
  `_validate_observation_mapping`/`_map_observations`.
- Change: map explicitly named manager groups to extra public observation keys;
  cache and validate names/dimensions once. With no extra mapping, outputs remain
  exactly `obs` and optional `critic`. This automatically extends the existing
  generic terminal-observation copy to `amp`.
- Tests: default snapshot, valid AMP group, duplicate/missing group, reset-row
  replacement and terminal identity.

**P3-C04 — G1 AMP observation and collision term**

- Owner/files:
  `src/unilab/tasks/locomotion/g1/amp_terms.py`,
  `src/unilab/base/entity.py`,
  `src/unilab/assets/robots/g1/amp_walk_task.xml`.
- Symbols: `G1AmpObservation`, `compute_g1_amp_observation`,
  `SelfCollisionPenalty`, `EntityScene.bind_sensor_history`.
- Change: produce the frozen 195D state from declared body selectors; expose
  only a thin entity facade over B1; declare the symmetric sensor in a task
  fragment; calculate TC-AMP-04 without backend-private state.
- Tests: independent pose/quaternion oracle, exact body order, 195D/dtype/finite,
  history facade error, collision threshold/symmetry/history order and XML scope.

**P3-C05 — AMP expert manifest**

- Owner/files: `src/unilab/tasks/locomotion/g1/amp_motion.py` and
  `src/unilab/assets/motions/g1/amp_walk/manifest.json`; the two `.npz` blobs
  live at the same relative paths in `unilabsim/unilab-motions` rather than in
  Git.
- Symbols: `ResolvedG1AmpMotionManifest`,
  `resolve_g1_amp_motion_manifest`, `load_g1_amp_expert_transitions`.
- Change: call `assets.hub.resolve_motion_files`, then validate frozen file SHA,
  schema, FPS, body order, required arrays, finite data and adjacency/no-wrap;
  return immutable/package-neutral float32 arrays. Do not broaden the generic
  motion-tracking loader.
- Tests: known manifest, every provenance mismatch, adjacent pair oracle,
  clip-boundary exclusion and cold-path-only loading.

**P3-C06 — task registration and Hydra owner config**

- Owner/files:
  `src/unilab/tasks/locomotion/g1/__init__.py`,
  `src/unilab/conf/appo/task/g1_amp_walk/base.yaml`,
  `src/unilab/conf/appo/task/g1_amp_walk/mujoco.yaml`.
- Symbols: `G1AMPWalk` registry entry and its `ManagerBasedRlEnvCfg` composition.
- Change: add a new MuJoCo-only task using the existing factory. Explicitly list
  the 13 bodies, public `amp` group, fixed command, forward/viability/task
  collision rewards, task XML and AMP runtime marker. Do not inherit gait phase
  or pose terms from `g1_walk_flat`.
- Tests: Hydra exact-value assertions, registry spawn construction and a
  forbidden-term snapshot. Any generic factory/CLI task branch is a deviation.

### B3 — uni_rl AMP async extension

**P3-C07 — extensible rollout fields and terminal pairing**

- Owner/files:
  `uni_rl/src/uni_rl/algos/appo/rollout.py`,
  `uni_rl/src/uni_rl/ipc/rollout_ring_buffer.py`,
  `uni_rl/src/uni_rl/algos/appo/staging.py`.
- Symbols: `PairedObservationFieldSpec`,
  `resolve_paired_observation_fields`,
  `RolloutRingBuffer.__init__(extra_step_field_dims=...)`,
  `estimate_appo_bytes(..., extra_step_dims=...)`.
- Change: define `amp -> amp_state/amp_next_state/195`; allow validated extra
  float32 fields without changing default core keys/layout; retain dynamic
  staging transpose and account for `2*195` memory.
- Tests: owner/attach shapes, field collision/invalid dims/attach mismatch,
  default key snapshot, staging transpose and slot reuse.

**P3-C08 — generic APPO hooks and collector payload**

- Owner/files:
  `uni_rl/src/uni_rl/algos/appo/runner.py`,
  `uni_rl/src/uni_rl/algos/appo/worker.py`, and the terminal-observation
  contract owner used by the worker.
- Symbols: `APPORunner._on_env_probe`, `_rollout_observation_specs`,
  `_learner_class`, `_learner_extra_kwargs`, `_restore_learner_checkpoint`,
  `_save_checkpoint`; paired-field handling in `appo_collector_fn`.
- Change: add default no-op hooks rather than AMP branches. Collector validates
  current and true-next AMP values before publication and closes all child
  resources in `finally` on failure.
- Tests: generic defaults unchanged; normal/terminal/autoreset pairing;
  missing/wrong/non-finite data publishes nothing; constructor/step failure
  cleanup; HORA contains neither AMP fields nor imports.

**P3-C09 — AMP runner, data and learner**

- Owner/files: new
  `uni_rl/src/uni_rl/algos/amp/{runner,data,learner}.py`.
- Symbols: `AMPAPPORunner`, `AMPExpertTransitions`, `AMPDiscriminator`,
  `AMPRunningNormalizer`, `AMPReplayBuffer`,
  `AMPAPPOLearner.process_batch/update`.
- Change: the runner receives expert arrays through package-neutral constructor
  kwargs prepared by UniLab before spawn, copies them into parent-owned CPU
  arrays, and closes its probe env. The learner implements TC-AMP-05;
  discriminator params never join base combined params. `uni_rl` does not own
  or export an AMP runtime resolver; runtime selection remains at C11.
- Tests: resolver, 195D/expert provenance, no UniLab imports, probe-env close,
  exact update-order spy, no discriminator actor gradient, health metrics and
  zero mutation on invalid/empty batch.

**P3-C10 — AMP checkpoint schema**

- Owner/file: new `uni_rl/src/uni_rl/algos/amp/checkpoint.py` plus AMP runner
  save/restore overrides.
- Symbols: `validate_amp_checkpoint`, `load_amp_resume_state`,
  `extract_amp_actor_state`, `save_amp_checkpoint_atomic`.
- Change: implement TC-AMP-06, with full validation before mutation and
  atomic replacement. A deterministic `method_fingerprint` covers every
  full-resume semantic field listed in TC-AMP-06, so equal dimensions cannot
  hide a reward/style/optimizer change. Actor-only extraction applies the
  narrower playback identity column. Generic APPORunner save remains
  byte-for-byte equivalent.
- Tests: cold state, exact round trip, replay/RNG restoration, preflight before
  mutation, interruption before replace, actor extraction, one mismatch test
  for every fingerprint category, and all legacy/mixed identity rejections.

### B4 — composition, playback and contract snapshot

**P3-C11 — runtime/expert composition**

- Owner/files:
  `src/unilab/scripts/amp_appo.py`,
  `src/unilab/scripts/train_appo.py`, and
  `uni_rl/src/uni_rl/algos/appo/runtime.py`.
- Symbols: `APPORuntime.runner_kwargs`, `resolve_amp_appo_runtime`, and
  collision-checked merge in `build_appo_runner_kwargs`.
- Change: resolve and validate expert assets before constructing the runner or
  spawning collectors. The UniLab-owned custom resolver returns the AMP runner
  plus package-neutral arrays/provenance. It is the only symbol named
  `resolve_amp_appo_runtime` and is the dotted Hydra resolver. Generic
  `resolve_appo_runtime` normalizes the custom result into `APPORuntime`, whose
  new `runner_kwargs: Mapping[str,Any]` field defaults to an empty mapping and
  whose `play_fn` is still replaced by `default_play_fn` when absent. The
  generic script rejects collisions and merges that mapping into its ordinary
  kwargs. Keep one generic train entrypoint and existing `try/finally` cleanup.
  No task-name conditional is allowed.
- Tests: default APPO/HORA kwargs unchanged, AMP resolver/kwargs, failure before
  runner/env allocation and tracker/runner cleanup.

**P3-C12 — checkpoint resolution and playback**

- Owner/files:
  `src/unilab/training/run.py`,
  `src/unilab/visualization/interactive_playback.py`,
  `src/unilab/scripts/play_interactive.py`.
- Symbols: `resolve_appo_checkpoint_from_config`,
  `create_appo_playback_session`, `_build_appo_actor`.
- Change: centralize pure checkpoint path resolution, run sim2sim preflight,
  call the uni_rl actor extractor, then construct/load only the actor. Existing
  APPO CLI composition selects `g1_amp_walk/mujoco` without an AMP-specific
  route.
- Tests: new AMP actor playback, missing/malformed/legacy rejection before env
  construction, generic/HORA/zero-action behavior unchanged and final env close.

**P3-C13 — sim2sim/runtime identity**

- Owner/file: `src/unilab/utils/sim2sim.py`.
- Symbols: `DENYLIST`, `SNAPSHOT_FIELDS`.
- Change: snapshot and strictly compare public observation-output mapping, AMP
  runtime identity and actor dimensions. Keep the current warning behavior for
  absent fields in old generic snapshots, while AMP checkpoints remain fail
  closed through their schema owner.
- Tests: identical AMP config pass, observation/runtime/dimension mismatch
  rejection and generic snapshot compatibility.

**P3-C14 — pinned package integration**

- Owner/files: root `pyproject.toml` and `uv.lock`.
- Symbols: exact dependency constraints and resolved artifact hashes for
  `mujoco-uni-runtime`, `unisim-core` and `unilab-rl`.
- Change: after B0-B3 publish their reviewed versions, pin the compatible
  runtime, backend and algorithm artifacts together and regenerate the lockfile.
  Development path/editable substitutions are forbidden in final evidence.
- Tests: `uv lock --check`, frozen/no-sync import identity, package source/commit
  receipt and the full UniLab gate.

## 7. Lifecycle and cross-owner impact

| Path | Before | After | Invariant |
|---|---|---|---|
| init/materialize | Hydra resolves generic APPO; backend exposes final sensors | Runtime marker selects AMP; expert manifest validates before allocation; task binds public history lazily | No asset parsing in reset/step; no reverse dependency |
| reset/resample | Manager reset recomputes normal output groups | Fixed command and row-scoped `amp` are recomputed; history rows are filled from post-reset value | Non-reset rows and generic tasks unchanged |
| step/refresh | One native backend call, then reward/events/command/observations | Runtime carries HISTORY per env across callback rounds and exports it once; unisim decodes four samples; reward consumes them; AMP state computes in read phase | Exactly one native `nstep=4` transaction and no cross-env history |
| terminal/autoreset | Generic code copies every public group before reset | `final_observation["amp"]` becomes terminal next state; autoreset AMP remains collector carry | Terminal and carry identities never alias semantically |
| async publish/update | Fixed rollout fields feed APPO | Validated AMP pairs stage to learner; frozen `D_k` reward precedes APPO; D update follows | No partial publication or cross-process AMP mutable state |
| save | Generic runner saves APPO state | AMP runner writes complete versioned CPU payload atomically | Generic save unchanged; interrupted write leaves prior checkpoint |
| reload/play | Generic checkpoint resolution and actor load | Full AMP resume preflights all state; actor-only playback uses new schema and sim2sim guard | No partial mutation; no training-only object in playback |
| error/cleanup | Parent runner owns join/terminate/IPC cleanup | Same ownership covers AMP fields and pre-spawn failures; child `finally` closes attachments/env | No zombie collector, leaked shared memory, queue or tracker |

## 8. Implementation batches and gates

### B0 — native runtime history export

Changes: P3-C00.  
Entry: user-confirmed exact plan hash; checkout whose release source matches the
bound 0.5.0 sdist; runtime issue/ADR and Git authority; current pre-slice
mature-method receipt.  
Exit: native direct/callback/history/compatibility tests; cross-platform wheels;
changelog; exact new artifact hash; refreshed post-slice mature-method receipt.  
Stop: the publisher source cannot be identity-bound, history requires multiple
Python/native step calls, default API/physics changes, or no released artifact
can be pinned by unisim.

### B1 — `unisim` history contract

Changes: P3-C01, P3-C02.  
Entry: confirmed plan hash; issue/ADR scope receipt; authorized isolated checkout;
pinned B0 runtime artifact; current pre-slice mature-method receipt.  
Exit: focused contract/MuJoCo tests, ON/OFF physical-state identity, package
checks, docs/changelog and a refreshed mature-method slice receipt.  
Stop: native runtime cannot expose history, dispatch count changes, reset rows
bleed, or disabled behavior changes.

### B2 — UniLab task/data boundary

Changes: P3-C03 through P3-C06. Depends on B1.  
Entry: B1 public import/reset semantics released or pinned; expert blob/manifest
identity available; confirmed Test Cards; current pre-slice mature-method
receipt.  
Exit: focused manager/task/asset/config tests and unchanged generic G1/APPO
snapshots.  
Stop: backend-private access, inherited pose/gait-phase authority, runtime asset
parsing, non-195D state or non-adjacent expert pair.

### B3 — `uni_rl` async AMP extension

Changes: P3-C07 through P3-C10. Depends on B2's frozen public observation and
expert-array contracts, but can implement RED owner tests after those interfaces
are published.  
Entry: uni_rl issue/ADR scope receipt, authorized isolated checkout, pinned
package-neutral types and Test Cards, and current pre-slice mature-method
receipt.  
Exit: exact order/terminal/IPC/checkpoint tests, existing APPO/HORA suite,
package type/lint/coverage gate, changelog and version at least `1.1.0`.  
Stop: reverse import, default rollout drift, invalid-batch mutation, partial
checkpoint mutation, or cleanup failure.

### B4 — composition and offline closure

Changes: P3-C11 through P3-C14. Depends on B2 and B3.  
Entry: pinned B0 runtime, B1 unisim and B3 uni_rl builds/schemas; all owner
suites green.  
Exit: UniLab focused tests, `make check`, `make test`, `make test-all`, exact
config/sim2sim/checkpoint save-reload proof and independent code review.  
Stop: generic path regression, legacy checkpoint acceptance, any incomplete
lifecycle proof, or current-head review finding.

### B5 — separately authorized bounded runtime probe

B5 makes no new production change. It is not admitted by this planning turn.
After B1-B4 and a formal-runtime-audit admission, run only a short fresh async
sentinel to prove spawn, rollout, learner ordering, metrics, atomic checkpoint,
resume-once and cleanup. Long training and policy-quality evaluation remain a
new user decision. No distillation process is started in Phase 3; its historical
process-lifecycle constraints are regression boundaries for generic cleanup only.

## 9. Verification ladder

1. RED: one independent behavior test per Test Card before its owner edit.
2. GREEN: smallest owner test; do not tune thresholds or iterations to force a
   pass.
3. Owner regression: unisim contract/MuJoCo, UniLab env/task/config/sim2sim,
   uni_rl IPC/APPO/HORA/checkpoint.
4. Package gates: unisim `make check && make package`; uni_rl documented
   format/mypy/pyright/pytest coverage gate; UniLab `make check && make test`.
5. Cross-package offline integration with exact pinned builds.
6. UniLab final gate `make test-all` on the final head.
7. Independent code review of the final diffs and current-head evidence.
8. Only after a new runtime authorization: bounded async sentinel. A sentinel
   proves wiring/lifecycle, not human-like gait quality.

## 10. Rollback and human decision boundary

Each of the four repositories/packages remains independently revertible by
commit and version pin.
B2 cannot ship without the exact B1 dependency; B4 pins the exact B3 checkpoint
schema. Never mix only some target packages into an old training run. If a batch
fails, preserve its failing test/evidence, revert only that batch in its isolated
branch, and keep the previous pinned package set.

This document is ready for review, not execution. The next human decision is to
confirm this exact content hash. Confirmation does not by itself authorize Git
actions, production edits, simulation or training; those remain explicit later
gates.
