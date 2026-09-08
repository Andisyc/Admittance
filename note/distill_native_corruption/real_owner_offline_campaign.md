# Distillation Real-Owner Offline Campaign

## Problem

The observed `frame` and `cell` values are impossible under the reachable Python
transformations. They establish a native-corruption symptom and moving Python victims,
not the component that first wrote or freed invalid memory. Additional victim-local
prints and another formal DAgger run are out of scope.

The r10 console is the most useful long learner trace: the real MoE learner completed
update 4914 and `run_offline_distillation_updates` caught
`TypeError("'cell' object is not callable")` at update 4915. The core traceback ends in
recursive `torch.nn.Module.train()`. That line is the detector. A normal module tree
cannot turn its call target into a cell.

## Scope

- Statically audit the complete saved-data owner chain.
- Rebuild the existing r10 aggregate through the production data owner in matched CPU
  and GPU fresh processes.
- Run the existing r10 aggregate/checkpoint through the production MoE offline owner in
  matched CPU and GPU fresh processes for 6000 updates, beyond the observed r10 failure
  window.
- Keep a real `PersistentDistillationRuntime` and `SharedWeightSync` resident while the
  learner updates, reload each output checkpoint, and exercise cleanup.
- Compare GPU continuous, restart-each-round, and two concurrent owner identities with
  equal per-process inputs and update budgets.
- Unpack the existing `gpu_sync_replay` Apport report before GDB, recover the handled
  exception preserved by CPython, collect new cores, and capture Xid/MCE/EDAC/GPU/RAM
  health facts.
- Produce one retrieval archive from one server command.

## Non-scope

- No simulator, environment reset, rollout, or formal DAgger training.
- No new local business-value prints and no training-semantic fix.
- No `UNILAB_NATIVE_ABORT_ON_CORRUPTION=1` in new offline stages, because that wrapper
  replaces the original Python exception with `SIGABRT`.
- No claim that a clean bounded run proves native safety or authorizes a live run.
- The synthetic numeric canary remains a platform-pressure auxiliary. It does not cross
  the real aggregate, MoE, offline update, checkpoint, or persistent runtime owners.

## Static owner audit

| Stage | Real owner and transition | Code-confirmed lifecycle fact | Risk / interpretation |
| --- | --- | --- | --- |
| Aggregate assembly | `scripts/train_distill.py::run_multitask_dataset_assembly` -> `data.py::build_multitask_distillation_dataset` | Every source is loaded with `device=_distill_device(cfg)`, annotated, retained, concatenated, validated, then saved through CPU-detached tensors | Formal GPU mode materializes all source tensors and the aggregate on CUDA before the CPU save. This is substantial allocator pressure, but r10 later failed inside update 4915, so assembly is not an owner-confirmed writer |
| Dataset load | `data.py::load_distillation_dataset` | `torch.load(map_location=device, weights_only=False)` is followed by `.to(device)` for each tensor and reconstruction of immutable label tuples | Correct serialization boundary; redundant `.to(device)` is pressure, not evidence of corruption |
| Trainer construction | `scripts/train_distill.py::build_distillation_trainer` and `_load_student_init_checkpoint` | Student, teacher, and optimizer are created on the selected device. The init checkpoint is loaded by `load_distillation_student_policy`, then by `load_distillation_checkpoint` again | `load_distillation_student_policy` itself performs one raw `torch.load` and another inside `load_distillation_checkpoint`; `_load_student_init_checkpoint` performs a third checkpoint load. This is code-confirmed repeated target-device loading, not a proven invalid operation |
| Offline loop | `offline.py::run_offline_distillation_updates` | The whole dataset stays resident on the selected device; balanced indices stage batches; each batch enters the real trainer and checkpoint save | r10 proves this owner ran normally through update 4914 and detected the impossible call at 4915 |
| MoE update | `trainer.py::BehaviorDistillationTrainer.update` | `student.train()` -> cached teacher target -> MoE forward/router/expert action -> role and command-intent targets -> loss/backward/Adam | `target_indices` is freshly allocated and the prior bytecode check falsified the stale-list hypothesis. `Module.train()` and target-label code are victims/detectors until a first invalid operation is captured |
| Checkpoint reload | `checkpoint.py` and `playback.py` | Checkpoint is loaded to the target device, a new MLP/MoE policy is constructed, state is loaded strictly, and the policy is frozen | Repeated model construction/loading changes CPU/GPU allocator state and is included in every campaign identity |
| Persistent runtime | `persistent_runtime.py::PersistentDistillationRuntime` | Parent loads a checkpoint, copies the full state dict to CPU, creates `SharedWeightSync`, spawns a resident service, and publishes monotonically versioned weights | The campaign keeps this runtime resident before learner update so the same-card learner/collector native runtimes coexist without a simulator |
| Shared weights | `ipc/weight_sync.py::SharedWeightSync` | State is flattened into a float32 POSIX shared-memory array; a locked version is stored after the data; the worker copies slices into its model | Shape/key validation exists. Dtype is implicitly float32, the version offset can be only 4-byte aligned when total numel is odd, and there is no final offset assertion. None is currently a first-invalid-operation finding |
| Cleanup | `PersistentDistillationRuntime.close` -> runner close -> worker service close -> parent `SharedWeightSync.cleanup` | Worker is joined before parent unlink. Queue and pipe resources are then closed | `SharedWeightSync.close/cleanup` swallow every exception and keep NumPy buffer views as attributes while closing the mapping. A close failure can skip unlink and become invisible. The campaign records whether the shared-memory name remains after close; this is a resource-lifecycle audit, not yet the corruption root cause |

## Differential contract

| Comparison | Constant facts | One changed variable |
| --- | --- | --- |
| Aggregate CPU vs GPU fresh | r10 seed aggregate metadata, exact source files/order/roles/scenarios, production builder, validation and roundtrip | data device |
| Offline CPU vs GPU fresh | r10 aggregate, r10 iteration-3 checkpoint, walking teacher, Hydra workflow config, batch 512, 6000 updates, persistent checkpoint service, exception policy | learner/runtime device |
| GPU continuous vs restart | same inputs, config, 2048 updates per round, three rounds, checkpoint publication and cleanup checks | whether process and persistent service survive across rounds |
| GPU continuous vs dual resident | identical per-process continuous workload | one vs two concurrent owner identities on the same GPU; extra system load is the direct consequence of this variable |

Every new stage clears active allocator/sanitizer settings and explicitly disables the
native-abort wrapper. The campaign does not stack CUDA synchronization, Compute
Sanitizer, Valgrind, allocator perturbation, or logging changes on these lifecycle
controls.

## Apport contract

`.crash` is an Apport report container, not an ELF core. The campaign runs
`apport-unpack <report.crash> <isolated-dir>`, reads `ExecutablePath`, and gives GDB the
extracted `CoreDump`. GDB records all native threads, all Python backtraces, libraries,
and the current `PyThreadState` handled exception fields. The original `.crash` remains
untouched; only the extracted multi-GB copy is removed after textual GDB evidence is
written.

## Files

- `scripts/deploy/check_distill_real_owner_path.py`: real aggregate and offline owner
  worker, persistent checkpoint service, SharedWeightSync identity and cleanup checks.
- `scripts/deploy/diagnose_distill_real_owner_one_shot.py`: matched stage matrix, health
  snapshots, stage isolation, core collection, classification, and archive.
- `scripts/deploy/diagnose_distill_native_corruption.py`: corrected Apport/raw-core GDB
  handling and CPython handled-exception extraction.
- `tests/scripts/test_distill_real_owner_campaign.py`: semantic aggregate, real MoE
  persistent checkpoint, stage matrix, cleanup, and Apport-unpack contracts.

## Completion gate

Local completion means focused tests and Ruff pass, the owner worker help/compose path
loads, the real tiny MoE checkpoint crosses a spawned persistent worker twice, and
Apport tests prove GDB never receives the `.crash` path. It does not mean the server bug
is fixed or reproduced.

Formal live training remains disallowed after a merely clean offline archive. It can be
considered only after the archive shows all real CPU/GPU owner paths are clean and a
separate evidence review positively implicates simulator or formal persistent-live
lifecycle.
