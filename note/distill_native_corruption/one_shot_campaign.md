# Distillation Native-Corruption One-Shot Campaign

> Superseded for the current investigation by
> `note/distill_native_corruption/real_owner_offline_campaign.md`. The lifecycle and
> synthetic stress paths below are supporting evidence only; they do not replace the
> real aggregate/MoE/checkpoint/persistent-runtime campaign.

## Problem

The persistent G1 distillation workflow has produced language-semantics-impossible
objects at moving detection sites (`frame` in scenario labels and `cell` at a builtin
call), plus native abort captures that preserve the detector but not the first invalid
operation. Repeating full training with more Python prints is no longer an acceptable
primary diagnostic.

## Scope

- Exercise the production Role Data lifecycle through the existing data owner:
  load, annotate, aggregate, save, reload, validate, and release.
- Run active diagnostics in isolated child processes from one campaign invocation.
- Capture input identity, command/environment identity, process telemetry, stdout,
  stderr, exit status, timeout state, and a machine-readable verdict.
- Optionally add an exact offline CUDA replay and bounded formal fresh attempts.
- Produce one archive outside the campaign directory for retrieval.

## Non-scope

- Do not change distillation data, collector, learner, or workflow semantics.
- Do not overwrite historical datasets, checkpoints, cores, or run directories.
- Do not call allocator settings, cache disabling, serialization, or worker shutdown a
  fix.
- Do not combine Valgrind, rr, allocator debug, CUDA synchronization, or Compute
  Sanitizer in the same child identity.
- Do not report a clean bounded run as proof that native code is safe.

## Core parameter path

```text
source dataset paths
-> load_distillation_dataset
-> annotate_distillation_dataset_scenario
-> build_multitask_distillation_dataset
-> save_distillation_dataset
-> release and gc.collect
-> load_distillation_dataset
-> exact label/metadata/tensor fingerprint comparison
```

The lifecycle worker validates exact `str` element types before hashing so that
normalization cannot hide an impossible object. It also checks that top-level label
tuples and duplicated metadata lists remain identical.

## Campaign identities

| Identity | Active diagnostic | Primary question |
| --- | --- | --- |
| `host_plain` | none | Does the real lifecycle contradict its own semantic fingerprint? |
| `host_allocator_debug` | CPython/glibc debug allocator | Does host heap damage surface earlier? |
| `host_memcheck` | Valgrind Memcheck, isolated | Is there a first invalid host read/write/free? |
| `host_rr` | rr record, isolated | Can one failing CPU execution be replayed? |
| `collector_persistent` | one resident collector worker | Does failure require retained worker/resource state? |
| `collector_restart_each_request` | matched worker restart | Does removing retained state change the boundary? |
| `gpu_sync_replay` | `CUDA_LAUNCH_BLOCKING=1`, isolated | Does CUDA failure surface at its issuing call? |
| `gpu_memcheck_replay` | Compute Sanitizer, isolated | Is there a precise device invalid access? |
| `formal_native_attempt_N` | allocator debug and native abort capture | Does the full owner chain reproduce under bounded attempts? |

Tools set to `auto` are skipped with an explicit capability record when unavailable.
Skipped tools are not silently treated as passes.

## Evidence levels and stop rules

- `runtime-confirmed`: a named production lifecycle boundary completed.
- `native-symptom-confirmed`: fatal allocator, signal, impossible object, or CUDA
  runtime symptom occurred.
- `first-invalid-operation-confirmed`: Memcheck or Compute Sanitizer reports the first
  invalid operation with a native location.
- `owner-confirmed`: requires later source/ownership correlation; the campaign does
  not infer this from a detector stack alone.

The campaign verdict is one of:

- `FIRST_INVALID_OPERATION_CAPTURED`
- `ROOT_CAUSE_BOUNDARY_ISOLATED`
- `INCONCLUSIVE_NOT_REPRODUCED`

Infrastructure failure is recorded separately as `campaign_status=error`. A child
timeout is evidence about tool cost, not a pass or a native symptom.

## Files and ownership

- `scripts/deploy/check_distill_role_data_lifecycle.py`: data-owner lifecycle worker.
- `scripts/deploy/diagnose_distill_native_corruption.py`: diagnostic orchestration,
  isolation, telemetry, core discovery/symbolization, classification, and archive
  creation.
- `scripts/deploy/check_unilab_g1_distill_persistent_runtime.py`: live collector
  sentinel with matched `persistent` and `restart_each_request` lifecycle modes.
- `tests/scripts/test_distill_native_corruption_campaign.py`: semantic fixture,
  isolation, failure capture, verdict, and archive contracts.
- `src/unilab/algos/torch/distill/data.py`: unchanged business owner.

## Verification checklist

- A semantic three-scenario fixture crosses annotation, aggregation, roundtrip, and
  release for multiple cycles.
- The worker rejects non-`str` labels before normalization.
- Allocator debug, Valgrind, rr, CUDA sync, and Compute Sanitizer never share one
  child environment/command.
- A synthetic failing child is captured without aborting campaign summarization.
- Missing optional tools become `skipped`, not `passed`.
- Inputs have before/after SHA256 entries.
- The archive is created beside the work directory and does not include itself.
- Local commands use `uv run`.

## Current local evidence

- `uv run pytest tests/scripts/test_distill_native_corruption_campaign.py
  tests/scripts/test_g1_distill_persistent_runtime_probe.py -q`: 9 passed.
- Scenario/multitask owner regressions: 8 passed, 84 deselected.
- Persistent runtime/worker regressions: 8 passed.
- Focused Ruff and `py_compile`: passed.
- Local two-cycle one-shot campaign: `campaign_status=completed`, both host stages
  `runtime-confirmed`, input identity unchanged, zero new core candidates, and
  `INCONCLUSIVE_NOT_REPRODUCED` as required for a clean bounded toy run.
- Hydra configuration-only compose confirmed the 6000-update GPU replay route,
  scenario quotas, and minimum transition replay settings without starting training.

## Live-only gap

Local toy verification proves command construction, lifecycle semantics, isolation,
classification, and evidence packaging. It cannot prove the intermittent server bug
reproduces, that CUDA kernels are safe, or that a native component owns the first
invalid operation. Those facts require the generated server campaign.
