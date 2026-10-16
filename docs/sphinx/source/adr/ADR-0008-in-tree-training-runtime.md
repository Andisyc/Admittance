---
orphan: true
---

# ADR-0008 In-tree Training Runtime

- Status: Accepted
- Date: 2026-10-16
- Owners: UniLab maintainers
- Supersedes: External uni_rl training ownership; does not supersede ADR-0007 physics extraction
- Superseded by: None

## Context

The current environment and script owners consume AMP, distillation and APPO
interfaces absent from the pinned published training package. The user explicitly
requested maintaining training implementations inside UniLab instead of waiting
for an external package release. This changes the training package owner, not
the selected robot, reward, observation layout or physics backend.

## Decision

Algorithms, runners, learners, collectors, IPC and training logging live in
`src/unilab/rl`. Runtime imports and owner configuration use `unilab.rl`; remove
the `unilab-rl` distribution dependency. Physics remains behind the declared
unisim backend interface. Torch and existing public foundation dependencies
remain dependencies of the project.

## Stable Contracts

- Keep pickleable `EnvFactory` injection; generic training modules do not
  construct task-specific environments or import task registrations.
- Preserve existing algorithm defaults, observation/action meaning, terminal
  transitions and resource cleanup while relocating modules.
- Protect APPO's single-producer/single-consumer rollout slots until the learner
  has copied them and advanced its read pointer. The restored source's unchecked
  overwrite on a full ring could mix different rollout generations during
  staging; use cancellable producer backpressure instead. Loss formulas remain
  unchanged, but collection scheduling and full-ring discard behavior change;
  throughput requires measurement rather than an equivalence claim.
- Snapshot only a payload writer's declared pre-step observation groups before
  calling the environment: EnvFactory does not promise immutable observation
  buffers. AMP declares its current-state group; default APPO adds no such copy.
- Restore AMP and distillation with real implementations. Empty import shims,
  ignored missing modules and bypassed CI are not implementation substitutes.
- No automatic conversion of old checkpoints or claims of compatibility solely
  from the namespace move. Existing state schemas remain governed by their
  loaders and dimension/identity validation.
- The initial generic source is Apache-2.0 `unilab-rl==1.0.0`; its provenance
  and copyright remain recorded in `src/unilab/rl/README.md`.

## Alternatives Considered

- Wait for unpublished external training releases: rejected by the user's
  explicit decision to own training code in this repository.
- Install the old ReGait `unilab` distribution as a dependency: rejected because
  it collides with this package and its interfaces do not match this checkout.
- Disable failing checks or substitute placeholder algorithms: rejected because
  imports alone would not establish a usable training path.

## Consequences

UniLab now maintains and tests its training implementation. The source relocation
does not prove that AMASS is connected to the Encoder/Decoder, that a training run
has succeeded or that the physics backend supports every optional capability.
Runtime validation remains necessary, and local training stays prohibited by the
current user instruction.

## Evidence In Repo

- `src/unilab/rl/`
- `src/unilab/base/env_factory.py`
- `src/unilab/scripts/train_appo.py`
- `src/unilab/scripts/train_distill.py`
- `pyproject.toml`
- `collaboration/overview.md`

## Related Documents

- {doc}`ADR index </adr/ADR-0000-index>`
- {doc}`Physics extraction </adr/ADR-0007-unisim-extraction-boundary>`
