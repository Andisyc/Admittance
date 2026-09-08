# Phase 3 AMP async local execution receipt

Status: `LOCAL_IMPLEMENTATION_VERIFIED / RELEASE_PIN_BLOCKED`  
Plan: `UNILAB-AMP-PHASE3-PLAN-v001`, revision 2  
Confirmed plan SHA-256: `f74a65f16d42acc471d6359c2cb72b6b7f3a0d6d4b489a5e0185ca008f746c27`  
Execution date: 2026-09-08  

## Result

The fixed-forward, walk-only G1 AMP path is implemented across
`mujoco-uni-runtime`, `unisim`, `uni_rl`, and UniLab. Local integration preserves
one native `nstep=4` transaction, the asynchronous APPO collector/learner
lifecycle, learner-only AMP mutable state, full AMP resume, and actor-only
playback. No standing, running, fall recovery, gait phase, height control,
right-hand-specific shaping, or distillation behavior was added.

The implementation is not release-closed. `mujoco-uni-runtime==0.6.0`, a new
`unisim-core` release containing `bind_sensor_history`, `unilab-rl==1.1.0`, and
the two expert motion blobs must be published before UniLab can update its
indexed dependency pins and lockfile. Local verification therefore used a
current native wheel plus editable `unisim` and `unilab_rl` checkouts with
`uv run --no-sync`. No commit, push, package publication, or long training was
performed.

## Verification

- `mujoco-uni-runtime`: 43 passed, 5 skipped; current native wheel built.
- `unisim`: 114 passed, 5 skipped in its owner gate; final focused integration
  rerun 32 passed, 5 skipped.
- `unilab-rl`: 389 passed, 26 skipped, 3 deselected; Ruff, Mypy, Pyright, wheel,
  and sdist passed.
- UniLab: Ruff, Mypy, and Pyright passed (one pre-existing optional Drake import
  warning); 1606 passed, 21 skipped, 903 deselected, 1 expected failure; coverage
  70%; benchmark smoke passed. The AMP MuJoCo slow construction/step test passed.
- The indexed-package-only test is intentionally not counted as passing because
  the required releases do not yet exist. It correctly rejects the editable
  local integration environment.

## Runtime sentinel

Fresh official entrypoint:

```bash
uv run --no-sync train --algo appo --task g1_amp_walk --sim mujoco \
  algo.num_envs=2 algo.steps_per_env=4 algo.max_iterations=2 algo.save_interval=1 \
  'algo.amp.discriminator_batch_size=16' 'algo.amp.hidden_dims=[64,32]' \
  training.device=cpu training.collector_device=cpu training.no_play=true \
  training.log_dir=/private/tmp/unilab-amp-phase3-final-20260908 \
  training.logger=tensorboard
```

The run completed and wrote `model_2.pt`. Its learner checkpoint has schema
`uni_rl.amp_appo` version 1, discriminator version 2, replay size 32, APPO update
counter 2, finite actor tensors, and expert adjacency
`within_clip_t_to_t_plus_1`.

One resume run completed from that checkpoint. The resulting learner state has
discriminator version 3, replay size 40, and APPO update counter 3, proving that
the discriminator, replay, and APPO state continued rather than restarted.

Actor-only official eval completed with the actor observation mapping, exported
and verified `policy.onnx` (`max_diff=1.49e-07`), and rendered a two-frame
`play_video.mp4`. No training-only AMP object or expert motion was loaded by the
play runtime. The separate native-interactive route is covered by an owner
composition regression proving that it applies the same runtime context and
selects the 96D actor observation rather than the 291D flattened observation;
the GUI itself was not launched during this headless verification. No collector
or training process remained after completion.

These sentinels prove wiring, checkpoint continuity, playback, and lifecycle
cleanup only. They do not establish learnability, final gait quality, throughput
on the training server, or a production training-duration estimate.

## Release gate

The next safe action is publication and pin closure in dependency order:
`mujoco-uni-runtime`, `unisim-core`, `unilab-rl`, motion blobs, then UniLab
`pyproject.toml` and `uv.lock`. After that, rerun ordinary indexed `uv run`,
`make check`, `make test`, and `make test-all` without editable packages. Long
training remains a separate human-authorized step.
