"""One-run launcher: verify limits, restore provenance, and record bounded telemetry."""

import argparse
import gc
import hashlib
import json
import math
import os
import shutil
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


def write_json(path, value):
    temporary = path.with_suffix(".part")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def cpu_temperature():
    values = []
    for hwmon in Path("/sys/class/hwmon").glob("hwmon*"):
        try:
            if (hwmon / "name").read_text().strip() == "coretemp":
                values.extend(int(p.read_text()) / 1000 for p in hwmon.glob("temp*_input"))
        except OSError:
            pass
    return max(values) if values else None


class ThermalGate:
    def __init__(self, process_group, config):
        self.process_group = process_group
        self.pause_c = config["pause_c"]
        self.resume_c = config["resume_c"]
        assert self.resume_c < self.pause_c
        self.paused = False
        self.pause_count = 0
        self.resume_count = 0
        self.last_resume_wall = 0.0
        self.pause_started = 0.0
        self.cooling_seconds = 0.0

    def update(self, temperature):
        if temperature is None:
            raise RuntimeError("CPU temperature sensor unavailable")
        if not self.paused and temperature >= self.pause_c:
            os.killpg(self.process_group, signal.SIGSTOP)
            self.paused = True
            self.pause_started = time.monotonic()
            self.pause_count += 1
            print("THERMAL_PAUSED", temperature, flush=True)
        elif self.paused and temperature <= self.resume_c:
            os.killpg(self.process_group, signal.SIGCONT)
            self.cooling_seconds += time.monotonic() - self.pause_started
            self.paused = False
            self.last_resume_wall = time.time()
            self.resume_count += 1
            print("THERMAL_RESUMED", temperature, flush=True)

    def snapshot(self):
        elapsed = time.monotonic() - self.pause_started if self.paused else 0.0
        return dict(
            paused=self.paused,
            pause_c=self.pause_c,
            resume_c=self.resume_c,
            pause_count=self.pause_count,
            resume_count=self.resume_count,
            total_cooling_s=round(self.cooling_seconds + elapsed, 2),
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    run = args.run.resolve()
    plan = json.loads((run / "launch_plan.json").read_text())
    project = Path(plan["project"])
    code = Path(plan["code"])
    group = Path("/sys/fs/cgroup") / Path("/proc/self/cgroup").read_text().strip().split("::", 1)[
        1
    ].lstrip("/")
    limits = {
        name: (group / name).read_text().strip()
        for name in ("memory.high", "memory.max", "memory.swap.max")
    }
    affinity = sorted(os.sched_getaffinity(0))
    assert int(limits["memory.max"]) == 28 * 1024**3, limits
    assert int(limits["memory.high"]) == 24 * 1024**3, limits
    assert int(limits["memory.swap.max"]) <= 512 * 1024**2, limits
    assert affinity == plan["cpu_ids"], affinity
    # This host delegates only memory/pids to the user manager. taskset and
    # the matching env.cpu_ids enforce inherited process affinity instead.
    print(
        "VERIFIED_LIMITS",
        json.dumps({"cgroup": str(group), "limits": limits, "affinity": affinity}),
        flush=True,
    )

    import torch

    torch.set_num_threads(2)
    checkpoint = Path(plan["checkpoint"])
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    assert {"actor", "critic", "optimizer"} <= state.keys()

    def check(value):
        if isinstance(value, torch.Tensor) and (value.is_floating_point() or value.is_complex()):
            assert bool(torch.isfinite(value).all()), "checkpoint contains nonfinite tensor"
        elif isinstance(value, dict):
            for child in value.values():
                check(child)
        elif isinstance(value, (tuple, list)):
            for child in value:
                check(child)

    check(state)
    digest = hashlib.file_digest(checkpoint.open("rb"), "sha256").hexdigest()
    write_json(
        run / "checkpoint_preflight.json",
        {
            "path": str(checkpoint),
            "sha256": digest,
            "keys": list(state),
            "finite": True,
            "limits": limits,
            "affinity": affinity,
        },
    )
    del state
    gc.collect()
    print("CHECKPOINT_VALID", checkpoint, digest, flush=True)

    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    events = EventAccumulator(str(run), size_guidance={"scalars": 0})
    command = [
        "uv",
        "run",
        "--no-sync",
        "train",
        "--algo",
        "appo",
        "--task",
        "anybody_g1",
        "--sim",
        "mujoco",
        "--config-dir",
        str(run / "hydra"),
        "--config-name",
        "formal",
    ]
    console = (run / "console.log").open("ab", buffering=0)
    child = subprocess.Popen(
        command, cwd=code, stdout=console, stderr=subprocess.STDOUT, start_new_session=True
    )
    (run / "training.pid").write_text(str(child.pid) + "\n")
    pointer = project / ".tmp/latest-formal-run.txt"
    temporary = pointer.with_suffix(".part")
    temporary.write_text(str(run) + "\n")
    temporary.replace(pointer)
    started = time.time()
    thermal = (
        ThermalGate(child.pid, plan["thermal_control"]) if plan.get("thermal_control") else None
    )
    gpu_failures = 0
    pressure_count = 0
    stop_reason = None
    latest = {}

    def stop(signum=None, _frame=None):
        nonlocal stop_reason
        stop_reason = stop_reason or f"received signal {signum}"
        if child.poll() is None:
            try:
                os.killpg(child.pid, signal.SIGCONT)
                os.killpg(child.pid, signal.SIGTERM)
                child.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait(timeout=5)
            except ProcessLookupError:
                pass

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    print("TRAIN_STARTED", child.pid, json.dumps(command), flush=True)
    while True:
        now = time.time()
        try:
            events.Reload()
            tags = set(events.Tags().get("scalars", []))
            rows = events.Scalars("loss/policy_loss") if "loss/policy_loss" in tags else []
            metrics = {}
            for tag in (
                "loss/policy_loss",
                "loss/value_loss",
                "reward/mean",
                "grad/global_norm",
                "optim/learning_rate",
            ):
                samples = events.Scalars(tag) if tag in tags else []
                if samples:
                    metrics[tag] = samples[-1].value
            if any(not math.isfinite(v) for v in metrics.values()):
                stop_reason = "nonfinite training metric"
        except (OSError, ValueError) as exc:
            rows, metrics = [], {}
            print("EVENT_READ_PENDING", type(exc).__name__, str(exc), flush=True)
        available = (
            int(
                next(
                    line.split()[1]
                    for line in Path("/proc/meminfo").read_text().splitlines()
                    if line.startswith("MemAvailable:")
                )
            )
            * 1024
        )
        resource = {
            "memory_current_gib": int((group / "memory.current").read_text()) / 1024**3,
            "host_available_gib": available / 1024**3,
            "root_free_gib": shutil.disk_usage("/").free / 1024**3,
            "ssd_free_gib": shutil.disk_usage(project).free / 1024**3,
        }
        pressure = []
        try:
            probe = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=memory.used,temperature.gpu,utilization.gpu",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                check=True,
                timeout=5,
            )
            memory, temperature, utilization = map(
                float, probe.stdout.strip().splitlines()[0].split(",")
            )
            resource.update(
                gpu_memory_mib=memory,
                gpu_temperature_c=temperature,
                gpu_utilization_pct=utilization,
            )
            gpu_failures = 0
            if memory > 20 * 1024:
                pressure.append("GPU memory exceeds 20 GiB")
            if temperature >= 85:
                pressure.append("GPU temperature reaches 85 C")
        except (subprocess.SubprocessError, ValueError) as exc:
            gpu_failures += 1
            resource["gpu_probe_error"] = str(exc)
        temperature = cpu_temperature()
        if temperature is not None:
            resource["cpu_max_temperature_c"] = temperature
            if temperature >= 90 and thermal is None:
                pressure.append("CPU temperature reaches 90 C")
        if thermal and child.poll() is None:
            try:
                thermal.update(temperature)
            except RuntimeError as exc:
                stop_reason = str(exc)
            except ProcessLookupError:
                pass
        thermal_state = thermal.snapshot() if thermal else None
        if available < 12 * 1024**3:
            pressure.append("host available RAM below 12 GiB")
        if resource["root_free_gib"] < 2 or resource["ssd_free_gib"] < 20:
            pressure.append("disk reserve threshold crossed")
        pressure_count = pressure_count + 1 if pressure else 0
        if pressure_count >= 3:
            stop_reason = "; ".join(pressure)
        if gpu_failures >= 3:
            stop_reason = "three consecutive GPU status probe failures"
        now = time.time()
        age = now - rows[-1].wall_time if rows else now - started
        active_age = now - max(
            rows[-1].wall_time if rows else started, thermal.last_resume_wall if thermal else 0.0
        )
        if active_age > 600 and not (thermal and thermal.paused):
            stop_reason = "no new learner update for 600 seconds"
        if stop_reason:
            stop()
        returncode = child.poll()
        phase = (
            ("guard_stopped" if stop_reason else ("completed" if returncode == 0 else "failed"))
            if returncode is not None
            else (
                "cooling"
                if thermal and thermal.paused
                else ("training" if rows else "initializing")
            )
        )
        latest = {
            "time": datetime.now().astimezone().isoformat(),
            "phase": phase,
            "unit": plan["unit"],
            "training_pid": child.pid,
            "checkpoint_iteration": plan["restored_iteration"],
            "new_update_records": len(rows),
            "planned_new_iterations": plan["planned_new_iterations"],
            "new_env_steps": rows[-1].step if rows else 0,
            "metric_age_s": round(age, 1),
            "active_metric_age_s": round(active_age, 1),
            "thermal": thermal_state,
            "metrics": metrics,
            "resources": resource,
            "returncode": returncode,
            "stop_reason": stop_reason,
        }
        write_json(run / "status.json", latest)
        print(
            f"{latest['time']} {phase} new_updates={len(rows)}/{plan['planned_new_iterations']} from_checkpoint={plan['restored_iteration']} reward={metrics.get('reward/mean')} RAM={resource['memory_current_gib']:.1f}/28GiB GPU={resource.get('gpu_memory_mib')}MiB reason={stop_reason}",
            flush=True,
        )
        if returncode is not None:
            break
        time.sleep(plan.get("monitor_interval_s", 15))
    console.close()
    return 0 if latest["phase"] == "completed" else 1


if __name__ == "__main__":
    sys.exit(main())
