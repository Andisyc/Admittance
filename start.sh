#!/usr/bin/env bash
set -euo pipefail

algo="sac"
task="g1_walk_flat"
task_explicit=0
sim="mujoco"
render_mode="interactive"
load_run=""
checkpoint=""
checkpoint_path=""
dry_run=0
overrides=()

usage() {
    cat <<'EOF'
Usage: ./start.sh [options] [RUN_NAME] [hydra.override=value ...]

Options:
  --algo <name>             Algorithm passed to the official eval CLI.
  --task <name>             Registry task name (default: g1_walk_flat).
  --sim <backend>           Simulation backend (default: mujoco).
  --render-mode <mode>      auto, interactive, record, or none.
  --load-run <name>         Run directory name resolved by the eval CLI.
  --checkpoint <value>      Forwarded as algo.checkpoint=<value>.
  --checkpoint-path <path>  Direct checkpoint path. Distill loads the student checkpoint.
  --dry-run                 Print the command without starting playback.

For --algo distill, the default task is g1_walk_height and --checkpoint-path is
required. The checkpoint must use the uni_rl.distill.checkpoint.v1 schema.
EOF
}

require_value() {
    if [ "$#" -lt 2 ] || [ -z "$2" ]; then
        echo "start.sh: $1 requires a value" >&2
        exit 2
    fi
}

while [ "$#" -gt 0 ]; do
    case "$1" in
        -h|--help)
            usage
            exit 0
            ;;
        --dry-run)
            dry_run=1
            shift
            ;;
        --algo|--task|--sim|--render-mode|--load-run|--checkpoint|--ckpt|--checkpoint-path|--ckpt-path)
            require_value "$@"
            flag=$1
            value=$2
            case "$flag" in
                --algo) algo=$value ;;
                --task) task=$value; task_explicit=1 ;;
                --sim) sim=$value ;;
                --render-mode) render_mode=$value ;;
                --load-run) load_run=$value ;;
                --checkpoint|--ckpt) checkpoint=$value ;;
                --checkpoint-path|--ckpt-path) checkpoint_path=$value ;;
            esac
            shift 2
            ;;
        --)
            shift
            overrides+=("$@")
            break
            ;;
        *=*)
            overrides+=("$1")
            shift
            ;;
        -*)
            echo "start.sh: unknown option: $1" >&2
            exit 2
            ;;
        *)
            if [ -n "$load_run" ]; then
                echo "start.sh: only one run name may be specified" >&2
                exit 2
            fi
            load_run=$1
            shift
            ;;
    esac
done

if [ "$algo" = "distill" ] && [ "$task_explicit" -eq 0 ]; then
    task="g1_walk_height"
fi

if [ -n "$load_run" ] && [ -n "$checkpoint_path" ]; then
    echo "start.sh: use either --load-run or --checkpoint-path, not both" >&2
    exit 2
fi

if [ "$algo" = "distill" ]; then
    if [ -n "$load_run" ] || [ -n "$checkpoint" ]; then
        echo "start.sh: distill playback currently requires --checkpoint-path; --load-run/--checkpoint are unsupported." >&2
        exit 2
    fi
    if [ -z "$checkpoint_path" ]; then
        echo "start.sh: --algo distill requires --checkpoint-path." >&2
        exit 2
    fi
fi

command=(
    uv run --no-sync eval
    --algo "$algo"
    --task "$task"
    --sim "$sim"
    --render-mode "$render_mode"
)
if [ -n "$load_run" ]; then
    command+=(--load-run "$load_run")
fi
if [ -n "$checkpoint_path" ]; then
    if [ "$algo" = "distill" ]; then
        command+=("student.initial_checkpoint_path=$checkpoint_path")
    else
        command+=("algo.load_run=$checkpoint_path")
    fi
fi
if [ -n "$checkpoint" ]; then
    command+=("algo.checkpoint=$checkpoint")
fi
if [ "${#overrides[@]}" -gt 0 ]; then
    command+=("${overrides[@]}")
fi

printf '[start.sh] command:'
printf ' %q' "${command[@]}"
printf '\n'

if [ "$dry_run" -eq 1 ]; then
    exit 0
fi

exec "${command[@]}"
