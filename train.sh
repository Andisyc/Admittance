#!/usr/bin/env bash
set -euo pipefail

task="g1_walk_height"
sim="mujoco"
dry_run=0
overrides=()

usage() {
    cat <<'EOF'
Usage: ./train.sh [--task g1_walk_height] [--sim mujoco] [--dry-run] [override=value ...]

Runs the official single-teacher G1 distillation route:
  uv run --no-sync train --algo distill --task g1_walk_height --sim mujoco \
    teacher.checkpoint_path=/path/to/model_5000.pt

This route does not implement AMP-teacher, standing, transition, or MoE distillation.
EOF
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
        --task|--sim)
            if [ "$#" -lt 2 ] || [ -z "$2" ]; then
                echo "train.sh: $1 requires a value" >&2
                exit 2
            fi
            if [ "$1" = "--task" ]; then task=$2; else sim=$2; fi
            shift 2
            ;;
        *=*)
            overrides+=("$1")
            shift
            ;;
        *)
            echo "train.sh: unsupported argument: $1" >&2
            exit 2
            ;;
    esac
done

command=(uv run --no-sync train --algo distill --task "$task" --sim "$sim")
if [ "${#overrides[@]}" -gt 0 ]; then
    command+=("${overrides[@]}")
fi

printf '[train.sh] command:'
printf ' %q' "${command[@]}"
printf '\n'

if [ "$dry_run" -eq 1 ]; then
    exit 0
fi
exec "${command[@]}"
