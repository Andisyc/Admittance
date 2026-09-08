#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

current_branch=$(git branch --show-current)
if [ -z "$current_branch" ]; then
    echo "gitpull.sh: detached HEAD; refusing to pull." >&2
    exit 2
fi

exec git pull --ff-only origin "$current_branch"
