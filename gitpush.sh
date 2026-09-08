#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

commit_message=${1:-"sync update"}
current_branch=$(git branch --show-current)
if [ -z "$current_branch" ]; then
    echo "gitpush.sh: detached HEAD; refusing to push." >&2
    exit 2
fi

# Staging is intentionally human-owned. Commit only changes staged beforehand.
if git diff --cached --quiet; then
    echo "gitpush.sh: no staged changes; skipping commit."
else
    git commit -m "$commit_message"
fi

echo "gitpush.sh: pushing origin/$current_branch"
git push -u origin "$current_branch"
