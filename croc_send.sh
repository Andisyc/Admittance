#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
    printf 'Usage: %s FILE\n' "$0" >&2
    exit 2
fi

# Run on the server. OSC 52 copies the receiver command to the local terminal.
croc send --no-local "$1" 2>&1 | while IFS= read -r line; do
    printf '%s\n' "$line"

    if [[ $line =~ (CROC_SECRET=\"[^\"]+\"[[:space:]]+croc.*)$ ]]; then
        command=${BASH_REMATCH[1]}
        encoded=$(printf '%s' "$command" | base64 | tr -d '\r\n')
        printf '\033]52;c;%s\a' "$encoded" > /dev/tty
        printf '\n[Copied to Mac clipboard] %s\n\n' "$command"
    fi
done
