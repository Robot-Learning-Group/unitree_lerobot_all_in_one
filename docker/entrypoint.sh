#!/usr/bin/env bash
set -euo pipefail

if (( $# == 0 )); then set -- bash; fi
if [[ $(id -u) == 0 ]]; then
    task_uid=${HOST_UID:-$(stat -c %u /workspace)}
    task_gid=${HOST_GID:-$(stat -c %g /workspace)}
    if [[ ! $task_uid =~ ^[0-9]+$ || ! $task_gid =~ ^[0-9]+$ ]]; then
        echo 'HOST_UID and HOST_GID must be numeric.' >&2
        exit 1
    fi
    if ! getent group "$task_gid" >/dev/null; then
        groupadd --gid "$task_gid" "lerobot_g${task_gid}"
    fi
    if ! getent passwd "$task_uid" >/dev/null; then
        useradd --uid "$task_uid" --gid "$task_gid" --create-home \
            --shell /bin/bash "lerobot_u${task_uid}"
    fi
    account_home=$(getent passwd "$task_uid" | cut -d: -f6)
    # HOME must match the selected account, not the image's root account.
    export HOME="$account_home"
    if [[ $task_uid != 0 ]]; then
        exec gosu "$task_uid:$task_gid" "$0" "$@"
    fi
fi

# Create only runtime directories, as the application user. Do not chown sources.
mkdir -p "$HF_HOME" "$HF_LEROBOT_HOME" "$TORCH_HOME" "$MPLCONFIGDIR" \
    /workspace/unitree_lerobot/outputs
exec "$@"
