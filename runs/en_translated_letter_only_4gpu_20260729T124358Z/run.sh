#!/usr/bin/env bash
# Per-run entrypoint. Paths are pinned in run_env.sh next to this file.
set -euo pipefail
RUN_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${RUN_DIR}/run_env.sh"
exec bash "${RUN_DIR}/scripts/train_launch.sh"
