#!/usr/bin/env bash
# Run ON the Mac. Copies run outputs (metrics, run.json, checkpoints, logs) back from a pod.
#   scripts/cloud/fetch_results.sh <host> <port>
set -euo pipefail
HOST=${1:?usage: fetch_results.sh <host> <port>}
PORT=${2:?usage: fetch_results.sh <host> <port>}
REPO=$(cd "$(dirname "$0")/../.." && pwd)
SSH="ssh -p $PORT -i $HOME/.ssh/id_ed25519_runpod -o StrictHostKeyChecking=accept-new"
rsync -rltz --stats -e "$SSH" \
  "root@$HOST:/workspace/peru-flood-segmentation/outputs/runs_cloud" \
  "root@$HOST:/workspace/peru-flood-segmentation/outputs/logs" "$REPO/outputs/"
