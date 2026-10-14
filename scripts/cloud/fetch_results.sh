#!/usr/bin/env bash
# Run ON the Mac. Copies a run folder (metrics, run.json, best checkpoints, logs) back
# from a pod's volume and checks the copy file by file.
#   scripts/cloud/fetch_results.sh <host> <port> [runs_cloud | runs_cloud_s1f11_ft]
# No compression (checkpoints do not compress, and -z stalled on a 3.9 GB file);
# --partial and --timeout let a stalled transfer be rerun and resume.
set -euo pipefail
HOST=${1:?usage: fetch_results.sh <host> <port> [name]}
PORT=${2:?usage: fetch_results.sh <host> <port> [name]}
NAME=${3:-runs_cloud}
REPO=$(cd "$(dirname "$0")/../.." && pwd)
SSH="ssh -p $PORT -i $HOME/.ssh/id_ed25519_runpod -o StrictHostKeyChecking=accept-new -o ServerAliveInterval=30"
rsync -rlt --partial --timeout=120 --stats -e "$SSH" "root@$HOST:/workspace/outputs/$NAME" "$REPO/outputs/"
# Same file list and sizes on both sides?
remote=$($SSH "root@$HOST" "cd /workspace/outputs/$NAME && find . -type f -printf '%s %p\n' | LC_ALL=C sort -k2")
local=$(cd "$REPO/outputs/$NAME" && find . -type f ! -name '.*' -exec stat -f '%z %N' {} + | LC_ALL=C sort -k2)
if [ "$remote" = "$local" ]; then echo "verified: $(echo "$local" | wc -l | tr -d ' ') files match"; else
  diff <(echo "$remote") <(echo "$local") | head; echo "MISMATCH"; exit 1; fi
