#!/usr/bin/env bash
# Run ON the Mac. Copies phase-D results back from a pod in parallel streams (one SSH stream to
# the pod can be slow, ~0.5 MB/s) and verifies every file (same list and sizes on both sides).
#   scripts/cloud/fetch_phaseD.sh <host> <port> [parallel=8]
# Copies outputs/phaseD/self_train/ (run folders: logs, run.json, packed dev predictions, final
# weights) and outputs/logits_dev/<regime>/ for every regime produced on the pod.
set -euo pipefail
HOST=${1:?usage: fetch_phaseD.sh <host> <port> [parallel]}
PORT=${2:?usage: fetch_phaseD.sh <host> <port> [parallel]}
NP=${3:-8}
REPO=$(cd "$(dirname "$0")/../.." && pwd)
R=/workspace/peru-flood-segmentation
SSH="ssh -p $PORT -i $HOME/.ssh/id_ed25519_runpod -o StrictHostKeyChecking=accept-new -o ServerAliveInterval=30"
cd "$REPO"
# One unit per run folder and per logits folder, relative to the repo.
units=$($SSH "root@$HOST" "cd $R && ls -d outputs/phaseD/self_train/*/*/seed* && for t in \$(ls outputs/phaseD/self_train); do ls -d outputs/logits_dev/\$t/*/seed* 2>/dev/null || true; done")
# The folder is passed as $1, not substituted into the script: BSD xargs -I limits a
# substituted argument to 255 bytes.
one="sleep \$((RANDOM % 4)); mkdir -p \"\$1\"; for t in 1 2 3 4; do rsync -rlt --partial --timeout=120 -e '$SSH' \"root@$HOST:$R/\$1/\" \"\$1/\" && exit 0; sleep \$((t * 5)); done; echo \"FAIL \$1\""
echo "$units" | xargs -P "$NP" -I{} bash -c "$one" _ {}
dirs=$(echo "$units" | tr '\n' ' ')
remote=$($SSH "root@$HOST" "cd $R && find $dirs -type f -printf '%s %p\n' | LC_ALL=C sort -k2")
local=$(find $dirs -type f ! -name '.*' -exec stat -f '%z %N' {} + | LC_ALL=C sort -k2)
if [ "$remote" = "$local" ]; then
  echo "verified: $(echo "$local" | wc -l | tr -d ' ') files match"
else
  diff <(echo "$remote") <(echo "$local") | head -20; echo "MISMATCH"; exit 1
fi
