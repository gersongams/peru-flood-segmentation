#!/usr/bin/env bash
# Run ON the Mac. Copies pod-session-2 runs (scripts/cloud/queue_phaseD2.sh) back in parallel
# streams and verifies every file (same list and sizes on both sides). Only finished seeds
# (a `done` marker) are copied, so it can run while other seeds are still training.
#   scripts/cloud/fetch_phaseD2.sh <host> <port> [parallel=8]
# /workspace/outputs/<RUN_NAME>/<model>/lightning_logs/seed<s> → outputs/<RUN_NAME>/..., plus each
# run folder's logs/ and queue.txt, the queue logs and prithvi_lr.txt (→ outputs/phaseD/pod2/).
set -euo pipefail
HOST=${1:?usage: fetch_phaseD2.sh <host> <port> [parallel]}
PORT=${2:?usage: fetch_phaseD2.sh <host> <port> [parallel]}
NP=${3:-8}
REPO=$(cd "$(dirname "$0")/../.." && pwd)
SSH="ssh -p $PORT -i $HOME/.ssh/id_ed25519_runpod -o StrictHostKeyChecking=accept-new -o ServerAliveInterval=30"
cd "$REPO"
units=$($SSH "root@$HOST" "cd /workspace && for d in outputs/s1f11_*/*/lightning_logs/seed*; do [ -f \$d/done ] && echo \$d; done; ls -d outputs/s1f11_*/logs")
one="sleep \$((RANDOM % 4)); mkdir -p \"\$1\"; for t in 1 2 3 4; do rsync -rlt --partial --timeout=120 -e '$SSH' \"root@$HOST:/workspace/\$1/\" \"\$1/\" && exit 0; sleep \$((t * 5)); done; echo \"FAIL \$1\""
echo "$units" | xargs -P "$NP" -I{} bash -c "$one" _ {}
mkdir -p outputs/phaseD/pod2
rsync -rlt -e "$SSH" "root@$HOST:/workspace/queue_phaseD2_*.log" "root@$HOST:/workspace/prithvi_lr.txt" "root@$HOST:/workspace/queue*.sh" outputs/phaseD/pod2/
for r in $(echo "$units" | cut -d/ -f2 | sort -u); do rsync -rlt -e "$SSH" "root@$HOST:/workspace/outputs/$r/queue.txt" "outputs/$r/"; done
dirs=$(echo "$units" | tr '\n' ' ')
remote=$($SSH "root@$HOST" "cd /workspace && find $dirs -type f -printf '%s %p\n' | LC_ALL=C sort -k2")
local=$(find $dirs -type f ! -name '.*' -exec stat -f '%z %N' {} + | LC_ALL=C sort -k2)
if [ "$remote" = "$local" ]; then
  echo "verified: $(echo "$local" | wc -l | tr -d ' ') files match"
else
  diff <(echo "$remote") <(echo "$local") | head -20; echo "MISMATCH"; exit 1
fi
