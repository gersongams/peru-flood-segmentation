#!/usr/bin/env bash
# Run ON the Mac. Pod session 2 (Prithvi source models, training-time alignment): copies the
# repo bundle, setup script and outputs/phaseD/alignment/train_params.npz (image statistics
# only). No Peru image or label goes to the pod; Sen1Floods11 is downloaded on the pod.
#   scripts/cloud/sync_phaseD2.sh <host> <port>
set -euo pipefail
HOST=${1:?usage: sync_phaseD2.sh <host> <port>}
PORT=${2:?usage: sync_phaseD2.sh <host> <port>}
REPO=$(cd "$(dirname "$0")/../.." && pwd)
SSH="ssh -p $PORT -i $HOME/.ssh/id_ed25519_runpod -o StrictHostKeyChecking=accept-new -o ServerAliveInterval=30"
R=/workspace/peru-flood-segmentation
cd "$REPO"
B=$(mktemp -d)/repo.bundle
git bundle create "$B" main
rsync -rlt -e "$SSH" "$B" scripts/cloud/setup_pod.sh "root@$HOST:/workspace/"
$SSH "root@$HOST" mkdir -p $R/outputs/phaseD/alignment
rsync -rlt -e "$SSH" outputs/phaseD/alignment/train_params.npz "root@$HOST:$R/outputs/phaseD/alignment/"
local_sum=$(shasum -a 256 outputs/phaseD/alignment/train_params.npz | cut -d' ' -f1)
remote_sum=$($SSH "root@$HOST" sha256sum $R/outputs/phaseD/alignment/train_params.npz | cut -d' ' -f1)
[ "$local_sum" = "$remote_sum" ] && echo "train_params.npz sha256 ok" || { echo "sha256 MISMATCH"; exit 1; }
$SSH "root@$HOST" "test ! -e $R/data/peru && echo 'no Peru data on pod'"
