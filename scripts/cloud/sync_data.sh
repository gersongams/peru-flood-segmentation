#!/usr/bin/env bash
# Run ON the Mac. Copies the repo (as a git bundle of main) and the Peru data to a pod.
#   scripts/cloud/sync_data.sh <host> <port>     # from the pod's SSH block (public IP + port)
#   SYNC_S1F11=1 scripts/cloud/sync_data.sh ...  # also copy Sen1Floods11 S2Hand + LabelHand + splits
set -euo pipefail
HOST=${1:?usage: sync_data.sh <host> <port>}
PORT=${2:?usage: sync_data.sh <host> <port>}
REPO=$(cd "$(dirname "$0")/../.." && pwd)
SSH="ssh -p $PORT -i $HOME/.ssh/id_ed25519_runpod -o StrictHostKeyChecking=accept-new"

cd "$REPO"
B=$(mktemp -d)/repo.bundle
git bundle create "$B" main
rsync -rltz -e "$SSH" "$B" scripts/cloud/setup_pod.sh "root@$HOST:/workspace/"
# Only what the Peru runs read; data/ keeps the repo layout so the configs work unchanged.
$SSH "root@$HOST" mkdir -p /workspace/peru-flood-segmentation/data/peru
rsync -rltz --stats -e "$SSH" data/peru/unosat2023 data/peru/emsr2017 \
  "root@$HOST:/workspace/peru-flood-segmentation/data/peru/"
if [ "${SYNC_S1F11:-0}" = 1 ]; then
  D=data/sen1floods11/v1.1
  $SSH "root@$HOST" mkdir -p /workspace/peru-flood-segmentation/$D/data/flood_events/HandLabeled
  rsync -rltz --stats -e "$SSH" $D/data/flood_events/HandLabeled/S2Hand $D/data/flood_events/HandLabeled/LabelHand \
    "root@$HOST:/workspace/peru-flood-segmentation/$D/data/flood_events/HandLabeled/"
  rsync -rltz -e "$SSH" $D/splits "root@$HOST:/workspace/peru-flood-segmentation/$D/"
fi
