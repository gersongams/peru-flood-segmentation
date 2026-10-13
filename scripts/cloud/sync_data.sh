#!/usr/bin/env bash
# Run ON the Mac. Copies the repo (as a git bundle of main) and the Peru data to a pod.
#   scripts/cloud/sync_data.sh <host> <port>     # from the pod's SSH block (public IP + port)
set -euo pipefail
HOST=${1:?usage: sync_data.sh <host> <port>}
PORT=${2:?usage: sync_data.sh <host> <port>}
REPO=$(cd "$(dirname "$0")/../.." && pwd)
SSH="ssh -p $PORT -i $HOME/.ssh/id_ed25519_runpod -o StrictHostKeyChecking=accept-new"

cd "$REPO"
B=$(mktemp -d)/repo.bundle
git bundle create "$B" main
rsync -az -e "$SSH" "$B" scripts/cloud/setup_pod.sh "root@$HOST:/workspace/"
# Only what the Peru runs read; data/ keeps the repo layout so the configs work unchanged.
$SSH "root@$HOST" mkdir -p /workspace/peru-flood-segmentation/data/peru
rsync -az --stats -e "$SSH" data/peru/unosat2023 data/peru/emsr2017 \
  "root@$HOST:/workspace/peru-flood-segmentation/data/peru/"
