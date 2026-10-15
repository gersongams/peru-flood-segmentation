#!/usr/bin/env bash
# Run ON the Mac. Copies what phase-D self-training needs to a pod, and NO Peru label file:
# repo bundle, the pool images (configs/pools/peru-train.csv), the peru-dev images (for
# label-free dev predictions on the pod), the Peru split lists, Sen1Floods11 S2Hand +
# LabelHand + splits (source data), and the slim source checkpoints staged by
# scripts/cloud/stage_phaseD.py.
#   scripts/cloud/sync_phaseD.sh <host> <port>
set -euo pipefail
HOST=${1:?usage: sync_phaseD.sh <host> <port>}
PORT=${2:?usage: sync_phaseD.sh <host> <port>}
REPO=$(cd "$(dirname "$0")/../.." && pwd)
SSH="ssh -p $PORT -i $HOME/.ssh/id_ed25519_runpod -o StrictHostKeyChecking=accept-new -o ServerAliveInterval=30"
R=/workspace/peru-flood-segmentation
cd "$REPO"
B=$(mktemp -d)/repo.bundle
git bundle create "$B" main
rsync -rlt -e "$SSH" "$B" scripts/cloud/setup_pod.sh "root@$HOST:/workspace/"
P=data/peru/unosat2023
L=$(mktemp)
{ tail -n +2 configs/pools/peru-train.csv | cut -d, -f1; cat $P/splits/dev.txt; } | sed 's/$/.tif/' > "$L"
$SSH "root@$HOST" mkdir -p $R/$P/s2 $R/$P/splits
rsync -rlt --partial --timeout=120 --stats -e "$SSH" --files-from="$L" $P/s2/ "root@$HOST:$R/$P/s2/" | tail -3
rsync -rlt -e "$SSH" $P/splits/train.txt $P/splits/dev.txt $P/splits/split.csv "root@$HOST:$R/$P/splits/"
D=data/sen1floods11/v1.1
$SSH "root@$HOST" mkdir -p $R/$D/data/flood_events/HandLabeled
rsync -rlt --partial --timeout=120 --stats -e "$SSH" $D/data/flood_events/HandLabeled/S2Hand $D/data/flood_events/HandLabeled/LabelHand \
  "root@$HOST:$R/$D/data/flood_events/HandLabeled/" | tail -3
rsync -rlt -e "$SSH" $D/splits "root@$HOST:$R/$D/"
rsync -rlt --partial --timeout=120 --stats -e "$SSH" outputs/phaseD/upload/outputs "root@$HOST:$R/" | tail -3
# No Peru label may be on the pod.
$SSH "root@$HOST" "test ! -e $R/$P/label && test ! -e $R/data/peru/emsr2017 && test ! -e $R/data/unosat2023 && echo 'no Peru labels on pod'"
