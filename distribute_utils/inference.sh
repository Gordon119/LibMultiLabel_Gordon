#!/bin/bash
set -e

REMOTE_USER="gordon1109"
REMOTE_HOST="csie.ntu.edu.tw"
REMOTE_BASE_DIR="LibMultiLabel_Gordon"

if [ $# -lt 2 ]; then
  echo "Usage: $0 <ensemble_runs> <sample_rate> <models> <datas>"
  echo
  echo "Arguments:"
  echo "  model         model name (e.g. 'cnn bert')"
  echo "  data        dataset (e.g. 'amazon eurlex')"
  exit 1
fi

model=$1
data=$2

# Hosts list
HOSTS=("local" "sally" "lucy")

run_on_host() {
    host=$1
    if [ "$host" == "local" ]; then
        echo "[local] Running commands locally..."
        rm -rf $model/$data/ensemble/*/preds
        bash run_utils/run_inference.sh $model $data
        echo "[local] Done"
    else
        echo "[$host] Running remote commands..."
        ssh "$REMOTE_USER@$host.$REMOTE_HOST" bash -c "'
            # Send commands to tmux session 0
            tmux send-keys -t 0 \"cd ~/LibMultiLabel_Gordon\" C-m
            tmux send-keys -t 0 \"git fetch origin\" C-m
            tmux send-keys -t 0 \"git reset --hard origin/ensemble\" C-m
            tmux send-keys -t 0 \"conda activate ensemble\" C-m
            tmux send-keys -t 0 \"ln -s ../data\" C-m
            tmux send-keys -t 0 \"bash run_utils/run_inference.sh $model $data\" C-m
        '"
        echo "[$host] Done"
    fi
}

for host in "${HOSTS[@]}"; do
    run_on_host "$host" &
done

wait
echo "All remote jobs finished."
