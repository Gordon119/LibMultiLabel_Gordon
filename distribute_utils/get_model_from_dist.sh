#!/bin/bash
set -e

# Remote server info
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

# Hosts to connect to
HOSTS=("woodstock" "lucy" "svm")
# Remote ensemble directory
REMOTE_ENSEMBLE_DIR="$REMOTE_BASE_DIR/$model/$data/ensemble"

# Local destination
LOCAL_DEST="test"
mkdir -p "$LOCAL_DEST"

echo "Starting parallel downloads from ensemble directories..."

download_from_host() {
    host=$1
    echo "[$host] Connecting..."

    # Get subdirectories list from remote
    subdirs=$(ssh "$REMOTE_USER@$host.$REMOTE_HOST" \
        "ls -d $REMOTE_ENSEMBLE_DIR/*/ 2>/dev/null | xargs -n1 basename" 2>/dev/null)

    if [ $? -ne 0 ] || [ -z "$subdirs" ]; then
        echo "[$host] No subdirectories found"
        return
    fi

    # Pre-create local directories
    for subdir in $subdirs; do
        mkdir -p "$LOCAL_DEST/$host/$subdir"
    done

    # Build SFTP batch file
    batchfile=$(mktemp)
    {
        echo "cd $REMOTE_ENSEMBLE_DIR"
        for subdir in $subdirs; do
            echo "cd $subdir"
            echo "lcd ~/LibMultiLabel_Gordon/$LOCAL_DEST/$host/$subdir"
            echo "get -pr preds*"
            echo "get -p logs.json"
            echo "cd .."
        done
        echo "bye"
    } > "$batchfile"

    # Run SFTP
    sftp -b "$batchfile" "$REMOTE_USER@$host.$REMOTE_HOST"
    rm -f "$batchfile"

    echo "[$host] Completed"
}

# Run downloads in parallel
for host in "${HOSTS[@]}"; do
    download_from_host "$host" &
done

wait

echo "All downloads finished!"
echo "Files saved under: $LOCAL_DEST/{host}/{subdir}/"
