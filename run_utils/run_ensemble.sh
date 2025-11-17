#!/bin/bash
set -e

if [ $# -lt 4 ]; then
  echo "Usage: $0 <ensemble_runs> <sample_rate> <models> <datas>"
  echo
  echo "Arguments:"
  echo "  ensemble_runs  Number of ensemble iterations (default: 1 if omitted)"
  echo "  sample_rate    Sampling rate used for training"
  echo "  models         Space-separated list of model names (e.g. 'cnn bert')"
  echo "  datas          Space-separated list of datasets (e.g. 'amazon eurlex')"
  exit 1
fi

ENSEMBLE_RUNS=${1:-1}
sample_rate=$2
models=$3
datas=$4
echo "total: $ENSEMBLE_RUNS"

for model in $models; do
    for data in $datas; do
        if [ "$ENSEMBLE_RUNS" -gt 0 ]; then
            for idx in $(seq 0 $((ENSEMBLE_RUNS-1))); do
                echo "iter: $idx"
                python3 main.py \
                    --config "example_config/$data/$model.yml" \
                    --ensemble \
                    --sample_rate "$sample_rate" \
                    --data_name "$data" \
                    --result_dir "$model/$data/ensemble"
            done
        fi
    done
done
