set -e

if [ $# -lt 3 ]; then
  echo "Usage: $0 <model> <data> <dist_dir>"
  echo
  echo "Arguments:"
  echo "  model     Model name (e.g. kimcnn, bert)"
  echo "  data      Dataset name (e.g. amazon-670k, eurlex)"
  echo "  dist_dir  Directory with distributed checkpoints"
  exit 1
fi


model=$1
data=$2
dist_dir=$3

python3 test.py \
    --model-dirs $dist_dir \
    --train-path data/$data/train.txt \
    --test-path data/$data/test.txt \
    --output-path $model/$data/ensemble/logs.json

python3 main.py \
    --config example_config/$data/$model.yml \
    --training_file data/$data/train.txt \
    --test_file data/$data/test.txt \
    --data_name $data \
    --result_dir $model/$data/base
