set -e

if [ $# -lt 2 ]; then
  echo "Usage: $0 <models> <datas>"
  echo
  echo "Arguments:"
  echo "  models         Space-separated list of model names (e.g. 'cnn bert')"
  echo "  datas          Space-separated list of datasets (e.g. 'amazon eurlex')"
  exit 1
fi

model=$1
datas=$2

for data in $datas
do
    python main.py \
    --config example_config/$data/$model.yml \
    --report_model_size

    # python main.py \
    # --config example_config/$data/$model.yml \
    # --report_model_size \
    # --ensemble \
    # --sample_rate 0.1
done