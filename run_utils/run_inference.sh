set -e
model=$1
data=$2
mode="last.ckpt"
for subdir in $model/$data/ensemble/*; do
  echo $subdir
  rm -rf $subdir/preds
  python main.py \
  --config example_config/$data/$model.yml \
  --indice_file $subdir/logs.json \
  --checkpoint_path $subdir/$mode \
  --result_dir $subdir \
  --word_dict_path $subdir/word_dict.pickle \
  --ensemble \
  --eval \
  --run_name preds
done
