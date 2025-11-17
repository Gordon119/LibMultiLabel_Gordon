set -e
model=$1
data=$2
mode="last.ckpt"
for subdir in $model/$data/ensemble/*; do
  echo $subdir
  rm -rf $subdir/preds
  python main.py \
  --config example_config/$data/$model.yml \
  --checkpoint_path $subdir/$mode \
  --result_dir $subdir \
  --ensemble \
  --eval \
  --run_name preds
done
