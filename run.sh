set -e
# classifier_lrs="0.01 0.005"
classifier_lrs="0.01"
encoder_lrs="0.00002 0.00004"
for classifier_lr in $classifier_lrs; do
    for encoder_lr in $encoder_lrs; do
        python main.py \
        --config example_config/Amazon-670K/deberta.yml \
        --learning_rate_encoder $encoder_lr \
        --learning_rate_classifier $classifier_lr
    done
done