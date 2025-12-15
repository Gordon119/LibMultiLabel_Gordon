set -e

for encoder_lr in 0.00005 0.00007;
do
    for clf_lr in 0.005 0.01;
    do
        python main.py \
        --config example_config/Amazon-131K/sbert_tune.yml \
        --learning_rate_encoder $encoder_lr \
        --learning_rate_classifier $clf_lr
    done
done
