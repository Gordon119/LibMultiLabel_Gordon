set -e

for lr in 0.006 0.007 0.008 0.02 0.03;
do
    python main.py --config example_config/AmazonCat-13K/sbert.yml --learning_rate_classifier $lr
done