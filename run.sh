set -e
sample_rate=0.1
for model in xmlcnn;
do
    for data in amazoncat-13k;
    do
        for idx in {0..25};
        do
            python3 main.py \
                --config example_config/xml_cnn.yml \
                --ensemble \
                --sample_rate 0.1 \
                --training_file data/$data/train.txt \
                --test_file data/$data/test.txt \
                --data_name $data \
                --result_dir $model/$data/ensemble
        done
        python3 test_eurlex.py \
            --pattern $model/$data \
            --train-path data/$data/train.txt \
            --test-path data/$data/test.txt \
            --output-path $model/$data/ensemble/logs.json

        python3 main.py \
            --config example_config/xml_cnn.yml \
            --training_file data/$data/train.txt \
            --test_file data/$data/test.txt \
            --data_name $data \
            --result_dir $model/$data/base
    done
done