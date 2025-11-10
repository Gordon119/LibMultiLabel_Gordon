set -e

for data in EUR-Lex Wiki10-31K AmazonCat-13K Amazon-670K
do
    if [ "$data" = "Amazon-670K" ]; then
        python main.py --config example_config/$data/kim_cnn.yml --report_model_size
        python main.py --config example_config/$data/kim_cnn.yml --report_model_size --ensemble --sample_rate 0.1
    else
        python main.py --config example_config/$data/xml_cnn.yml --report_model_size
        python main.py --config example_config/$data/xml_cnn.yml --report_model_size --ensemble --sample_rate 0.1
    fi
done