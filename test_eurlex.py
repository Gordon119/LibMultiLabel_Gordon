import glob
import numpy as np
import os
from tqdm import tqdm
import json
from libmultilabel.nn.metrics import get_metrics, tabulate_metrics
import libmultilabel.nn.data_utils as data_utils
import pickle
import torch
import warnings
import argparse

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

def prepare_data(data_path):
    datasets = data_utils.load_datasets(
        training_data=data_path["train"],
        test_data=data_path["test"]
    )
    classes = data_utils.load_or_build_label(datasets)
    word_dict, _ = data_utils.load_or_build_text_dict(
        dataset=datasets["train"],
        vocab_file=None,
        min_vocab_freq=500,
        embed_file="glove.6B.300d",
        silent=True,
        normalize_embed=False,
        embed_cache_dir=None,
    )
    dataloader = data_utils.get_dataset_loader(
            data=datasets["test"],
            classes=classes,
            device=torch.device("cuda"),
            batch_size=256,
            shuffle=False,
            word_dict=word_dict
        )
    return dataloader, classes

def metrics_from_npz_predictions(dataloader, batch_size, classes, pattern):
    pattern = pattern + "*/data.npz"
    prediction_files = glob.glob(pattern)
    print(f"Found {len(prediction_files)} prediction files")

    # Initialize metrics (assuming you have linear.get_metrics available)
    metrics = get_metrics(0.5, ["P@1", "P@3", "P@5"], num_classes=len(classes))
    
    for batch_idx, batch in tqdm(enumerate(dataloader)):
        total_preds = np.zeros([batch["label"].shape[0], len(classes)])
        total_cnts = np.zeros(len(classes))
        for pred_file in prediction_files:
            root = os.path.dirname(pred_file)
            indices = json.load(open(os.path.join(root, "logs.json"), "r"))["config"]["indices"]
            preds = np.load(pred_file, allow_pickle=True)['arr_0'][batch_idx]["pred_scores"]
            total_preds[:, indices] += preds
            total_cnts[indices] += 1
        target = batch["label"]
        total_preds = np.divide(total_preds, total_cnts[None, :], 
                       out=np.zeros_like(total_preds), 
                       where=total_cnts[None, :] != 0)
        metrics.update(torch.from_numpy(total_preds), target)

    return metrics.compute()

def parse_args():
    parser = argparse.ArgumentParser(description='Ensemble multi-label predictions')
    
    parser.add_argument('--pattern', type=str, 
                        default="ensemble/eurlex",
                        help='Glob pattern for prediction files (default: ensemble/eurlex*/data.npz)')
    
    parser.add_argument('--train-path', type=str,
                        default="data/eurlex/train.txt",
                        help='Path to training data (default: data/eurlex/train.txt)')
    
    parser.add_argument('--test-path', type=str,
                        default="data/eurlex/test.txt", 
                        help='Path to test data (default: data/eurlex/test.txt)')
    
    parser.add_argument('--output-path', type=str,
                        default="xmlcnn/eurlex/ensemble/logs.json",
                        help='Path to save results JSON (default: ensemble/logs.json)')
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    
    dataloader, classes = prepare_data({
        "train": args.train_path, 
        "test": args.test_path
    })
    
    results = metrics_from_npz_predictions(dataloader, 256, classes, args.pattern)
    
    # Convert tensor values to Python scalars for JSON serialization
    for key, value in results.items():
        results[key] = value.item()
    
    print("Metrics results:", tabulate_metrics(results, "test"))
    
    # Save results to specified output path
    json.dump(results, open(args.output_path, "w"))
    print(f"Results saved to: {args.output_path}")