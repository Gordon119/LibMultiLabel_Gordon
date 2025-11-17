import os
import json
import argparse
import warnings
import torch
from tqdm import tqdm
import numpy as np
import glob
import scipy

import libmultilabel.nn.data_utils as data_utils
from libmultilabel.nn.metrics import get_metrics, tabulate_metrics
from pytorch_lightning import seed_everything

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)


def prepare_data(data_path, batch_size, embed_file):
    datasets = data_utils.load_datasets(
        training_data=data_path["train"],
        test_data=data_path["test"]
    )
    classes = data_utils.load_or_build_label(datasets)
    word_dict, _ = data_utils.load_or_build_text_dict(
        dataset=datasets["train"],
        vocab_file=None,
        min_vocab_freq=1,
        embed_file=embed_file,
        silent=True,
        normalize_embed=False,
        embed_cache_dir=None,
    )
    dataloader = data_utils.get_dataset_loader(
        data=datasets["test"],
        classes=classes,
        device=torch.device("cpu"),
        batch_size=batch_size,
        shuffle=False,
        word_dict=word_dict
    )
    return dataloader, classes


def metrics_from_npz_predictions(dataloader, classes, root_dir, max_models=100):
    metrics = get_metrics(0.5, ["P@1", "P@3", "P@5"], num_classes=len(classes))

    # Load all model info upfront
    hosts = glob.glob(f"{root_dir}/*")
    model_info = []
    
    for host in hosts:
        model_dirs = glob.glob(f"{host}/*")
        for model_dir in model_dirs:
            if len(model_info) >= max_models:
                break
            
            log_path = os.path.join(model_dir, "logs.json")
            pred_path = os.path.join(model_dir, "preds")

            if not os.path.exists(log_path) or not os.path.exists(pred_path):
                print(f"Skipping {model_dir}, missing preds dir or logs.json")
                continue

            try:
                with open(log_path, "r") as f:
                    log_data = json.load(f)
                    if "config" not in log_data:
                        print(f"Skipping {model_dir}, missing config in logs.json")
                        continue
                    
                    # Get indices if available (sparse format), otherwise None (dense format)
                    indices = log_data["config"].get("indices", None)
                
                model_info.append((pred_path, indices))
            except (json.JSONDecodeError, KeyError) as e:
                print(f"Skipping {model_dir}, error reading logs.json: {e}")
                continue
        
        if len(model_info) >= max_models:
            break

    if len(model_info) == 0:
        raise ValueError(f"No valid models found in {root_dir}")

    print(f"Loaded {len(model_info)} models")
    
    # Check label coverage
    all_indices = set()
    for _, indices in model_info:
        if indices is not None:
            all_indices.update(indices)
        else:
            all_indices.update(range(len(classes)))
    
    coverage = len(all_indices) / len(classes) * 100
    print(f"Label coverage: {len(all_indices)}/{len(classes)} ({coverage:.1f}%)")
    
    if coverage < 100:
        uncovered = len(classes) - len(all_indices)
        print(f"Warning: {uncovered} labels not covered by any model")
    
    # Main evaluation loop
    for batch_idx, batch in enumerate(tqdm(dataloader, desc="Evaluating")):
        bsz = batch["label"].shape[0]
        preds_sum = np.zeros((bsz, len(classes)), dtype=np.float32)
        preds_cnt = np.zeros((bsz, len(classes)), dtype=np.int32)

        for pred_dir, indices in model_info:
            batch_file = os.path.join(pred_dir, f"preds_{batch_idx}.npz")
            
            if not os.path.exists(batch_file):
                print(f"Warning: Missing batch {batch_idx} for model {pred_dir}, skipping")
                continue
            
            data = np.load(batch_file)
            
            # Check format: sparse (has 'indices') or dense (only 'values')
            if 'indices' in data:
                # Sparse format
                top_idx = data['indices']  # Shape: (batch_size, K)
                top_val = data['values']    # Shape: (batch_size, K)
                
                if top_idx.shape[0] != bsz:
                    print(f"Warning: Batch size mismatch in {batch_file}: "
                          f"expected {bsz}, got {top_idx.shape[0]}")
                    continue
                
                top_val = scipy.special.expit(top_val)
                
                indices_array = np.array(indices, dtype=np.int32)
                global_idx = indices_array[top_idx]            

                preds_sum[np.arange(bsz)[:, None], global_idx] += top_val
                preds_cnt[np.arange(bsz)[:, None], global_idx] += 1
            else:
                # Dense format
                values = data['values']  # Shape: (batch_size, num_labels)
                
                if values.shape[0] != bsz:
                    print(f"Warning: Batch size mismatch in {batch_file}: "
                          f"expected {bsz}, got {values.shape[0]}")
                    continue
                
                values = scipy.special.expit(values)
                
                if indices is not None:
                    # Model predicts subset of labels
                    indices_array = np.array(indices, dtype=np.int32)
                    preds_sum[:, indices_array] += values
                    preds_cnt[:, indices_array] += 1
                else:
                    # Model predicts all labels
                    preds_sum += values
                    preds_cnt += 1

        # Average predictions per sample per label
        preds_avg = np.divide(
            preds_sum,
            preds_cnt,
            out=np.zeros_like(preds_sum),
            where=preds_cnt != 0,
        )

        target = batch["label"].cpu()
        metrics.update(torch.from_numpy(preds_avg), target)

    return metrics.compute()


def parse_args():
    parser = argparse.ArgumentParser(description="Ensemble multi-label predictions using checkpoints")

    parser.add_argument(
        "--model-dirs", type=str, required=True,
        help="Root directory containing model subdirectories with preds/ and logs.json"
    )
    parser.add_argument("--train-path", type=str, default="data/amazon-670k/train.txt")
    parser.add_argument("--test-path", type=str, default="data/amazon-670k/test.txt")
    parser.add_argument("--output-path", type=str, default="xmlcnn/amazon-670k/ensemble/logs.json")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--embed-file", default="glove.6B.300d")
    parser.add_argument("--max-models", type=int, default=100,
                        help="Maximum number of models to ensemble (default: 100)")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    seed_everything(1337)

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)

    dataloader, classes = prepare_data(
        {"train": args.train_path, "test": args.test_path},
        batch_size=args.batch_size, embed_file=args.embed_file
    )
    
    results = metrics_from_npz_predictions(
        dataloader, classes, args.model_dirs, max_models=args.max_models
    )

    results = {k: v.item() for k, v in results.items()}

    print("\n" + "="*50)
    print("Ensemble Results:")
    print("="*50)
    print(tabulate_metrics(results, "test"))
    print("="*50 + "\n")
    
    json.dump(results, open(args.output_path, "w"), indent=2)
    print(f"Results saved to: {args.output_path}")