import argparse
import os
import sys

from pathlib import Path

import torch

from datasets import load_dataset
from transformers import AutoModelForSequenceClassification

import util
from config import ModelConfig
from dataloader import AutoTokenizer, CustomDataset, DataLoader

parser = argparse.ArgumentParser()
parser.add_argument("model_config_file", type=str)
parser.add_argument("model_path", type=Path)
parser.add_argument("--save_path", type=str, default="./output")
parser.add_argument("--num_workers", type=int, default=1)

args = parser.parse_args()


def main():
    if not args.model_path.exists():
        print(f"model_path ({args.model_path}) does not exist", file=sys.stderr)
        sys.exit(1)

    model_config = ModelConfig.load_config(args.model_config_file)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_config.model_name, num_labels=model_config.num_labels
    )

    device_type = 'cuda'if torch.cuda.is_available() else 'cpu'
    device = torch.device(device_type) 
    dtype=torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float32

    checkpoint = torch.load(
        args.model_path,
        weights_only=False,
        map_location=device,
    )
    model.load_state_dict(checkpoint["state_dict"])
    model = torch.compile(model)
    model = model.to(device)

    dataset = load_dataset("mmcarpi/caroldb-sentences", split="test")

    tokenizer = AutoTokenizer.from_pretrained(model_config.model_name)
    dataset = CustomDataset(dataset, tokenizer, model_config.max_length)
    dataloader = DataLoader(
        dataset,
        batch_size=model_config.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        pin_memory=False,
    )

    compute_metrics = util.create_compute_metrics(
        model_config.num_labels, argmax_first=True
    )

    y_pred = []
    y_true = []
    model.eval()
    with torch.no_grad(), torch.autocast(device_type=device_type, dtype=dtype):
        for x, y in dataloader:
            x = {k : v.to(device) for k, v in x.items() }
            y = y.to(device)
            y_true.append(y)
            y_pred.append(model(**x).logits)

    y_true = torch.concatenate(y_true).to("cpu").numpy()
    y_pred = torch.concatenate(y_pred).to("cpu", dtype=torch.float32).numpy()
    metrics = compute_metrics(y_pred, y_true)

    save_path = Path(args.save_path) / Path(model_config.model_name) / "eval.json"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    util.save_metrics(metrics, save_path)


if __name__ == "__main__":
    main()
