import argparse
import sys

from pathlib import Path

import torch
from torch.utils.data import DataLoader

from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

import util
from config import ModelConfig
from dist import collate_fn  # TODO move to utils, move to config

parser = argparse.ArgumentParser()
parser.add_argument("model_config_file", type=str)
parser.add_argument("model_path", type=Path)
parser.add_argument("--save_path", type=str, default="./output")
parser.add_argument("--num_workers", type=int, default=1)
parser.add_argument("--cuda", action="store_true")

args = parser.parse_args()


def main():
    if not args.model_path.exists():
        print(f"model_path ({args.model_path}) does not exist", file=sys.stderr)
        sys.exit(1)

    config = ModelConfig.load_config(args.model_config_file)
    model = AutoModelForSequenceClassification.from_pretrained(
        config.model_name, num_labels=config.num_labels
    )
    tokenizer = AutoTokenizer.from_pretrained(config.model_name, use_fast=False)

    checkpoint = torch.load(
        args.model_path,
        weights_only=False,
        map_location=None if args.cuda else torch.device("cpu"),
    )
    model.load_state_dict(checkpoint["state_dict"])

    dataset = load_dataset("mmcarpi/caroldb-sentences", split="test")
    dataset = util.tokenize_dataset(dataset, tokenizer, config.max_length)
    dataset = dataset.with_format("torch")

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=config.batch_size,
        shuffle=False,
        pin_memory=False,
        num_workers=args.num_workers,
        drop_last=False,
        collate_fn=collate_fn,
    )

    compute_metrics = util.create_compute_metrics(config.num_labels, argmax_first=True)

    y_pred = []
    y_true = []
    model.eval()
    with torch.no_grad():
        for x, y in dataloader:
            y_true.append(y)
            y_pred.append(model(**x).logits)

    y_true = torch.concatenate(y_true).to("cpu").numpy()
    y_pred = torch.concatenate(y_pred).to("cpu").numpy()
    metrics = compute_metrics(y_pred, y_true)
    save_path = Path(args.save_path) / Path(config.model_name) / 'eval.json'
    save_path.parent.mkdir(parents=True, exist_ok=True)
    util.save_metrics(metrics, save_path)


main()
