import argparse
import sys

from pathlib import Path

import torch

from datasets import load_dataset
from transformers import AutoModelForSequenceClassification

import util
from config import ModelConfig
from dist import create_dataloader

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

    checkpoint = torch.load(
        args.model_path,
        weights_only=False,
        map_location=torch.device("cpu"),
    )
    model.load_state_dict(checkpoint["state_dict"])

    dataset = load_dataset("mmcarpi/caroldb-sentences", split="test").to_dict()
    dataloader = create_dataloader(
        dataset["text"],
        dataset["domain"],
        model_config.model_name,
        model_config.max_length,
        model_config.batch_size,
        num_workers=args.num_workers,
        is_distributed=False,
    )

    compute_metrics = util.create_compute_metrics(
        model_config.num_labels, argmax_first=True
    )

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
    save_path = Path(args.save_path) / Path(model_config.model_name) / "eval.json"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    util.save_metrics(metrics, save_path)


if __name__ == "__main__":
    main()
