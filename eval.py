import argparse
import sys

from pathlib import Path

import torch
import datasets
import polars as pl

from transformers import AutoModelForSequenceClassification, AutoTokenizer
from torch.utils.data import DataLoader

from config import ModelConfig
from dataloader import CustomDataset

parser = argparse.ArgumentParser()
parser.add_argument("model_config_file", type=str)
parser.add_argument("model_path", type=Path)
parser.add_argument("--save_path", type=str, default="./output")
parser.add_argument("--num_workers", type=int, default=0)
parser.add_argument("--batch_size", type=int, default=0)

args = parser.parse_args()

datasets.config.IN_MEMORY_MAX_SIZE = int(2e+10)
torch.set_float32_matmul_precision('high')

def main():
    if not args.model_path.exists():
        print(f"model_path ({args.model_path}) does not exist", file=sys.stderr)
        sys.exit(1)

    model_config = ModelConfig.load_config(args.model_config_file)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_config.model_name, num_labels=model_config.num_labels
    )

    device_type = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_type)
    dtype = torch.float32

    checkpoint = torch.load(
        args.model_path,
        weights_only=False,
        map_location=device,
    )
    model.load_state_dict(checkpoint["state_dict"])
    model = model.to(device)
    model = torch.compile(model)

    tokenizer = AutoTokenizer.from_pretrained(model_config.model_name)
    dataset = datasets.load_dataset("carolina-c4ai/carol-domain-sents", split="test")
    dataset = CustomDataset(dataset, tokenizer, 512)

    dataloader = DataLoader(
        dataset,
        sampler=None,
        batch_size=args.batch_size or model_config.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        pin_memory=False,
    )

    y_pred = []
    y_true = []
    model.eval()
    with torch.no_grad(), torch.autocast(device_type=device_type, dtype=dtype):
        for x, y in dataloader:
            x = {k: v.to(device) for k, v in x.items()}
            y = y.to(device)
            y_true.append(y)
            y_pred.append(model(**x).logits.argmax(dim=-1))

    y_true = torch.concatenate(y_true).to("cpu").numpy()
    y_pred = torch.concatenate(y_pred).to("cpu").numpy()

    save_path = (
            Path(args.save_path) / Path(model_config.model_name) / "eval.csv"
    )
    save_path.parent.mkdir(parents=True, exist_ok=True)
    df = pl.DataFrame(dict(y_pred=y_pred, y_true=y_true))
    df.write_csv(save_path)


if __name__ == "__main__":
    main()
