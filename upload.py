import json
from argparse import ArgumentParser
from pathlib import Path

import requests
import torch
from huggingface_hub import create_repo
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from config import ModelConfig

parser = ArgumentParser()
parser.add_argument("repo_id")
parser.add_argument("model_config_file", type=Path)
parser.add_argument("checkpoint", type=Path)

args = parser.parse_args()

create_repo(args.repo_id, private=True, exist_ok=True)


model_config = ModelConfig.load_config(args.model_config_file)

cpu = torch.device('cpu')
model = AutoModelForSequenceClassification.from_pretrained(
    model_config.model_name,
    num_labels=model_config.num_labels
).to(cpu)
tokenizer = AutoTokenizer.from_pretrained(model_config.model_name)


checkpoint = torch.load(args.checkpoint, map_location=cpu, weights_only=False)
model.load_state_dict(checkpoint["state_dict"])

model.push_to_hub(args.repo_id, commit_message="Upload model tensors")
tokenizer.push_to_hub(args.repo_id, commit_message="Upload model tokenizer")

