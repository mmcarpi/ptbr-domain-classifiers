import torch
from torch.utils.data import Dataset
from functools import partial


def tokenize_dataset(dataset, tokenizer, max_length):

    def preprocess_function(examples, label2id, max_length):
        processed = tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=max_length,
        )
        processed["label"] = [label2id[domain] for domain in examples["domain"]]
        return processed

    domains = sorted(dataset.unique("domain"))
    label2id = dict(zip(domains, range(len(domains))))
    dataset = dataset.map(
        partial(preprocess_function, label2id=label2id, max_length=max_length),
        batched=True,
        batch_size=2048,
        keep_in_memory=True,
        remove_columns=[
            col for col in dataset.features.keys() if col not in ["domain", "text"]
        ],
    )

    return dataset


class DumbDataset(Dataset):
    def __init__(self, hfdataset, device):
        self.input_ids = []
        self.attn_mask = []
        self.label = []
        for row in hfdataset:
            self.input_ids.append(torch.tensor(row["input_ids"], device=device))
            self.attn_mask.append(torch.tensor(row["attention_mask"], device=device))
            self.label.append(torch.tensor(row["label"], device=device))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, index):
        out = dict(
            input_ids=self.input_ids[index],
            attention_mask=self.attn_mask[index],
            label=self.label[index],
        )
        return out
