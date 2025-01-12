import ctypes

import torch
import numpy as np
import torch.multiprocessing as mp

from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoTokenizer


def create_dataloader(
    data, tokenizer_name, max_length, batch_size, num_workers=0, is_distributed=False
):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    dataset = CustomDataset(data, tokenizer, max_length)

    if is_distributed:
        sampler = DistributedSampler(dataset)
        shuffle = False
    else:
        sampler = None
        shuffle = True

    dataloader = DataLoader(
        dataset,
        sampler=sampler,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        pin_memory=True,
    )

    return dataloader


def create_shared_array(ctype, size):
    shared_array_base = mp.Array(ctype, size)
    shared_array = np.ctypeslib.as_array(shared_array_base.get_obj())
    torch_array = torch.from_numpy(shared_array)
    return torch_array


class CustomDataset(Dataset):
    def __init__(self, data, tokenizer, max_length):
        self.x_shared_array = create_shared_array(
            ctypes.c_int, len(data) * max_length * 2
        ).reshape(len(data), max_length, 2)
        self.y_shared_array = create_shared_array(ctypes.c_long, len(data))
        self.use_cache = False
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        unique_labels = sorted(set(data["domain"]))
        self.encoder = {label: i for i, label in enumerate(unique_labels)}

    def set_use_cache(self, use_cache):
        self.use_cache = use_cache

    def __getitem__(self, index):
        if not self.use_cache:
            data = self.data[index]
            text = data["text"]
            label = self.encoder[data["domain"]]
            tokenized = self.tokenizer(
                text,
                add_special_tokens=True,
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            self.x_shared_array[index, :, 0] = tokenized["input_ids"]
            self.x_shared_array[index, :, 1] = tokenized["attention_mask"]
            self.y_shared_array[index] = label
        x = {
            "input_ids": self.x_shared_array[index, :, 0],
            "attention_mask": self.x_shared_array[index, :, 1],
        }
        y = self.y_shared_array[index]
        return x, y

    def __len__(self):
        return len(self.data)
