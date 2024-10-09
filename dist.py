import argparse
import gc
import os
import sys
import time
from contextlib import nullcontext

import torch
from torch.distributed import init_process_group, destroy_process_group
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

from datasets import load_dataset

import optuna
from optuna.trial import TrialState

from transformers import AutoTokenizer, AutoModelForSequenceClassification
#from train import Trainer, init_model, warm_up_scheduler

import data

start_time = None


def start_timer():
    global start_time
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    start_time = time.time()


def end_timer_and_print(local_msg):
    #torch.cuda.synchronize()
    end_time = time.time()
    print("\n" + local_msg)
    print("Total execution time = %.3f sec" % (end_time - start_time))
    print("Max memory used by tensors = %d bytes" % (torch.cuda.max_memory_allocated()))


def forward_backward_autocast(model, optimizer, loss_fn, input_ids, attention_mask, target, device_type):
    with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
        output = model(input_ids=input_ids, attention_mask=attention_mask).logits
        loss = loss_fn(output, target)

    loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)


def train_loop_autocast(model, dataloader, optimizer, loss_fn, num_epochs, gradient_accumulation_steps, device_type):
    start_timer()
    model.train()
    for epoch in range(num_epochs):  # 0 epochs, this section is for illustration only
        dataloader.sampler.set_epoch(epoch)
        #for input_ids, attn_mask, target in dataloader:
        for i, data in enumerate(dataloader):
            # Runs the forward pass under ``autocast``.
            with (nullcontext() if i % gradient_accumulation_steps == 0 else model.no_sync()):
                forward_backward_autocast(model, optimizer, loss_fn, data['input_ids'], data['attention_mask'], data['label'], device_type)

    end_timer_and_print("Automatic precision")


def eval_loop_autocast(model, batched_dataset):
    model.eval()
    with torch.no_grad():
        acc = torch.Tensor(0.0)
        for input_ids, attn_mask, target in batched_dataset:
            prediction = model(
                input_ids=input_ids, attention_mask=attn_mask
            ).logits.argmax(dim=-1)
            acc += (prediction == target).sum()
        acc /= len(batched_dataset) * batched_dataset.batch_size
    return acc.item()


def main():

    model_name = "neuralmind/bert-base-portuguese-cased"
    num_labels = 5
    max_length = 256
    batch_size = 64
    gradient_accumulation_steps = 1024 // batch_size

    num_epochs = 1
    learning_rate = 1e-4
    weight_decay = 0.01

    init_process_group(backend="nccl")
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    print("Hello from", local_rank, rank)

    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
    model = torch.compile(model).to(local_rank)
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )

    dataset = load_dataset("mmcarpi/caroldb-sentences", split="hps")#.select(range(16_000))
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)

    dataset = data.tokenize_dataset(dataset, tokenizer, max_length)
    dataset = dataset.with_format('torch', device=local_rank)
    #dataset = data.DumbDataset(dataset, device=local_rank)


    sampler = DistributedSampler(dataset)

    dataloader = DataLoader(
            dataset=dataset,
            sampler=sampler,
            batch_size=batch_size,
            shuffle=False,
            pin_memory=False,
    )

    loss_fn = torch.nn.CrossEntropyLoss().to(local_rank)

    print(f"[GPU{local_rank}] starting training soon...")
    train_loop_autocast(model, dataloader, optimizer, loss_fn, num_epochs, gradient_accumulation_steps, device_type='cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[GPU{local_rank}] finished training")

    destroy_process_group()


if __name__ == "__main__":
    main()
sys.exit(0)

model_name = "neuralmind/bert-base-portuguese-cased"
model_name = "PORTULAN/albertina-900m-portuguese-ptbr-encoder"
#model_name = "PORTULAN/albertina-1b5-portuguese-ptbr-encoder-256"

def train_loop(model, opt, loss_fn, data, mask, target):
    start_timer()
    for epoch in range(num_epochs):
        for x, m, t in zip(data, mask, target):
            output = model(input_ids=x, attention_mask=m).logits
            loss = loss_fn(output, t)
            loss.backward()
            opt.step()
            opt.zero_grad(set_to_none=True)
    end_timer_and_print("Default precision")



def eval_loop_autocast(model, batched_dataset):
    model.eval()
    with torch.no_grad():
        acc = torch.Tensor(0.0)
        for input_ids, attn_mask, target in batched_dataset:
            prediction = model(
                input_ids=input_ids, attention_mask=attn_mask
            ).logits.argmax(dim=-1)
            acc += (prediction == target).sum()
        acc /= len(batched_dataset) * batched_dataset.batch_size
    return acc.item()
