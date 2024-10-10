import argparse
import gc
import os
import time
from contextlib import nullcontext

import torch

from torch.distributed import (
    init_process_group,
    destroy_process_group,
    all_reduce,
    ReduceOp,
)
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

from datasets import load_dataset

import optuna
from optuna.trial import TrialState

from transformers import AutoTokenizer, AutoModelForSequenceClassification


from data import tokenize_dataset

start_time = None


def start_timer():
    global start_time
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    start_time = time.time()


def end_timer_and_print(local_msg):
    torch.cuda.synchronize()
    end_time = time.time()
    print("\n" + local_msg)
    print("Total execution time = %.3f sec" % (end_time - start_time))
    print("Max memory used by tensors = %d bytes" % (torch.cuda.max_memory_allocated()))


def train_loop_autocast(
    local_rank,
    model,
    train_dataloader,
    optimizer,
    loss_fn,
    num_epochs,
    gradient_accumulation_steps,
    device_type,
):
    start_timer()
    model.train()
    for epoch in range(num_epochs):  # 0 epochs, this section is for illustration only
        train_dataloader.sampler.set_epoch(epoch)
        # for input_ids, attn_mask, target in dataloader:
        for i, data in enumerate(train_dataloader):
            # Runs the forward pass under ``autocast``.
            with (
                nullcontext()
                if i % gradient_accumulation_steps == 0
                else model.no_sync()
            ):
                with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                    output = model(
                        input_ids=data["input_ids"].to(local_rank, non_blocking=True),
                        attention_mask=data["attention_mask"].to(
                            local_rank, non_blocking=True
                        ),
                    ).logits
                    loss = loss_fn(
                        output, data["label"].to(local_rank, non_blocking=True)
                    )

                loss.backward()
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

    end_timer_and_print("Automatic precision")


def eval_loop_autocast(local_rank, model, test_dataloader, device_type):
    model.eval()
    with torch.no_grad(), torch.autocast(device_type=device_type, dtype=torch.bfloat16):
        acc = torch.tensor(0.0, device=local_rank)
        for data in test_dataloader:
            prediction = model(
                input_ids=data["input_ids"].to(local_rank, non_blocking=True),
                attention_mask=data["attention_mask"].to(local_rank, non_blocking=True),
            ).logits.argmax(dim=-1)
            acc += (prediction == data["label"].to(local_rank, non_blocking=True)).sum()
        all_reduce(acc, ReduceOp.SUM, async_op=False)
        acc /= len(test_dataloader.dataset)
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

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=num_labels
    )
    model = torch.compile(model, disable=True).to(local_rank)
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )

    dataset = load_dataset("mmcarpi/caroldb-sentences", split="hps").select(
        range(1_000)
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)

    dataset = tokenize_dataset(dataset, tokenizer, max_length)
    # dataset = dataset.with_format("torch", device=local_rank)# TODO: Try using pin_memory
    dataset = dataset.with_format("torch")  # TODO: Try using pin_memory
    dataset = dataset.train_test_split(test_size=0.1)

    train_dataset = dataset["train"]

    train_sampler = DistributedSampler(train_dataset)

    train_dataloader = DataLoader(
        dataset=train_dataset,
        sampler=train_sampler,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
    )

    loss_fn = torch.nn.CrossEntropyLoss().to(local_rank)

    print(f"[GPU{local_rank}] starting training soon...")
    device_type = "cuda" if torch.cuda.is_available() else "cpu"
    train_loop_autocast(
        local_rank,
        model,
        train_dataloader,
        optimizer,
        loss_fn,
        num_epochs,
        gradient_accumulation_steps,
        device_type,
    )
    print(f"[GPU{local_rank}] finished training")
    test_dataset = dataset["test"]
    test_sampler = DistributedSampler(test_dataset)
    test_dataloader = DataLoader(
        test_dataset,
        sampler=test_sampler,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=False,
    )
    acc = eval_loop_autocast(
        local_rank, model, test_dataloader, device_type=device_type
    )

    if rank == 0:
        print("accuracy on test_dataset: %.3f" % acc)

    destroy_process_group()


if __name__ == "__main__":
    main()

# model_name = "neuralmind/bert-base-portuguese-cased"
# model_name = "PORTULAN/albertina-900m-portuguese-ptbr-encoder"
# model_name = "PORTULAN/albertina-1b5-portuguese-ptbr-encoder-256"
