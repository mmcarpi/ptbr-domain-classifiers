import argparse
import gc
import os
import time
from contextlib import nullcontext

import torch
from torch.amp import GradScaler

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


def end_timer_and_print(local_msg, local_rank):
    torch.cuda.synchronize()
    end_time = time.time()
    print("\n" + local_msg)
    print("Total execution time = %.3f sec" % (end_time - start_time))
    print(
        "Max memory used by tensors = %d bytes"
        % (torch.cuda.max_memory_allocated(device=local_rank))
    )


def train_loop_autocast(
    local_rank,
    model,
    train_dataloader,
    optimizer,
    scheduler,
    loss_fn,
    num_epochs,
    iters_to_accumulate,
    device_type,
):
    start_timer()
    model.train()
    scaler = GradScaler(device=device_type)
    step = 0
    for epoch in range(num_epochs):  # 0 epochs, this section is for illustration only
        train_dataloader.sampler.set_epoch(epoch)
        # for input_ids, attn_mask, target in dataloader:
        for i, data in enumerate(train_dataloader):
            # Runs the forward pass under ``autocast``.
            accumulation_iteration = (i + 1) % iters_to_accumulate == 0
            context_manager = (
                nullcontext() if accumulation_iteration else model.no_sync()
            )
            with context_manager:
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
                    loss /= iters_to_accumulate
                scaler.scale(loss).backward()
                if accumulation_iteration:
                    scaler.step(optimizer)
                    scaler.update()
                    scheduler.step(step := step + 1)
                    optimizer.zero_grad()

    end_timer_and_print("Automatic precision", local_rank)


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


def main(args):
    #TODO: Rewrite this so we have the same interface as the hyperparameter-search.py
    model_name = args.model_name
    num_labels = 5
    max_length = 256
    batch_size = args.batch_size
    iters_to_accumulate = 1

    num_epochs = 1
    learning_rate = 1e-4
    weight_decay = 0.01
    warm_up_ratio = 0.05
    seed = 50

    torch.manual_seed(seed)

    init_process_group(backend="nccl")
    world_size = int(os.environ["WORLD_SIZE"])
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    print("Hello from", local_rank, rank, world_size)

    dataset = load_dataset("mmcarpi/caroldb-sentences", split="hps")
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)

    dataset = tokenize_dataset(dataset, tokenizer, max_length)
    dataset = dataset.with_format("torch")
    dataset = dataset.train_test_split(test_size=0.1, seed=seed)

    train_dataset = dataset["train"]

    train_sampler = DistributedSampler(train_dataset)

    train_dataloader = DataLoader(
        dataset=train_dataset,
        sampler=train_sampler,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=num_labels
    )
    model = torch.compile(model, disable=True).to(local_rank)
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )

    total_steps = len(train_dataset) // (world_size * batch_size)
    warm_up_steps = int(total_steps * warm_up_ratio)

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: (step / warm_up_steps) if step < warm_up_steps else 1.0,
    )

    loss_fn = torch.nn.CrossEntropyLoss().to(local_rank)

    if rank == 0:
        print(f"\nTraining info:")
        print(f"\t{model_name=}")
        print(f"\t{batch_size=}")
        print(f"\t{len(train_dataset)=}")
        print(f"\t{total_steps=}")
        print(f"\t{warm_up_steps=}")
        print(f"\n")

    print(f"[GPU{rank}] starting training soon...")
    device_type = "cuda" if torch.cuda.is_available() else "cpu"
    train_loop_autocast(
        local_rank,
        model,
        train_dataloader,
        optimizer,
        scheduler,
        loss_fn,
        num_epochs,
        iters_to_accumulate,
        device_type,
    )
    print(f"[GPU{rank}] finished training")
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
    parser = argparse.ArgumentParser()
    parser.add_argument("model_name", type=str)
    parser.add_argument("batch_size", type=int)

    args = parser.parse_args()
    main(args)
