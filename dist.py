import argparse
import gc
import json
import os
import time

from contextlib import nullcontext
from dataclasses import dataclass, asdict
from pathlib import Path

import torch
from torch.amp import GradScaler

from torch.distributed import (
    init_process_group,
    destroy_process_group,
    all_gather,
    barrier,
)
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

from datasets import load_dataset

from transformers import AutoTokenizer, AutoModelForSequenceClassification


import util

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


@dataclass
class DeviceConfig:
    rank: int
    local_rank: int
    world_size: int
    device_type: str


@dataclass
class Config:
    model_name: int
    num_labels: int
    batch_size: int
    max_length: int

    weight_decay: float
    warm_up_ratio: float
    learning_rate: float

    iters_to_accumulate: int
    save_every_epoch: bool
    save_path: str

    @classmethod
    def read_config(cls, path):
        with open(path, "r") as config_file:
            config = json.load(config_file)
        return cls(**config)

    def save_config(self, path):
        with open(path, "w") as config_file:
            json.dump(asdict(self), config_file, indent=4)


class Trainer:
    def __init__(
        self,
        model,
        optimizer,
        scheduler,
        loss_fn,
        compute_metrics,
        train_config,
        device_config,
    ):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_fn = loss_fn
        self.compute_metrics = compute_metrics
        self.cfg = train_config

        self.rank = device_config.rank
        self.local_rank = device_config.local_rank
        self.world_size = device_config.world_size
        self.device_type = device_config.device_type
        self.dtype = (
            torch.bfloat16
            if self.device_type == "cuda" and torch.cuda.is_bf16_supported()
            else torch.float32
        )

    def train(
        self,
        train_dataloader,
        eval_dataloader,
        num_epochs,
        start_epoch=0,
        evals_per_epoch=1,
    ):
        start_timer()
        self.model.train()
        scaler = GradScaler(device=self.device_type)

        epoch_steps = len(train_dataloader.dataset) // (
            train_dataloader.batch_size * self.world_size
        )

        iters_to_evaluate = epoch_steps // evals_per_epoch

        step = start_epoch * epoch_steps

        for epoch in range(start_epoch, num_epochs):
            train_dataloader.sampler.set_epoch(epoch)
            epoch_loss = torch.tensor(0.0, device=self.local_rank)
            for i, data in enumerate(train_dataloader):
                # Runs the forward pass under ``autocast``.
                eval_iteration = (i + 1) % iters_to_evaluate == 0
                accumulation_iteration = (i + 1) % self.cfg.iters_to_accumulate == 0
                context_manager = (
                    nullcontext() if accumulation_iteration else self.model.no_sync()
                )
                with context_manager:
                    with torch.autocast(device_type=self.device_type, dtype=self.dtype):
                        output = self.model(
                            input_ids=data["input_ids"].to(
                                self.local_rank, non_blocking=True
                            ),
                            attention_mask=data["attention_mask"].to(
                                self.local_rank, non_blocking=True
                            ),
                        ).logits
                        loss = self.loss_fn(
                            output,
                            data["label"].to(self.local_rank, non_blocking=True),
                        )
                        loss /= self.cfg.iters_to_accumulate
                        epoch_loss += loss
                    scaler.scale(loss).backward()
                    if accumulation_iteration:
                        scaler.step(self.optimizer)
                        scaler.update()
                        self.scheduler.step(step := step + 1)
                        self.optimizer.zero_grad()

                if eval_iteration:
                    metrics = self.eval(eval_dataloader)
                    # all_reduce(epoch_loss, op=) Reduce?
                    metrics["loss"] = epoch_loss.item() / (i + 1)
                    if self.rank == 0:
                        self.save_checkpoint(
                            metrics,
                            epoch,
                            step,
                        )
                    barrier()

        # end_timer_and_print("Automatic precision", local_rank)

    def eval(self, eval_dataloader):
        model_state = self.model.training
        self.model.training = False
        with torch.no_grad():
            predictions = []
            labels = []
            for data in eval_dataloader:
                outputs = self.model(
                    input_ids=data["input_ids"].to(self.local_rank, non_blocking=True),
                    attention_mask=data["attention_mask"].to(
                        self.local_rank, non_blocking=True
                    ),
                ).logits
                predictions.append(outputs)
                labels.append(data["label"])

            predictions = torch.concatenate(predictions).to(self.local_rank)
            labels = torch.concatenate(labels).to(self.local_rank)

            all_predictions = [
                torch.zeros_like(predictions, device=self.local_rank)
                for _ in range(self.world_size)
            ]
            all_labels = [
                torch.zeros_like(labels, device=self.local_rank)
                for _ in range(self.world_size)
            ]

            all_gather(all_predictions, predictions, async_op=False)
            all_gather(all_labels, labels, async_op=False)

            all_predictions = torch.concatenate(all_predictions).to("cpu").numpy()
            all_labels = torch.concatenate(all_labels).to("cpu").numpy()

            metrics = self.compute_metrics(all_predictions, all_labels)

        self.model.training = model_state
        return metrics

    def save_checkpoint(self, metrics, epoch, step):
        checkpoint = {
            "model_state": self.model.state_dict(),
            # "optimizer_state": self.optimizer.state_dict(),
            # "scheduler_state": self.scheduler.state_dict(),
            # "epoch": epoch,
            # "step": step,
        }

        base_path = Path(self.cfg.save_path)
        model_path = base_path / Path(self.cfg.model_name).name
        model_path.mkdir(parents=True, exist_ok=True)
        save_path = model_path / f"checkpoint-{epoch}-{step}.pth"

        torch.save(checkpoint, save_path)

        with open(model_path / f"metrics-{epoch}-{step}.json", "w") as metric_file:
            json.dump(metrics, metric_file, indent=True)


def main():
    torch.manual_seed(args.seed)

    init_process_group(backend="nccl")

    device_config = DeviceConfig(
        rank=int(os.environ["RANK"]),
        local_rank=int(os.environ["LOCAL_RANK"]),
        world_size=int(os.environ["WORLD_SIZE"]),
        device_type="cuda" if torch.cuda.is_available() else "cpu",
    )

    train_config = Config.read_config(args.config_file)

    dataset = load_dataset("mmcarpi/caroldb-sentences", split="train").select(
        range(1000)
    )
    tokenizer = AutoTokenizer.from_pretrained(train_config.model_name, use_fast=False)

    dataset = util.tokenize_dataset(dataset, tokenizer, train_config.max_length)
    dataset = dataset.with_format("torch")

    temp_dataset = dataset.train_test_split(test_size=0.2)

    train_dataset = temp_dataset["train"]
    train_sampler = DistributedSampler(train_dataset)

    train_dataloader = DataLoader(
        dataset=train_dataset,
        sampler=train_sampler,
        batch_size=train_config.batch_size,
        shuffle=False,
        pin_memory=True,
    )

    eval_dataset = temp_dataset["test"]
    eval_sampler = DistributedSampler(eval_dataset)

    eval_dataloader = DataLoader(
        dataset=eval_dataset,
        sampler=eval_sampler,
        batch_size=train_config.batch_size,
        shuffle=False,
        pin_memory=True,
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        train_config.model_name, num_labels=train_config.num_labels
    ).to(device_config.local_rank)

    if args.load_checkpoint:
        checkpoint = torch.load(args.load_checkpoint)
        model.state_dict.load(checkpoint["model_state"])

    # model = torch.compile(model, disable=args.disable).to(local_rank)
    model = DDP(
        model,
        device_ids=[device_config.local_rank],
        output_device=device_config.local_rank,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_config.learning_rate,
        weight_decay=train_config.weight_decay,
    )

    total_steps = (
        args.num_epochs
        * len(train_dataset)
        // (device_config.world_size * train_config.batch_size)
    )
    warm_up_steps = int(total_steps * train_config.warm_up_ratio)

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: (step / warm_up_steps) if step < warm_up_steps else 1.0,
    )

    loss_fn = torch.nn.CrossEntropyLoss().to(device_config.local_rank)

    compute_metrics = util.create_compute_metrics(
        train_config.num_labels, argmax_first=True
    )

    if device_config.rank == 0:
        print("Training Info:")
        for k, v in asdict(train_config).items():
            print(f"{k}={v}")

        if args.load_checkpoint:
            print("Starting from:")

    print(f"[GPU{device_config.rank}] starting training soon...")
    trainer = Trainer(
        model,
        optimizer,
        scheduler,
        loss_fn,
        compute_metrics,
        train_config,
        device_config,
    )

    trainer.train(
        train_dataloader,
        eval_dataloader,
        args.num_epochs,
        start_epoch=args.start_epoch,
        evals_per_epoch=args.evals_per_epoch,
    )
    print(f"[GPU{device_config.rank}] finished training")
    destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", type=str)
    parser.add_argument("num_epochs", type=int)
    parser.add_argument("evals_per_epoch", type=int)

    parser.add_argument("--load-checkpoint", type=str, default="")
    parser.add_argument("--start_epoch", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    if args.start_epoch and not args.load_checkpoint:
        parser.error("Argument --load-checkpoint is required to change epoch start")

    main()
