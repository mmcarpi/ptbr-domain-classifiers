import argparse
import json
import os

from contextlib import nullcontext
from dataclasses import dataclass, asdict
from pathlib import Path

import torch
from torch.amp import GradScaler

from torch.distributed import (
    init_process_group,
    destroy_process_group,
    all_gather,
    all_reduce,
    barrier,
)
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

from datasets import load_dataset

from transformers import AutoTokenizer, AutoModelForSequenceClassification


import util


@dataclass
class DeviceConfig:
    rank: int
    local_rank: int
    world_size: int
    device_type: str
    num_gpus_per_node: int


@dataclass
class Config:
    model_name: int
    num_labels: int
    batch_size: int
    max_length: int

    weight_decay: float
    warm_up_ratio: float
    learning_rate: float

    save_path: str

    @classmethod
    def read_config(cls, path):
        with open(path, "r") as config_file:
            config = json.load(config_file)
        return cls(**config)

    def save_config(self, path):
        with open(path, "w") as config_file:
            json.dump(asdict(self), config_file, indent=4)


class AverageMeter:
    def __init__(self, local_rank):
        self.local_rank = local_rank
        self.reset()

    def __str__(self):
        return f"{self.avg:.4f}"

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.cnt = 0.0

    def update(self, val, n=1):
        self.val = val
        self.sum += n * val
        self.cnt += n
        self.avg = self.sum / self.cnt

    def all_reduce(self):
        total = torch.tensor(
            [self.sum, self.cnt], dtype=torch.float32, device=self.local_rank
        )
        all_reduce(total, async_op=False)
        self.sum, self.cnt = total.tolist()
        self.avg = self.sum / self.cnt


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
        self.num_gpus_per_node = device_config.num_gpus_per_node
        self.is_distributed = device_config.world_size > 1
        self.is_cuda = self.device_type == "cuda"
        self.dtype = (
            torch.bfloat16
            if self.is_cuda and torch.cuda.is_bf16_supported()
            else torch.float32
        )

    def train(
        self,
        train_dataloader,
        eval_dataloader,
        num_epochs,
        start_epoch=0,
        evals_per_epoch=1,
        accumulate_frequency=1,
        log_frequency=1,
    ):
        self.model.train()
        scaler = GradScaler(device=self.device_type)

        epoch_steps = len(train_dataloader.dataset) // self.cfg.batch_size
        total_steps = num_epochs * epoch_steps

        evaluate_frequency = epoch_steps // evals_per_epoch

        step = start_epoch * epoch_steps
        for epoch in range(start_epoch, num_epochs):
            train_dataloader.sampler.set_epoch(epoch)
            epoch_eval = 0
            epoch_loss = AverageMeter(self.local_rank)
            for i, (input, label) in enumerate(train_dataloader, start=1):
                accumulation_iteration = i % accumulate_frequency == 0
                eval_iteration = i % evaluate_frequency == 0
                log_iteration = step % log_frequency == 0
                context_manager = (
                    nullcontext() if accumulation_iteration else self.model.no_sync()
                )
                with context_manager:
                    with torch.autocast(device_type=self.device_type, dtype=self.dtype):
                        output = self.model(**input).logits
                        loss = self.loss_fn(
                            output, label.to(self.local_rank, non_blocking=True)
                        )
                        loss /= accumulate_frequency
                        epoch_loss.update(loss.item(), input["input_ids"].size(0))
                    scaler.scale(loss).backward()
                    if accumulation_iteration:
                        scaler.step(self.optimizer)
                        scaler.update()
                        self.scheduler.step(step := step + 1)
                        self.optimizer.zero_grad()

                if log_iteration and self.rank == 0:
                    print(
                        f"[GPU{self.rank}] epoch {epoch+1}/{num_epochs} | step {step}/{total_steps} | loss {epoch_loss}"
                    )

                if eval_iteration:
                    epoch_eval += 1
                    metrics = self.eval(eval_dataloader)
                    metrics["loss"] = epoch_loss.avg
                    barrier()
                    if self.rank == 0:
                        self.save_checkpoint(
                            metrics,
                            epoch,
                            epoch_eval,
                        )
                    barrier()
                    self.model.train()

    def eval(self, eval_dataloader):
        self.model.eval()
        with torch.no_grad():
            predictions = []
            labels = []
            for input, label in eval_dataloader:
                outputs = self.model(**input).logits
                predictions.append(outputs)
                labels.append(label)

            predictions = torch.concatenate(predictions).to(
                self.local_rank, non_blocking=True
            )
            labels = torch.concatenate(labels).to(self.local_rank, non_blocking=True)

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

        return metrics

    def save_checkpoint(self, metrics, epoch, epoch_eval):
        base_path = Path(self.cfg.save_path)
        model_path = base_path / Path(self.cfg.model_name).name
        model_path.mkdir(parents=True, exist_ok=True)
        save_path = model_path / f"checkpoint-{epoch}-{epoch_eval}.pth"

        if isinstance(self.model, DDP):
            torch.save(self.model.module.state_dict(), save_path)
        else:
            torch.save(self.model.state_dict(), save_path)

        with open(
            model_path / f"metrics-{epoch}-{epoch_eval}.json", "w"
        ) as metric_file:
            json.dump(metrics, metric_file, indent=True)


def main():
    torch.manual_seed(args.seed)

    init_process_group(backend="nccl")

    device_config = DeviceConfig(
        rank=int(os.environ["RANK"]),
        local_rank=int(os.environ["LOCAL_RANK"]),
        world_size=int(os.environ["WORLD_SIZE"]),
        device_type="cuda" if torch.cuda.is_available() else "cpu",
        num_gpus_per_node=torch.cuda.device_count() if torch.cuda.is_available() else 1,
    )

    train_config = Config.read_config(args.config_file)

    dataset = load_dataset("mmcarpi/caroldb-sentences", split="train")
    tokenizer = AutoTokenizer.from_pretrained(train_config.model_name, use_fast=False)

    dataset = util.tokenize_dataset(dataset, tokenizer, train_config.max_length)
    dataset = dataset.with_format("torch")

    temp_dataset = dataset.train_test_split(test_size=0.01)

    train_dataset = temp_dataset["train"]
    eval_dataset = temp_dataset["test"]

    batch_size = train_config.batch_size // device_config.num_gpus_per_node

    train_sampler = DistributedSampler(train_dataset)
    eval_sampler = DistributedSampler(eval_dataset, drop_last=True)

    def collate_fn(data):
        input_ids = torch.stack([item["input_ids"] for item in data])
        attention_mask = torch.stack([item["attention_mask"] for item in data])
        label = torch.stack([item["label"] for item in data])

        input = dict(input_ids=input_ids, attention_mask=attention_mask)
        return input, label

    train_dataloader = DataLoader(
        dataset=train_dataset,
        sampler=train_sampler,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=args.num_workers,
        drop_last=True,
        collate_fn=collate_fn,
    )
    eval_dataloader = DataLoader(
        dataset=eval_dataset,
        sampler=eval_sampler,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=args.num_workers,
        drop_last=True,
        collate_fn=collate_fn,
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        train_config.model_name, num_labels=train_config.num_labels
    ).to(device_config.local_rank)

    if args.load_checkpoint:
        state = torch.load(args.load_checkpoint, weights_only=True)
        model.load_state_dict(state)

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

    total_steps = args.num_epochs * (len(train_dataset) // train_config.batch_size)
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
            print(f"{k!r}: {v!r}")

        if args.load_checkpoint:
            print("Loaded checkpoint:", args.load_checkpoint)
    barrier()

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
        accumulate_frequency=args.accumulate_frequency,
        log_frequency=args.log_frequency,
    )
    print(f"[GPU{device_config.rank}] finished training")
    destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", type=str)
    parser.add_argument("num_epochs", type=int)
    parser.add_argument("evals_per_epoch", type=int)

    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--log_frequency", type=int, default=1)
    parser.add_argument("--accumulate_frequency", type=int, default=1)
    parser.add_argument("--load_checkpoint", type=str, default="")
    parser.add_argument("--start_epoch", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    if args.start_epoch and not args.load_checkpoint:
        parser.error("Argument --load-checkpoint is required to change epoch start")

    main()
