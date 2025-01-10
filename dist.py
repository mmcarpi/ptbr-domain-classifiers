import argparse
import multiprocessing
import os

from contextlib import nullcontext
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
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification

import util
from config import DeviceConfig, ModelConfig


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
        model_config,
        device_config,
        save_path,
    ):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_fn = loss_fn
        self.compute_metrics = compute_metrics
        self.cfg = model_config

        self.model_path = Path(save_path) / Path(self.cfg.model_name)

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
        self.model_path.mkdir(parents=True, exist_ok=True)
        self.model.train()
        scaler = GradScaler(device=self.device_type)

        epoch_steps = len(train_dataloader.dataset) // self.cfg.batch_size
        total_steps = (start_epoch + num_epochs) * epoch_steps

        evaluate_frequency = epoch_steps // evals_per_epoch

        step = start_epoch * epoch_steps
        for epoch in range(start_epoch, start_epoch + num_epochs):
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
                        step += 1
                        scaler.step(self.optimizer)
                        scaler.update()
                        self.scheduler.step()
                        self.optimizer.zero_grad()

                if log_iteration and self.rank == 0:
                    print(
                        (
                            f"[GPU{self.rank}] epoch {epoch+1}/{start_epoch+num_epochs} |"
                            f"step {step}/{total_steps} |"
                            f"loss {epoch_loss} |"
                            f"learning_rate {self.scheduler.get_last_lr()}"
                        )
                    )

                if eval_iteration:
                    epoch_eval += 1
                    metrics = self.eval(eval_dataloader)
                    metrics["loss"] = epoch_loss.avg
                    barrier()
                    if self.rank == 0:
                        util.save_metrics(
                            metrics,
                            self.model_path / f"metrics-{epoch}-{epoch_eval}.json",
                        )
                    barrier()
                    self.model.train()
            self.save_checkpoint(epoch)

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

    def save_checkpoint(self, epoch):
        checkpoint_path = self.model_path / "checkpoint"
        checkpoint_path.mkdir(exist_ok=True)

        if isinstance(self.model, DDP):
            model_state_dict = self.model.module.state_dict()
        else:
            model_state_dict = self.model.state_dict()

        torch.save(
            {
                "epoch": epoch,
                "state_dict": model_state_dict,
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict(),
            },
            checkpoint_path / f"checkpoint-{epoch}.pth",
        )


class CustomDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
        self._create_labels(labels)

    def _create_labels(self, labels):
        unique_labels = sorted(set(labels))
        self.encoder = {label: i for i, label in enumerate(unique_labels)}
        self.labels = [self.encoder[label] for label in labels]

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
        }, label


def create_dataloader(
    texts,
    labels,
    tokenizer_name,
    max_length,
    batch_size,
    num_workers=0,
    is_distributed=False,
):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    dataset = CustomDataset(texts, labels, tokenizer, max_length)

    if is_distributed:
        sampler = DistributedSampler(dataset)
        shuffle = False  # shuffle must be false when using DistributedSampler
    else:
        sampler = None
        shuffle = True

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        sampler=sampler,
        pin_memory=True,
        multiprocessing_context="fork"
        if multiprocessing.get_start_method(allow_none=True) != "spawn"
        else "spawn",
    )
    return dataloader


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

    model_config = ModelConfig.load_config(args.model_config_file)

    dataset = load_dataset("mmcarpi/caroldb-sentences", split="train")

    temp_dataset = dataset.train_test_split(test_size=0.01)

    train_dataset = temp_dataset["train"].to_dict()
    eval_dataset = temp_dataset["test"].to_dict()

    batch_size = model_config.batch_size // device_config.num_gpus_per_node

    train_dataloader = create_dataloader(
        train_dataset["text"],
        train_dataset["domain"],
        model_config.model_name,
        model_config.max_length,
        batch_size,
        num_workers=args.num_workers,
        is_distributed=device_config.world_size > 1,
    )

    eval_dataloader = create_dataloader(
        eval_dataset["text"],
        eval_dataset["domain"],
        model_config.model_name,
        model_config.max_length,
        batch_size,
        num_workers=args.num_workers,
        is_distributed=device_config.world_size > 1,
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        model_config.model_name, num_labels=model_config.num_labels
    ).to(device_config.local_rank)

    # model = torch.compile(model, disable=args.disable).to(local_rank)

    model = DDP(
        model,
        device_ids=[device_config.local_rank],
        output_device=device_config.local_rank,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=model_config.learning_rate,
        weight_decay=model_config.weight_decay,
    )

    total_steps = args.num_epochs * (len(train_dataset) // model_config.batch_size)
    warm_up_steps = int(total_steps * model_config.warm_up_ratio)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: (step / warm_up_steps) if step < warm_up_steps else 1.0,
    )

    start_epoch = 0
    if args.resume:
        checkpoint = torch.load(args.resume, weights_only=False)
        model.module.load_state_dict(checkpoint["state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = checkpoint["epoch"] + 1

    loss_fn = torch.nn.CrossEntropyLoss().to(device_config.local_rank)

    compute_metrics = util.create_compute_metrics(
        model_config.num_labels, argmax_first=True
    )

    if device_config.rank == 0:
        print("Training Info:")
        print(model_config)
        if args.resume:
            print("Loaded checkpoint:", args.resume)
    barrier()

    print(f"[GPU{device_config.rank}] starting training soon...")
    trainer = Trainer(
        model,
        optimizer,
        scheduler,
        loss_fn,
        compute_metrics,
        model_config,
        device_config,
        args.save_path,
    )

    trainer.train(
        train_dataloader,
        eval_dataloader,
        args.num_epochs,
        start_epoch=start_epoch,
        evals_per_epoch=args.evals_per_epoch,
        accumulate_frequency=args.accumulate_frequency,
        log_frequency=args.log_frequency,
    )
    print(f"[GPU{device_config.rank}] finished training")
    destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model_config_file", type=str)
    parser.add_argument("num_epochs", type=int)
    parser.add_argument("evals_per_epoch", type=int)

    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--log_frequency", type=int, default=1)
    parser.add_argument("--accumulate_frequency", type=int, default=1)
    parser.add_argument("--resume", type=str, default="")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_path", type=str, default="./output")

    args = parser.parse_args()

    main()
