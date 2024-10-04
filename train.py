import os

from dataclasses import dataclass

import torch
import torch.multiprocessing as mp
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from transformers import AutoTokenizer, AutoModelForSequenceClassification

import data


@dataclass
class Config:
    model_name: str
    dataset_path: str
    max_length: int
    learning_rate: float
    weight_decay: float
    warm_up_ratio: float
    num_epochs: int
    batch_size: int
    save_every: int = 0


class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        train_data: DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        gpu_id: int,
        save_every: int,
    ):
        self.gpu_id = gpu_id
        self.model = model.to(gpu_id)
        self.train_data = train_data
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.save_evey = save_every
        self.model = DDP(model, device_ids=[gpu_id])

    def _run_batch(self, input_ids, attention_masks, targets, i, epoch):
        self.optimizer.zero_grad()
        output = self.model(input_ids=input_ids, attention_mask=attention_masks)
        loss = F.cross_entropy(output, targets)
        loss.backward()
        self.optmizer.step()
        self.scheduler.step(i * epoch)

    def _run_epoch(self, epoch: int):
        b_sz = len(next(iter(self.train_data))[0])
        print(
            f"[GPU{self.gpu_id}] Epoch {epoch} | Batchsize: {b_sz} | Steps: {len(self.train_data)}"
        )
        self.train_data.sampler.set_epoch(epoch)
        for i, (input_ids, attetion_masks, targets) in enumerate(self.train_data):
            input_ids = input_ids.to(self.gpu_id)
            attention_masks = attetion_masks.to(self.gpu_id)
            targets = targets.to(self.gpu_id)
            self._run_batch(input_ids, attention_masks, targets, i, epoch)

    def _save_checkpoint(self, epoch: int):
        ckp = self.model.module.state_dict()
        PATH = "checkpoint.pt"
        torch.save(ckp, PATH)
        print(f"Epoch {epoch} | Training checkpoint saved at {PATH}")

    def train(self, num_epochs: int):
        for epoch in range(num_epochs):
            self._run_epoch(epoch)
            if self.gpu_id == 0 and epoch % self.save_every == 0:
                self._save_checkpoint(epoch)


def init_model(model_name):
    model = AutoModelForSequenceClassification.from_pretrained(model_name).to(
        dtype=torch.bfloat16
    )
    return model


def load_train_objs(config: Config):
    model = init_model(config.model_name)

    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    train_set = data.tokenize(config.dataset_path, tokenizer=tokenizer)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )

    total_steps = config.num_epochs * len(train_set) // config.batch_size
    warm_up_steps = int(total_steps * config.warm_up_ratio)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: ((step + 1) / total_steps if step < warm_up_steps else 1.0),
    )
    return train_set, model, optimizer, scheduler


def prepare_dataloader(
    dataset: Dataset, pad_value: int, max_length: int, batch_size: int
):
    def pad(x, value):
        return torch.nn.functional.pad(x, (0, max_length - x.shape[0]), value=value)

    def collate_fn(data):
        x, y = zip(*data)

        x = [torch.tensor(xi) for xi in x]
        m = [torch.ones(xi.shape) for xi in x]

        x = torch.stack([pad(xi, pad_value) for xi in x])
        m = torch.stack([pad(mi, 0) for mi in m])

        y = torch.tensor(y, dtype=torch.int32).reshape(-1, 1)
        return x, m, y

    # collate_fn_opt = torch.compile(collate_fn)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        # num_workers=16,
        # prefetch_factor=512,
        # persistent_workers=False,
        sampler=DistributedSampler(dataset),
    )
    return dataloader


def ddp_setup(rank: int, world_size: int):
    """
    Args:
        rank: Unique identifier of each process
        world_size: Total number of processes
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    torch.cuda.set_device(rank)
    init_process_group(backend="nccl", rank=rank, world_size=world_size)


def main(
    rank: int,
    world_size: int,
    config: Config,
):
    ddp_setup(rank, world_size)
    dataset, model, optimizer, scheduler = load_train_objs(config)
    train_data = prepare_dataloader(dataset, config.max_length, config.batch_size)
    trainer = Trainer(model, train_data, optimizer, scheduler, rank, config.save_every)
    trainer.train(config.num_epochs)
    destroy_process_group()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Simple distributed training job")
    parser.add_argument("model_name", type=str)
    parser.add_argument("dataset_path", type=str)
    parser.add_argument("max_length", type=int)
    parser.add_argument("learning_rate", type=float)
    parser.add_argument("weight_decay", type=float)
    parser.add_argument("warm_up_ratio", type=float)
    parser.add_argument("num_epochs", type=int)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--save_every", type=int, default=0)

    args = parser.parse_args()
    config = Config(
        args.model_name,
        args.dataset_path,
        args.max_length,
        args.learning_rate,
        args.weight_decay,
        args.warm_up_ratio,
        args.num_epochs,
        args.batch_size,
        args.save_every,
    )
    world_size = torch.cuda.device_count()
    mp.spawn(
        main,
        args=(world_size, config),
        nprocs=world_size,
    )
