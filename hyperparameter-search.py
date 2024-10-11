import argparse
import os
from contextlib import nullcontext
from functools import partial

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification

import optuna
from optuna.trial import TrialState

import torch
import torch.distributed as dist
from torch.amp import GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from data import tokenize_dataset


# TODO: Refactor this function to make local_rank, epoch, step, iters_to_accumulate, and device_type keyword arguments
# TODO: After the refactor, put the new version on dist.py and import it here
def train_loop_autocast(
    local_rank,
    model,
    train_dataloader,
    optimizer,
    scheduler,
    loss_fn,
    epoch,
    step,
    iters_to_accumulate,
    device_type,
):
    model.train()
    scaler = GradScaler()
    train_dataloader.sampler.set_epoch(epoch)
    # for input_ids, attn_mask, target in dataloader:
    for i, data in enumerate(train_dataloader):
        # Runs the forward pass under ``autocast``.
        accumulation_iteration = (i + 1) % iters_to_accumulate == 0
        context_manager = nullcontext() if accumulation_iteration else model.no_sync()
        with context_manager:
            with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                output = model(
                    input_ids=data["input_ids"].to(local_rank, non_blocking=True),
                    attention_mask=data["attention_mask"].to(
                        local_rank, non_blocking=True
                    ),
                ).logits
                loss = loss_fn(output, data["label"].to(local_rank, non_blocking=True))
                loss /= iters_to_accumulate
            scaler.scale(loss).backward()
            if accumulation_iteration:
                scaler.step(optimizer)
                scaler.update()
                scheduler.step(step := step + 1)
                optimizer.zero_grad()
    return step


# TODO: Apply the same refactor of the previous function
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
        dist.all_reduce(acc, dist.ReduceOp.SUM, async_op=False)
        acc /= len(test_dataloader.dataset)
    return acc.item()


def objective(single_trial, **kwargs):
    trial = optuna.integration.TorchDistributedTrial(single_trial)

    rank = kwargs["rank"]
    local_rank = kwargs["local_rank"]
    world_size = kwargs["world_size"]

    model_name = kwargs["model_name"]
    dataset_name = kwargs["dataset_name"]
    num_labels = kwargs["num_labels"]
    max_length = kwargs["max_length"]

    batch_size = kwargs["batch_size"]
    num_epochs = kwargs["num_epochs"]

    torch_seed = kwargs["torch_seed"]
    disable_compile = kwargs["disable_compile"]
    iters_to_accumulate = kwargs["iters_to_accumulate"]

    learning_rate = trial.suggest_float("learning_rate", 2e-5, 1e-4, log=True)
    warm_up_ratio = trial.suggest_float("warm_up_ratio", 0.0, 0.2, step=0.05)
    weight_decay = trial.suggest_float("weight_decay", 0.0, 0.1, step=0.01)

    torch.manual_seed(torch_seed)

    dataset = load_dataset(dataset_name, split="hps")
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)

    dataset = tokenize_dataset(dataset, tokenizer, max_length)
    dataset = dataset.with_format("torch")
    dataset = dataset.train_test_split(test_size=0.1, seed=torch_seed)

    train_dataset = dataset["train"]
    train_sampler = DistributedSampler(train_dataset)
    train_dataloader = DataLoader(
        dataset=train_dataset,
        sampler=train_sampler,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
    )

    test_dataset = dataset["test"]
    test_sampler = DistributedSampler(test_dataset)
    test_dataloader = DataLoader(
        test_dataset,
        sampler=test_sampler,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=False,
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=num_labels
    )
    model = torch.compile(model, disable=disable_compile).to(local_rank)
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )

    total_steps = len(train_dataset) // (world_size * batch_size)
    warm_up_steps = int(total_steps * warm_up_ratio)
    if rank == 0:
        print(f"{total_steps=} {warm_up_steps=}")

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: (step / warm_up_steps) if step < warm_up_steps else 1.0,
    )

    loss_fn = torch.nn.CrossEntropyLoss().to(local_rank)

    device_type = "cuda" if torch.cuda.is_available() else "cpu"

    step = 0
    accuracy = 0
    for epoch in range(num_epochs):
        step = train_loop_autocast(
            local_rank,
            model,
            train_dataloader,
            optimizer,
            scheduler,
            loss_fn,
            epoch,
            step,
            iters_to_accumulate,
            device_type,
        )

        accuracy = eval_loop_autocast(
            local_rank, model, test_dataloader, device_type=device_type
        )

        trial.report(accuracy, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return accuracy


def main(args):
    # Set environmental variables required by torch.distributed.

    rank = int(os.environ.get("RANK"))
    local_rank = int(os.environ.get("LOCAL_RANK"))
    world_size = int(os.environ.get("WORLD_SIZE"))

    objective_args = dict(
        rank=rank,
        local_rank=local_rank,
        world_size=world_size,
        model_name=args.model_name,
        dataset_name=args.dataset_name,
        num_labels=args.num_labels,
        max_length=args.max_length,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        torch_seed=args.torch_seed,
        disable_compile=args.disable_compile,
        iters_to_accumulate=args.iters_to_accumulate,
    )

    dist.init_process_group("nccl")
    if rank == 0:
        # Download dataset before starting the optimization.
        # TODO: Check if this is necessary in the environment
        load_dataset(args.dataset_name)
    dist.barrier()

    study = None
    if rank == 0:
        name = args.model_name.split("/")[-1] + "-hp-search"
        study = optuna.create_study(
            study_name=name,
            storage=f"sqlite:///results/{name}.db",
            direction="maximize",
            load_if_exists=True,
        )
        study.optimize(partial(objective, **objective_args), n_trials=args.num_trials)
    else:
        for _ in range(args.num_trials):
            try:
                objective(None, **objective_args)
            except optuna.TrialPruned:
                pass

    if rank == 0:
        assert study is not None
        pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
        complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

        print("Study statistics: ")
        print("  Number of finished trials: ", len(study.trials))
        print("  Number of pruned trials: ", len(pruned_trials))
        print("  Number of complete trials: ", len(complete_trials))

        print("Best trial:")
        trial = study.best_trial

        print("  Value: ", trial.value)

        print("  Params: ")
        for key, value in trial.params.items():
            print("    {}: {}".format(key, value))

    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model_name", type=str)
    parser.add_argument("dataset_name", type=str)
    parser.add_argument("num_labels", type=int)
    parser.add_argument("max_length", type=int)

    parser.add_argument("batch_size", type=int)
    parser.add_argument("num_epochs", type=int)

    parser.add_argument("num_trials", type=int, default=50)

    parser.add_argument("--torch_seed", type=int, default=42)
    parser.add_argument("--disable_compile", action="store_true", default=False)
    parser.add_argument("--iters_to_accumulate", type=int, default=1)

    args = parser.parse_args()
    main(args)
