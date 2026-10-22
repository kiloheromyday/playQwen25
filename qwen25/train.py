from dataclasses import dataclass
import math
from pathlib import Path
import random
import time
from typing import Callable, Iterable, Iterator

import torch
from torch import Tensor, nn


@dataclass
class TrainStats:
    step: int
    loss: float
    tokens: int
    tokens_per_second: float = 0.0
    learning_rate: float = 0.0


def train_steps(
    model: nn.Module,
    batches: Iterable[Tensor],
    optimizer: torch.optim.Optimizer,
    steps: int,
    device: torch.device,
    grad_accumulation_steps: int = 1,
    use_bf16: bool = False,
) -> list[TrainStats]:
    model.train()
    stats = []
    iterator = _repeating_iterator(batches)
    optimizer.zero_grad(set_to_none=True)
    for step in range(1, steps + 1):
        total_loss = 0.0
        token_count = 0
        started = time.perf_counter()
        for _ in range(grad_accumulation_steps):
            batch = next(iterator).to(device)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16):
                result = model(batch, labels=batch)
            loss = result["loss"] / grad_accumulation_steps
            loss.backward()
            total_loss += float(result["loss"].detach())
            token_count += batch.numel() - batch.size(0)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        elapsed = max(time.perf_counter() - started, 1e-9)
        stats.append(
            TrainStats(
                step=step,
                loss=total_loss / grad_accumulation_steps,
                tokens=token_count,
                tokens_per_second=token_count / elapsed,
                learning_rate=optimizer.param_groups[0]["lr"],
            )
        )
    return stats


def _repeating_iterator(batches: Iterable[Tensor]) -> Iterator[Tensor]:
    while True:
        yielded = False
        for batch in batches:
            yielded = True
            yield batch
        if not yielded:
            raise ValueError("batches must contain at least one batch")


def train_loader_steps(
    model: nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    steps: int,
    device: torch.device,
    grad_accumulation_steps: int = 1,
    use_bf16: bool = False,
    scheduler=None,
    start_step: int = 0,
    on_step: Callable[[TrainStats], None] | None = None,
) -> list[TrainStats]:
    """Train from a DataLoader without materializing the complete dataset in memory."""
    if steps <= 0:
        return []
    if grad_accumulation_steps <= 0:
        raise ValueError("grad_accumulation_steps must be positive")
    model.train()
    stats: list[TrainStats] = []
    iterator = _repeating_iterator(loader)
    optimizer.zero_grad(set_to_none=True)
    for local_step in range(1, steps + 1):
        total_loss = 0.0
        token_count = 0
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        started = time.perf_counter()
        for _ in range(grad_accumulation_steps):
            batch = next(iterator).to(device)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16):
                result = model(batch, labels=batch)
            (result["loss"] / grad_accumulation_steps).backward()
            total_loss += float(result["loss"].detach())
            token_count += batch.numel() - batch.size(0)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed = max(time.perf_counter() - started, 1e-9)
        stats.append(
            TrainStats(
                step=start_step + local_step,
                loss=total_loss / grad_accumulation_steps,
                tokens=token_count,
                tokens_per_second=token_count / elapsed,
                learning_rate=optimizer.param_groups[0]["lr"],
            )
        )
        if on_step is not None:
            on_step(stats[-1])
    return stats


@torch.no_grad()
def evaluate_loss(model: nn.Module, batches: Iterable[Tensor], device: torch.device, use_bf16: bool = False) -> tuple[float, float]:
    was_training = model.training
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    try:
        for batch in batches:
            batch = batch.to(device)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16):
                loss = model(batch, labels=batch)["loss"]
            tokens = batch.numel() - batch.size(0)
            total_loss += float(loss) * tokens
            total_tokens += tokens
    finally:
        model.train(was_training)
    if total_tokens == 0:
        raise ValueError("evaluation batches contain no target tokens")
    average_loss = total_loss / total_tokens
    return average_loss, math.exp(min(average_loss, 20.0))


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    path: str | Path,
    step: int,
    scheduler=None,
    metadata: dict | None = None,
) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": step,
        "python_random_state": random.getstate(),
        "torch_random_state": torch.get_rng_state(),
        "metadata": metadata or {},
    }
    if scheduler is not None:
        payload["scheduler"] = scheduler.state_dict()
    if torch.cuda.is_available():
        payload["cuda_random_state"] = torch.cuda.get_rng_state_all()
    torch.save(payload, path)


def load_checkpoint(model: nn.Module, optimizer: torch.optim.Optimizer, path: str | Path, scheduler=None) -> int:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    if scheduler is not None and "scheduler" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler"])
    if "python_random_state" in checkpoint:
        random.setstate(checkpoint["python_random_state"])
    if "torch_random_state" in checkpoint:
        torch.set_rng_state(checkpoint["torch_random_state"])
    if torch.cuda.is_available() and "cuda_random_state" in checkpoint:
        torch.cuda.set_rng_state_all(checkpoint["cuda_random_state"])
    return int(checkpoint["step"])
