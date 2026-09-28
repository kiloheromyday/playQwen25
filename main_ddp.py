import argparse
import os
from pathlib import Path
import time

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

from qwen25.config import QwenConfig
from qwen25.data import build_data_bundle, load_text_documents, split_documents
from qwen25.model import QwenForCausalLM
from qwen25.tokenizer import SimpleCharTokenizer
from qwen25.train import save_checkpoint


def setup_process_group():
    launched_by_torchrun = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    if not launched_by_torchrun:
        return 0, 1, 0, False
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(backend=backend, init_method="env://")
    return rank, world_size, local_rank, True


def parse_args():
    parser = argparse.ArgumentParser(description="DDP training smoke test for the tiny Qwen2.5-style model.")
    parser.add_argument("--data", default="data/tiny_corpus")
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--sequence-length", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--checkpoint", default="artifacts/ddp_checkpoint.pt")
    return parser.parse_args()


def run_local_check(args):
    config = QwenConfig.from_json("configs/tiny.json")
    model = QwenForCausalLM(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    ids = torch.randint(0, config.vocab_size, (args.batch_size, args.sequence_length))
    for _ in range(args.steps):
        optimizer.zero_grad(set_to_none=True)
        loss = model(ids, labels=ids)["loss"]
        loss.backward()
        optimizer.step()
    print("local DDP-compatible optimizer check: PASS")
    print("For real multi-GPU synchronization, run this file with torchrun on Linux.")
    print("Example: torchrun --standalone --nproc-per-node=GPU数量 main_ddp.py")


def run_distributed(args, rank: int, world_size: int, local_rank: int):
    use_cuda = torch.cuda.is_available()
    device = torch.device(f"cuda:{local_rank}" if use_cuda else "cpu")
    if use_cuda:
        torch.cuda.set_device(local_rank)

    documents = load_text_documents(args.data)
    train_documents, _ = split_documents(documents, valid_ratio=0.25, seed=0)
    tokenizer = SimpleCharTokenizer.from_texts(train_documents)
    bundle = build_data_bundle(
        documents,
        tokenizer,
        sequence_length=args.sequence_length,
        valid_ratio=0.25,
        seed=0,
    )
    sampler = DistributedSampler(
        bundle.train_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        seed=0,
        drop_last=False,
    )
    loader = DataLoader(bundle.train_dataset, batch_size=args.batch_size, sampler=sampler, drop_last=False)

    config = QwenConfig.from_json("configs/tiny.json")
    config.vocab_size = tokenizer.vocab_size
    model = QwenForCausalLM(config).to(device)
    wrapped = DDP(model, device_ids=[local_rank] if use_cuda else None)
    optimizer = torch.optim.AdamW(wrapped.parameters(), lr=args.lr)

    iterator = iter(loader)
    local_losses = []
    started = time.perf_counter()
    for step in range(1, args.steps + 1):
        try:
            batch = next(iterator)
        except StopIteration:
            sampler.set_epoch(step)
            iterator = iter(loader)
            batch = next(iterator)
        batch = batch.to(device)
        optimizer.zero_grad(set_to_none=True)
        use_bf16 = use_cuda and torch.cuda.is_bf16_supported()
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16):
            loss = wrapped(batch, labels=batch)["loss"]
        loss.backward()
        torch.nn.utils.clip_grad_norm_(wrapped.parameters(), 1.0)
        optimizer.step()
        local_losses.append(float(loss.detach()))

    elapsed = max(time.perf_counter() - started, 1e-9)
    loss_tensor = torch.tensor(sum(local_losses) / len(local_losses), device=device)
    token_tensor = torch.tensor(
        args.steps * (args.batch_size * (args.sequence_length - 1)),
        device=device,
        dtype=torch.long,
    )
    dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
    dist.all_reduce(token_tensor, op=dist.ReduceOp.SUM)
    if rank == 0:
        average_loss = float(loss_tensor / world_size)
        tokens_per_second = int(token_tensor) / elapsed
        save_checkpoint(
            model,
            optimizer,
            Path(args.checkpoint),
            step=args.steps,
            metadata={"world_size": world_size, "batch_size_per_rank": args.batch_size},
        )
        print("backend:", dist.get_backend())
        print("world_size:", world_size)
        print("device:", device)
        print("average loss:", average_loss)
        print("global tokens/s:", tokens_per_second)
        print("checkpoint:", args.checkpoint)
        print("DDP train smoke test: PASS")
    dist.barrier()
    dist.destroy_process_group()


def main():
    args = parse_args()
    if args.steps <= 0:
        raise ValueError("--steps must be positive")
    rank, world_size, local_rank, from_torchrun = setup_process_group()
    if not from_torchrun:
        run_local_check(args)
        return
    run_distributed(args, rank, world_size, local_rank)


if __name__ == "__main__":
    main()
