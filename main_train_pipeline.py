import argparse
from dataclasses import asdict
import math
from pathlib import Path
import random

import torch

from qwen25.config import QwenConfig
from qwen25.data import (
    build_data_bundle,
    load_text_documents,
    make_dataloader,
    split_documents,
)
from qwen25.model import QwenForCausalLM
from qwen25.tokenizer import SimpleCharTokenizer
from qwen25.train import (
    evaluate_loss,
    load_checkpoint,
    save_checkpoint,
    train_loader_steps,
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def cosine_with_warmup(step: int, warmup_steps: int, total_steps: int) -> float:
    if warmup_steps > 0 and step < warmup_steps:
        return float(step + 1) / warmup_steps
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    progress = min(max(progress, 0.0), 1.0)
    return 0.5 * (1.0 + math.cos(math.pi * progress))


def main():
    parser = argparse.ArgumentParser(description="Train the tiny Qwen2.5-style model on local text files.")
    parser.add_argument("--data", default="data/tiny_corpus")
    parser.add_argument("--config", default="configs/tiny.json")
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--eval-interval", type=int, default=10)
    parser.add_argument("--sequence-length", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accumulation", type=int, default=1)
    parser.add_argument("--valid-ratio", type=float, default=0.25)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--warmup-steps", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--checkpoint", default="artifacts/pipeline_checkpoint.pt")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    if args.steps <= 0:
        raise ValueError("--steps must be positive")
    if args.eval_interval <= 0:
        raise ValueError("--eval-interval must be positive")
    set_seed(args.seed)
    device = torch.device(args.device)
    documents = load_text_documents(args.data)
    train_documents, _ = split_documents(documents, args.valid_ratio, seed=args.seed)
    tokenizer = SimpleCharTokenizer.from_texts(train_documents)
    bundle = build_data_bundle(
        documents,
        tokenizer,
        sequence_length=args.sequence_length,
        valid_ratio=args.valid_ratio,
        seed=args.seed,
    )
    train_loader = make_dataloader(bundle.train_dataset, args.batch_size, shuffle=True, seed=args.seed)
    valid_loader = (
        make_dataloader(bundle.valid_dataset, args.batch_size, shuffle=False, seed=args.seed)
        if bundle.valid_dataset is not None
        else None
    )

    config = QwenConfig.from_json(args.config)
    config.vocab_size = tokenizer.vocab_size
    model = QwenForCausalLM(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    checkpoint_path = Path(args.checkpoint)
    start_step = 0
    if args.resume:
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"checkpoint does not exist: {checkpoint_path}")
        checkpoint_info = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        start_step = int(checkpoint_info["step"])
    total_schedule_steps = start_step + args.steps
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: cosine_with_warmup(step, args.warmup_steps, total_schedule_steps),
    )
    if args.resume:
        start_step = load_checkpoint(model, optimizer, checkpoint_path, scheduler=scheduler)
        print("resumed from step:", start_step)

    print("device:", device)
    print("documents:", len(documents))
    print("train/valid documents:", len(bundle.train_documents), len(bundle.valid_documents))
    print("train/valid sequences:", len(bundle.train_dataset), len(bundle.valid_dataset) if bundle.valid_dataset else 0)
    print("vocab size:", tokenizer.vocab_size)
    def log_step(stats):
        if stats.step == start_step + 1 or stats.step % args.eval_interval == 0 or stats.step == start_step + args.steps:
            message = (
                f"step={stats.step:04d} train_loss={stats.loss:.4f} "
                f"lr={stats.learning_rate:.6g} tokens/s={stats.tokens_per_second:.1f}"
            )
            if valid_loader is not None:
                valid_loss, valid_ppl = evaluate_loss(model, valid_loader, device, use_bf16=args.bf16)
                message += f" valid_loss={valid_loss:.4f} valid_ppl={valid_ppl:.2f}"
            print(message)

    train_loader_steps(
        model, train_loader, optimizer, steps=args.steps, device=device,
        grad_accumulation_steps=args.grad_accumulation, use_bf16=args.bf16,
        scheduler=scheduler, start_step=start_step, on_step=log_step,
    )

    final_step = start_step + args.steps
    metadata = {
        "data": str(Path(args.data).resolve()),
        "sequence_length": args.sequence_length,
        "batch_size": args.batch_size,
        "valid_ratio": args.valid_ratio,
        "seed": args.seed,
        "tokenizer": tokenizer.to_dict(),
        "config": asdict(config),
    }
    save_checkpoint(model, optimizer, checkpoint_path, final_step, scheduler=scheduler, metadata=metadata)
    tokenizer_path = checkpoint_path.with_suffix(".tokenizer.json")
    tokenizer.save(tokenizer_path)
    print("checkpoint:", checkpoint_path)
    print("tokenizer:", tokenizer_path)
    print("training pipeline: PASS")


if __name__ == "__main__":
    main()
