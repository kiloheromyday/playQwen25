import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path
import time

import torch

from main_train_pipeline import set_seed
from qwen25.config import QwenConfig
from qwen25.data import build_data_bundle, load_text_documents, make_dataloader, split_documents
from qwen25.model import QwenForCausalLM
from qwen25.tokenizer import SimpleCharTokenizer
from qwen25.train import evaluate_loss, load_checkpoint, save_checkpoint, train_loader_steps


def main():
    parser = argparse.ArgumentParser(description="Small local training with measurable checks.")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", default="artifacts/local_smoke")
    args = parser.parse_args()
    if args.steps < 10:
        raise ValueError("use at least 10 steps for the learning check")
    set_seed(0)
    device = torch.device(args.device)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    documents = load_text_documents("data/tiny_corpus")
    train_documents, _ = split_documents(documents, 0.25, seed=0)
    tokenizer = SimpleCharTokenizer.from_texts(train_documents)
    bundle = build_data_bundle(documents, tokenizer, 32, 0.25, seed=0)
    train_loader = make_dataloader(bundle.train_dataset, 2, shuffle=True)
    train_eval_loader = make_dataloader(bundle.train_dataset, 2, shuffle=False)
    valid_loader = make_dataloader(bundle.valid_dataset, 2, shuffle=False)
    config = QwenConfig.from_json("configs/tiny.json")
    config.vocab_size = tokenizer.vocab_size
    model = QwenForCausalLM(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.003)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    history = []

    def record(step):
        train_loss, _ = evaluate_loss(model, train_eval_loader, device)
        valid_loss, valid_ppl = evaluate_loss(model, valid_loader, device)
        row = dict(step=step, train_loss=train_loss, valid_loss=valid_loss, valid_ppl=valid_ppl)
        history.append(row)
        print(json.dumps(row), flush=True)

    record(0)

    def on_step(stats):
        if stats.step % 10 == 0 or stats.step == args.steps:
            record(stats.step)

    started = time.perf_counter()
    stats = train_loader_steps(model, train_loader, optimizer, args.steps, device,
                               grad_accumulation_steps=2, on_step=on_step)
    elapsed = time.perf_counter() - started
    assert all(math.isfinite(row["loss"]) for row in map(asdict, stats))
    assert history[-1]["train_loss"] < history[0]["train_loss"], "training loss did not fall"

    checkpoint = output / "checkpoint.pt"
    save_checkpoint(model, optimizer, checkpoint, args.steps,
                    metadata={"config": asdict(config), "tokenizer": tokenizer.to_dict()})
    tokenizer.save(output / "tokenizer.json")
    restored = QwenForCausalLM(config).to(device)
    restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=0.003)
    restored_step = load_checkpoint(restored, restored_optimizer, checkpoint)
    probe = next(iter(train_eval_loader)).to(device)
    model.eval()
    restored.eval()
    with torch.no_grad():
        difference = (model(probe)["logits"] - restored(probe)["logits"]).abs().max().item()
    assert restored_step == args.steps and difference < 1e-6
    prompt = probe[:1, :8]
    generated = restored.generate(prompt, max_new_tokens=24)
    assert generated.shape == (1, 32)
    report = {
        "torch": torch.__version__, "cuda_runtime": torch.version.cuda,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "precision": "float32", "config": asdict(config),
        "parameters": sum(p.numel() for p in model.parameters()),
        "train_documents": len(bundle.train_documents), "valid_documents": len(bundle.valid_documents),
        "train_sequences": len(bundle.train_dataset), "valid_sequences": len(bundle.valid_dataset),
        "steps": args.steps, "gradient_accumulation": 2,
        "target_tokens_processed": sum(s.tokens for s in stats),
        "elapsed_seconds_including_evaluation": elapsed,
        "peak_allocated_mib": torch.cuda.max_memory_allocated(device) / 2**20 if device.type == "cuda" else None,
        "checkpoint_max_logit_difference": difference,
        "prompt": tokenizer.decode(prompt[0].tolist()),
        "generated": tokenizer.decode(generated[0].tolist()),
        "history": history, "updates": [asdict(s) for s in stats],
        "scope": "Teaching corpus only; no language capability or official Qwen parity claim.",
    }
    (output / "results.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("parameters:", report["parameters"])
    print("peak allocated MiB:", report["peak_allocated_mib"])
    print("checkpoint max difference:", difference)
    print("generated:", report["generated"])
    print("results:", output / "results.json")
    print("local small-scale training: PASS")


if __name__ == "__main__":
    main()
