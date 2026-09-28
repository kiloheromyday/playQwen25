import argparse
from pathlib import Path

import torch

from qwen25.config import QwenConfig
from qwen25.data import make_next_token_batch, pack_token_ids
from qwen25.model import QwenForCausalLM
from qwen25.tokenizer import SimpleCharTokenizer
from qwen25.train import save_checkpoint, train_steps


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    torch.manual_seed(0)
    device = torch.device(args.device)
    texts = [
        "我喜欢学习人工智能。模型根据前面的token预测下一个token。",
        "Qwen2.5 style decoder uses attention and an MLP.",
        "def add(a, b): return a + b",
    ]
    tokenizer = SimpleCharTokenizer.from_texts(texts)
    config = QwenConfig.from_json("configs/tiny.json")
    config.vocab_size = tokenizer.vocab_size
    ids = []
    for text in texts * 32:
        ids.extend(tokenizer.encode(text, add_eos=True))
    sequences = pack_token_ids(ids, sequence_length=32, eos_token_id=tokenizer.eos_token_id)
    dataset = torch.tensor(sequences, dtype=torch.long)
    batches = [make_next_token_batch(dataset[index : index + 4]).input_ids for index in range(0, len(dataset), 4)]
    if not batches:
        raise RuntimeError("not enough tokenized data for one sequence")
    model = QwenForCausalLM(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)
    cycling_batches = (batch for batch in batches)
    stats = train_steps(model, cycling_batches, optimizer, args.steps, device)
    for item in stats[::max(1, len(stats) // 5)]:
        print(f"step={item.step:03d} loss={item.loss:.4f} tokens={item.tokens}")
    output = Path("artifacts/tiny_checkpoint.pt")
    save_checkpoint(model, optimizer, output, stats[-1].step)
    print("device:", device)
    print("vocab_size:", config.vocab_size)
    print("sequences:", len(dataset))
    print("checkpoint:", output)


if __name__ == "__main__":
    main()
