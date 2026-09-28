from pathlib import Path

import torch

from qwen25.config import QwenConfig
from qwen25.model import QwenForCausalLM
from qwen25.train import load_checkpoint, save_checkpoint


def main():
    torch.manual_seed(0)
    config = QwenConfig.from_json("configs/tiny.json")
    model = QwenForCausalLM(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    input_ids = torch.randint(0, config.vocab_size, (1, 12))
    loss = model(input_ids, labels=input_ids)["loss"]
    loss.backward()
    optimizer.step()
    path = Path("artifacts/checkpoint_smoke.pt")
    save_checkpoint(model, optimizer, path, step=7)
    restored_model = QwenForCausalLM(config)
    restored_optimizer = torch.optim.AdamW(restored_model.parameters(), lr=1e-3)
    step = load_checkpoint(restored_model, restored_optimizer, path)
    with torch.no_grad():
        original_logits = model(input_ids)["logits"]
        restored_logits = restored_model(input_ids)["logits"]
    max_diff = (original_logits - restored_logits).abs().max().item()
    print("checkpoint:", path)
    print("restored step:", step)
    print("max logit difference:", max_diff)
    assert step == 7
    assert max_diff < 1e-6
    print("checkpoint save/restore test: PASS")


if __name__ == "__main__":
    main()
