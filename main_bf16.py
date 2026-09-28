import torch

from qwen25.config import QwenConfig
from qwen25.model import QwenForCausalLM


def main():
    config = QwenConfig.from_json("configs/tiny.json")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    supports_bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()
    model = QwenForCausalLM(config).to(device)
    ids = torch.randint(0, config.vocab_size, (2, 16), device=device)
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=supports_bf16):
        result = model(ids, labels=ids)
    print("device:", device)
    print("bf16 supported:", supports_bf16)
    print("logits dtype:", result["logits"].dtype)
    print("loss:", float(result["loss"].detach()))
    assert torch.isfinite(result["loss"])
    print("BF16 autocast smoke test: PASS")


if __name__ == "__main__":
    main()
