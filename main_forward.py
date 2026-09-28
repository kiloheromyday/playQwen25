import torch

from qwen25.config import QwenConfig
from qwen25.model import QwenForCausalLM


def main():
    torch.manual_seed(0)
    config = QwenConfig.from_json("configs/tiny.json")
    model = QwenForCausalLM(config)
    input_ids = torch.randint(0, config.vocab_size, (2, 16))
    result = model(input_ids, labels=input_ids)
    print("input_ids:", tuple(input_ids.shape))
    print("logits:", tuple(result["logits"].shape))
    print("loss:", float(result["loss"].detach()))
    assert result["logits"].shape == (2, 16, config.vocab_size)
    assert result["loss"].ndim == 0
    assert torch.isfinite(result["loss"])
    print("forward and next-token loss test: PASS")


if __name__ == "__main__":
    main()
