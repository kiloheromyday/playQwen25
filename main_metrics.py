import torch

from qwen25.config import QwenConfig
from qwen25.model import QwenForCausalLM
from qwen25.train import evaluate_loss


def main():
    config = QwenConfig.from_json("configs/tiny.json")
    model = QwenForCausalLM(config)
    batches = [torch.randint(0, config.vocab_size, (2, 16)) for _ in range(3)]
    loss, perplexity = evaluate_loss(model, batches, torch.device("cpu"))
    print("validation loss:", loss)
    print("perplexity:", perplexity)
    assert loss > 0 and perplexity > 1
    print("validation loss and PPL test: PASS")


if __name__ == "__main__":
    main()
