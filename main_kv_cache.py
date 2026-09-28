import torch

from qwen25.config import QwenConfig
from qwen25.model import QwenForCausalLM


def main():
    torch.manual_seed(0)
    config = QwenConfig.from_json("configs/tiny.json")
    model = QwenForCausalLM(config).eval()
    prompt = torch.randint(0, config.vocab_size, (1, 8))
    continuation = torch.randint(0, config.vocab_size, (1, 4))
    full_input = torch.cat((prompt, continuation), dim=1)

    with torch.no_grad():
        full_logits = model(full_input)["logits"][:, -continuation.size(1):]
        first = model(prompt, use_cache=True)
        cached_logits = []
        past_key_values = first["past_key_values"]
        for index in range(continuation.size(1)):
            step = model(
                continuation[:, index : index + 1],
                past_key_values=past_key_values,
                use_cache=True,
            )
            cached_logits.append(step["logits"])
            past_key_values = step["past_key_values"]
        cached_logits = torch.cat(cached_logits, dim=1)

    max_difference = (full_logits - cached_logits).abs().max().item()
    print("prompt shape:", tuple(prompt.shape))
    print("cached KV shape per layer:", tuple(past_key_values[0][0].shape))
    print("max logit difference:", max_difference)
    assert max_difference < 1e-5
    print("KV cache incremental/full equivalence: PASS")


if __name__ == "__main__":
    main()
