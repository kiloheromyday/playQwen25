import math
import time

import torch


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@torch.inference_mode()
def generate_text(model, tokenizer, text, device, max_new_tokens=48, temperature=0.7,
                  top_p=0.9, repetition_penalty=1.1, seed=0):
    """Bounded single-sequence generation; timings include synchronized GPU work."""
    if not text.strip():
        raise ValueError("输入不能为空")
    if not 1 <= max_new_tokens < model.config.max_position_embeddings:
        raise ValueError("生成长度必须小于模型上下文长度")
    if not 0 <= temperature <= 2 or not 0 < top_p <= 1 or not 1 <= repetition_penalty <= 2:
        raise ValueError("采样参数超出范围")
    if not all(math.isfinite(x) for x in (temperature, top_p, repetition_penalty)):
        raise ValueError("采样参数必须为有限数值")
    model.eval()
    encoded = tokenizer.encode(text)
    capacity = model.config.max_position_embeddings - max_new_tokens
    prompt_ids = encoded[-capacity:]
    ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    generator = torch.Generator(device=device).manual_seed(seed)
    generated = []
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    synchronize(device)
    started = time.perf_counter()
    result = model(ids, use_cache=True)
    first_at = None
    stopped_on_eos = False
    for index in range(max_new_tokens):
        logits = result["logits"][0, -1].float().clone()
        repeated = torch.tensor(sorted(set(prompt_ids + generated)), device=device)
        logits[repeated] = torch.where(logits[repeated] > 0,
                                      logits[repeated] / repetition_penalty,
                                      logits[repeated] * repetition_penalty)
        logits[tokenizer.pad_token_id] = -float("inf")
        if temperature == 0:
            next_id = logits.argmax().item()
        else:
            probabilities = (logits / temperature).softmax(-1)
            values, order = probabilities.sort(descending=True)
            remove = values.cumsum(-1) - values >= top_p
            values[remove] = 0
            choice = torch.multinomial(values, 1, generator=generator)
            next_id = order[choice].item()
        generated.append(next_id)
        if index == 0:
            synchronize(device)
            first_at = time.perf_counter()
        if next_id == tokenizer.eos_token_id:
            stopped_on_eos = True
            break
        if index + 1 < max_new_tokens:
            result = model(torch.tensor([[next_id]], device=device),
                           past_key_values=result["past_key_values"], use_cache=True)
    synchronize(device)
    ended = time.perf_counter()
    visible = [i for i in generated if i not in (tokenizer.eos_token_id, tokenizer.pad_token_id)]
    bigrams = list(zip(visible, visible[1:]))
    return {
        "text": tokenizer.decode(visible), "generated_ids": generated,
        "prompt_tokens": len(prompt_ids), "input_tokens_before_truncation": len(encoded),
        "truncated_tokens": len(encoded) - len(prompt_ids),
        "unknown_input_rate": sum(i == tokenizer.unk_token_id for i in prompt_ids) / len(prompt_ids),
        "generated_tokens": len(generated), "stopped_on_eos": stopped_on_eos,
        "ttft_ms": (first_at - started) * 1000,
        "latency_ms": (ended - started) * 1000,
        "tokens_per_second": len(generated) / max(ended - started, 1e-9),
        "decode_tokens_per_second": (len(generated) - 1) / max(ended - first_at, 1e-9) if len(generated) > 1 else None,
        "distinct_1": len(set(visible)) / len(visible) if visible else None,
        "distinct_2": len(set(bigrams)) / len(bigrams) if bigrams else None,
        "repeated_bigram_rate": 1 - len(set(bigrams)) / len(bigrams) if bigrams else None,
        "peak_allocated_mib": torch.cuda.max_memory_allocated(device) / 2**20 if device.type == "cuda" else None,
    }
