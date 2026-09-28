import torch

from qwen25.config import QwenConfig
from qwen25.model import GroupedQueryAttention, RMSNorm, RotaryEmbedding, SwiGLUMLP


def main():
    torch.manual_seed(0)
    config = QwenConfig.from_json("configs/tiny.json")
    hidden = torch.randn(2, 8, config.hidden_size)
    norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
    mlp = SwiGLUMLP(config)
    attention = GroupedQueryAttention(config)
    q = torch.randn(2, config.num_attention_heads, 8, config.head_dim)
    k = torch.randn(2, config.num_key_value_heads, 8, config.head_dim)
    rotary = RotaryEmbedding(config)
    rotated_q, rotated_k = rotary(q, k)
    print("RMSNorm:", hidden.shape, "->", norm(hidden).shape)
    print("RoPE Q:", q.shape, "->", rotated_q.shape)
    print("RoPE K:", k.shape, "->", rotated_k.shape)
    print("GQA:", hidden.shape, "->", attention(hidden).shape)
    print("SwiGLU:", hidden.shape, "->", mlp(hidden).shape)
    assert attention.num_heads % attention.num_key_value_heads == 0
    assert torch.isfinite(attention(hidden)).all()
    print("component smoke test: PASS")


if __name__ == "__main__":
    main()
