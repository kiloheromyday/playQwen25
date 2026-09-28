import math
from typing import Optional

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .config import QwenConfig


class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, hidden_states: Tensor) -> Tensor:
        variance = hidden_states.float().pow(2).mean(dim=-1, keepdim=True)
        normalized = hidden_states * torch.rsqrt(variance + self.eps).to(hidden_states.dtype)
        return normalized * self.weight


class RotaryEmbedding(nn.Module):
    def __init__(self, config: QwenConfig):
        super().__init__()
        inv_freq = 1.0 / (
            config.rope_theta ** (torch.arange(0, config.head_dim, 2).float() / config.head_dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.max_seq_len_cached = 0
        self.register_buffer("cos_cached", torch.empty(0), persistent=False)
        self.register_buffer("sin_cached", torch.empty(0), persistent=False)

    def _build_cache(self, seq_len: int, device: torch.device, dtype: torch.dtype):
        if seq_len <= self.max_seq_len_cached and self.cos_cached.device == device:
            return
        positions = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(positions, self.inv_freq.to(device))
        emb = torch.cat((freqs, freqs), dim=-1)
        self.cos_cached = emb.cos()[None, None, :, :].to(dtype)
        self.sin_cached = emb.sin()[None, None, :, :].to(dtype)
        self.max_seq_len_cached = seq_len

    def forward(self, q: Tensor, k: Tensor, position_offset: int = 0):
        seq_len = q.size(-2) + position_offset
        self._build_cache(seq_len, q.device, q.dtype)
        cos = self.cos_cached[:, :, position_offset:seq_len, :]
        sin = self.sin_cached[:, :, position_offset:seq_len, :]
        return apply_rotary(q, cos, sin), apply_rotary(k, cos, sin)


def rotate_half(x: Tensor) -> Tensor:
    half = x.size(-1) // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def apply_rotary(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    return (x * cos) + (rotate_half(x) * sin)


class GroupedQueryAttention(nn.Module):
    def __init__(self, config: QwenConfig):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.num_groups = self.num_heads // self.num_key_value_heads
        projection_size = self.num_heads * self.head_dim
        kv_projection_size = self.num_key_value_heads * self.head_dim
        self.q_proj = nn.Linear(config.hidden_size, projection_size, bias=config.attention_bias)
        self.k_proj = nn.Linear(config.hidden_size, kv_projection_size, bias=config.attention_bias)
        self.v_proj = nn.Linear(config.hidden_size, kv_projection_size, bias=config.attention_bias)
        self.o_proj = nn.Linear(projection_size, config.hidden_size, bias=config.attention_bias)
        self.rotary = RotaryEmbedding(config)
        self.dropout = config.attention_dropout

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Optional[Tensor] = None,
        past_key_value: Optional[tuple[Tensor, Tensor]] = None,
        use_cache: bool = False,
    ):
        batch, seq_len, _ = hidden_states.shape
        q = self.q_proj(hidden_states).view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(hidden_states).view(batch, seq_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(hidden_states).view(batch, seq_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        past_length = 0 if past_key_value is None else past_key_value[0].size(-2)
        q, k = self.rotary(q, k, position_offset=past_length)
        if past_key_value is not None:
            past_k, past_v = past_key_value
            k = torch.cat((past_k, k), dim=-2)
            v = torch.cat((past_v, v), dim=-2)
        present_key_value = (k, v)
        if self.num_groups > 1:
            k = k.repeat_interleave(self.num_groups, dim=1)
            v = v.repeat_interleave(self.num_groups, dim=1)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        total_length = k.size(-2)
        query_positions = torch.arange(past_length, past_length + seq_len, device=hidden_states.device)[:, None]
        key_positions = torch.arange(total_length, device=hidden_states.device)[None, :]
        causal = key_positions > query_positions
        scores = scores.masked_fill(causal[None, None, :, :], torch.finfo(scores.dtype).min)
        if attention_mask is not None:
            scores = scores + attention_mask
        weights = F.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype)
        weights = F.dropout(weights, p=self.dropout, training=self.training)
        output = torch.matmul(weights, v).transpose(1, 2).contiguous().view(batch, seq_len, -1)
        output = self.o_proj(output)
        if use_cache:
            return output, present_key_value
        return output


class SwiGLUMLP(nn.Module):
    def __init__(self, config: QwenConfig):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=config.mlp_bias)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=config.mlp_bias)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=config.mlp_bias)

    def forward(self, hidden_states: Tensor) -> Tensor:
        return self.down_proj(F.silu(self.gate_proj(hidden_states)) * self.up_proj(hidden_states))


class DecoderLayer(nn.Module):
    def __init__(self, config: QwenConfig):
        super().__init__()
        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.self_attn = GroupedQueryAttention(config)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.mlp = SwiGLUMLP(config)
        self.hidden_dropout = config.hidden_dropout

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Optional[Tensor] = None,
        past_key_value: Optional[tuple[Tensor, Tensor]] = None,
        use_cache: bool = False,
    ):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        attention_result = self.self_attn(
            hidden_states,
            attention_mask,
            past_key_value=past_key_value,
            use_cache=use_cache,
        )
        if use_cache:
            attention_output, present_key_value = attention_result
        else:
            attention_output = attention_result
        hidden_states = residual + F.dropout(attention_output, p=self.hidden_dropout, training=self.training)
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = residual + F.dropout(self.mlp(hidden_states), p=self.hidden_dropout, training=self.training)
        if use_cache:
            return hidden_states, present_key_value
        return hidden_states


class QwenForCausalLM(nn.Module):
    def __init__(self, config: QwenConfig):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([DecoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.weight
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        input_ids: Tensor,
        labels: Optional[Tensor] = None,
        past_key_values: Optional[list[tuple[Tensor, Tensor]]] = None,
        use_cache: bool = False,
    ) -> dict:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, seq_len]")
        past_length = 0 if past_key_values is None else past_key_values[0][0].size(-2)
        if input_ids.size(1) + past_length > self.config.max_position_embeddings:
            raise ValueError("sequence length exceeds max_position_embeddings")
        if past_key_values is not None and len(past_key_values) != len(self.layers):
            raise ValueError("past_key_values must contain one key/value pair per decoder layer")
        hidden_states = self.embed_tokens(input_ids)
        next_key_values = [] if use_cache else None
        for layer_index, layer in enumerate(self.layers):
            past_key_value = None if past_key_values is None else past_key_values[layer_index]
            layer_result = layer(hidden_states, past_key_value=past_key_value, use_cache=use_cache)
            if use_cache:
                hidden_states, present_key_value = layer_result
                next_key_values.append(present_key_value)
            else:
                hidden_states = layer_result
        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)
        result = {"logits": logits}
        if use_cache:
            result["past_key_values"] = next_key_values
        if labels is not None:
            if labels.shape != input_ids.shape:
                raise ValueError("labels must have the same shape as input_ids")
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            result["loss"] = F.cross_entropy(shift_logits.view(-1, logits.size(-1)), shift_labels.view(-1), ignore_index=-100)
        return result

    @torch.no_grad()
    def generate(self, input_ids: Tensor, max_new_tokens: int, eos_token_id: Optional[int] = None) -> Tensor:
        self.eval()
        if max_new_tokens <= 0:
            return input_ids
        if input_ids.size(1) > self.config.max_position_embeddings:
            input_ids = input_ids[:, -self.config.max_position_embeddings :]
        result = self(input_ids, use_cache=True)
        past_key_values = result["past_key_values"]
        logits = result["logits"][:, -1, :]
        for _ in range(max_new_tokens):
            next_token = logits.argmax(dim=-1, keepdim=True)
            input_ids = torch.cat((input_ids, next_token), dim=-1)
            if eos_token_id is not None and torch.all(next_token == eos_token_id):
                break
            if past_key_values[0][0].size(-2) >= self.config.max_position_embeddings:
                result = self(input_ids[:, -self.config.max_position_embeddings :], use_cache=True)
            else:
                result = self(next_token, past_key_values=past_key_values, use_cache=True)
            past_key_values = result["past_key_values"]
            logits = result["logits"][:, -1, :]
        return input_ids
