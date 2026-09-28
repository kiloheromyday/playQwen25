from dataclasses import asdict, dataclass
import json
from pathlib import Path


@dataclass
class QwenConfig:
    vocab_size: int = 256
    hidden_size: int = 128
    intermediate_size: int = 352
    num_hidden_layers: int = 4
    num_attention_heads: int = 4
    num_key_value_heads: int = 2
    max_position_embeddings: int = 512
    rope_theta: float = 10000.0
    rms_norm_eps: float = 1e-6
    attention_dropout: float = 0.0
    hidden_dropout: float = 0.0
    tie_word_embeddings: bool = True
    attention_bias: bool = False
    mlp_bias: bool = False

    def __post_init__(self):
        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError("hidden_size must be divisible by num_attention_heads")
        if self.num_attention_heads % self.num_key_value_heads != 0:
            raise ValueError("num_attention_heads must be divisible by num_key_value_heads")
        if self.num_hidden_layers <= 0:
            raise ValueError("num_hidden_layers must be positive")
        if self.vocab_size <= 0 or self.max_position_embeddings <= 0:
            raise ValueError("vocab_size and max_position_embeddings must be positive")
        if self.intermediate_size <= 0:
            raise ValueError("intermediate_size must be positive")

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads

    @classmethod
    def from_json(cls, path: str | Path) -> "QwenConfig":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        supported = {field.name for field in cls.__dataclass_fields__.values()}
        unknown = sorted(set(data) - supported)
        if unknown:
            raise ValueError(f"Unsupported config fields: {unknown}")
        return cls(**data)

    @classmethod
    def from_hf_json(cls, path: str | Path) -> "QwenConfig":
        """Read the subset of a Hugging Face Qwen config used by this implementation."""
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        field_names = {field.name for field in cls.__dataclass_fields__.values()}
        values = {name: data[name] for name in field_names if name in data}
        if "num_key_value_heads" not in values and "num_attention_heads" in values:
            values["num_key_value_heads"] = values["num_attention_heads"]
        missing = sorted(name for name in ("vocab_size", "hidden_size", "num_hidden_layers", "num_attention_heads") if name not in values)
        if missing:
            raise ValueError(f"Hugging Face config is missing required fields: {missing}")
        return cls(**values)

    def to_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")
