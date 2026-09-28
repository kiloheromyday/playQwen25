from collections import Counter
import json
from pathlib import Path


class SimpleCharTokenizer:
    """A tiny teaching tokenizer; it is not the official Qwen tokenizer."""

    PAD = "<|pad|>"
    EOS = "<|eos|>"
    UNK = "<|unk|>"

    def __init__(self, vocab: list[str]):
        specials = [self.PAD, self.EOS, self.UNK]
        ordered = []
        for token in specials + vocab:
            if token not in ordered:
                ordered.append(token)
        self.id_to_token = ordered
        self.token_to_id = {token: index for index, token in enumerate(ordered)}

    @classmethod
    def from_texts(cls, texts: list[str], min_frequency: int = 1) -> "SimpleCharTokenizer":
        counts = Counter(char for text in texts for char in text)
        vocab = sorted(char for char, count in counts.items() if count >= min_frequency)
        return cls(vocab)

    @classmethod
    def from_file(cls, path: str | Path) -> "SimpleCharTokenizer":
        text = Path(path).read_text(encoding="utf-8")
        return cls.from_texts([text])

    @property
    def vocab_size(self) -> int:
        return len(self.id_to_token)

    @property
    def pad_token_id(self) -> int:
        return self.token_to_id[self.PAD]

    @property
    def eos_token_id(self) -> int:
        return self.token_to_id[self.EOS]

    @property
    def unk_token_id(self) -> int:
        return self.token_to_id[self.UNK]

    def encode(self, text: str, add_eos: bool = False) -> list[int]:
        ids = [self.token_to_id.get(char, self.unk_token_id) for char in text]
        if add_eos:
            ids.append(self.eos_token_id)
        return ids

    def decode(self, ids: list[int]) -> str:
        tokens = [self.id_to_token[index] for index in ids]
        return "".join(token for token in tokens if token not in {self.PAD, self.EOS})

    def to_dict(self) -> dict:
        return {"type": "simple_char", "vocab": self.id_to_token}

    @classmethod
    def from_dict(cls, data: dict) -> "SimpleCharTokenizer":
        if data.get("type") != "simple_char":
            raise ValueError("unsupported tokenizer type")
        vocab = list(data["vocab"])
        return cls([token for token in vocab if token not in {cls.PAD, cls.EOS, cls.UNK}])

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "SimpleCharTokenizer":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


class QwenTokenizerAdapter:
    """Optional adapter for a real Qwen tokenizer from Transformers."""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    @classmethod
    def from_pretrained(cls, model_name_or_path: str, **kwargs):
        try:
            from transformers import AutoTokenizer
        except ImportError as exc:
            raise ImportError("Install transformers to load the official Qwen tokenizer") from exc
        return cls(AutoTokenizer.from_pretrained(model_name_or_path, **kwargs))

    @property
    def vocab_size(self) -> int:
        return len(self.tokenizer)

    @property
    def pad_token_id(self) -> int | None:
        return self.tokenizer.pad_token_id

    @property
    def eos_token_id(self) -> int | None:
        return self.tokenizer.eos_token_id

    def encode(self, text: str, add_eos: bool = False) -> list[int]:
        ids = self.tokenizer.encode(text, add_special_tokens=False)
        if add_eos and self.eos_token_id is not None:
            ids.append(self.eos_token_id)
        return ids

    def decode(self, ids: list[int]) -> str:
        return self.tokenizer.decode(ids, skip_special_tokens=True)
