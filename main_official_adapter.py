import argparse
from pathlib import Path

from qwen25.config import QwenConfig
from qwen25.tokenizer import QwenTokenizerAdapter


def main():
    parser = argparse.ArgumentParser(
        description="Inspect a local official Qwen tokenizer/config without downloading anything."
    )
    parser.add_argument("--model-path", required=True, help="Local directory containing config.json and tokenizer files.")
    args = parser.parse_args()
    model_path = Path(args.model_path)
    config_path = model_path / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"missing local Hugging Face config: {config_path}")

    config = QwenConfig.from_hf_json(config_path)
    print("config vocab size:", config.vocab_size)
    print("hidden size:", config.hidden_size)
    print("layers:", config.num_hidden_layers)
    print("attention heads:", config.num_attention_heads)
    print("KV heads:", config.num_key_value_heads)
    print("head dim:", config.head_dim)
    print("rope theta:", config.rope_theta)

    try:
        tokenizer = QwenTokenizerAdapter.from_pretrained(
            str(model_path),
            local_files_only=True,
            trust_remote_code=False,
        )
    except ImportError as exc:
        print("tokenizer check skipped:", exc)
        print("Install transformers on the server before loading the official tokenizer.")
        return
    sample = "模型根据前面的 token 预测下一个 token。"
    ids = tokenizer.encode(sample, add_eos=True)
    print("tokenizer vocab size:", tokenizer.vocab_size)
    print("sample token count:", len(ids))
    print("decoded sample:", tokenizer.decode(ids))
    print("official local adapter smoke test: PASS")


if __name__ == "__main__":
    main()
