from pathlib import Path

from qwen25.config import QwenConfig


def main():
    root = Path(__file__).parent
    tiny = QwenConfig.from_json(root / "configs/tiny.json")
    reference = QwenConfig.from_json(root / "configs/qwen25_0_5b_reference.json")
    print("[tiny]")
    print(tiny)
    print("head_dim =", tiny.head_dim)
    print("[Qwen2.5 0.5B reference]")
    print(reference)
    print("head_dim =", reference.head_dim)
    print("This reference config is for shape inspection; it is not a three-week from-scratch training target.")


if __name__ == "__main__":
    main()
