from qwen25.tokenizer import SimpleCharTokenizer


def main():
    texts = ["我喜欢学习人工智能。", "Qwen2.5 style model", "def add(a, b): return a + b"]
    tokenizer = SimpleCharTokenizer.from_texts(texts)
    for text in texts:
        ids = tokenizer.encode(text, add_eos=True)
        print("text:", repr(text))
        print("ids:", ids)
        print("decoded:", repr(tokenizer.decode(ids)))
        print("length:", len(ids))
    print("vocab_size:", tokenizer.vocab_size)
    print("This is a character-level teaching tokenizer, not the official Qwen tokenizer.")


if __name__ == "__main__":
    main()
