import argparse
from pathlib import Path

from qwen25.data import build_data_bundle, load_text_documents, make_dataloader, split_documents
from qwen25.tokenizer import SimpleCharTokenizer


def main():
    parser = argparse.ArgumentParser(description="Build and inspect a document-level train/valid pipeline.")
    parser.add_argument("--data", default="data/tiny_corpus")
    parser.add_argument("--sequence-length", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--valid-ratio", type=float, default=0.25)
    args = parser.parse_args()

    documents = load_text_documents(args.data)
    train_documents, _ = split_documents(documents, args.valid_ratio, seed=0)
    tokenizer = SimpleCharTokenizer.from_texts(train_documents)
    bundle = build_data_bundle(
        documents,
        tokenizer,
        sequence_length=args.sequence_length,
        valid_ratio=args.valid_ratio,
        seed=0,
    )
    train_loader = make_dataloader(bundle.train_dataset, args.batch_size, shuffle=True, seed=0)
    first_batch = next(iter(train_loader))
    print("source:", Path(args.data).resolve())
    print("documents:", len(documents))
    print("train documents:", len(bundle.train_documents))
    print("valid documents:", len(bundle.valid_documents))
    print("vocab size:", tokenizer.vocab_size)
    print("train sequences:", len(bundle.train_dataset))
    print("valid sequences:", len(bundle.valid_dataset) if bundle.valid_dataset else 0)
    print("first batch shape:", tuple(first_batch.shape))
    print("pipeline test: PASS")


if __name__ == "__main__":
    main()
