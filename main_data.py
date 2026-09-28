import torch

from qwen25.data import make_next_token_batch, pack_token_ids


def main():
    token_ids = list(range(1, 20))
    sequences = pack_token_ids(token_ids, sequence_length=8, eos_token_id=0)
    tensor = torch.tensor(sequences, dtype=torch.long)
    batch = make_next_token_batch(tensor)
    print("raw token ids:", token_ids)
    print("packed sequences:")
    print(tensor)
    print("input_ids shape:", tuple(batch.input_ids.shape))
    print("labels shape:", tuple(batch.labels.shape))
    print("next-token example:", batch.input_ids[0, :-1].tolist(), "->", batch.labels[0, 1:].tolist())
    assert tensor.size(1) == 8
    assert batch.input_ids.shape == batch.labels.shape
    print("packing and label alignment test: PASS")


if __name__ == "__main__":
    main()
