import torch

from qwen25.config import QwenConfig
from qwen25.model import QwenForCausalLM


def main():
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = QwenConfig.from_json("configs/tiny.json")
    model = QwenForCausalLM(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)
    batch = torch.tensor([[4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]], dtype=torch.long, device=device)
    losses = []
    for step in range(1, 101):
        result = model(batch, labels=batch)
        result["loss"].backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        losses.append(float(result["loss"].detach()))
        if step == 1 or step % 20 == 0:
            print(f"step={step:03d} loss={losses[-1]:.4f}")
    print("device:", device)
    print("initial_loss:", losses[0])
    print("final_loss:", losses[-1])
    assert losses[-1] < losses[0] * 0.5, "single-batch overfit did not reduce loss enough"
    print("single-batch overfit test: PASS")


if __name__ == "__main__":
    main()
