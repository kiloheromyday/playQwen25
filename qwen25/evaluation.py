import math

import torch
import torch.nn.functional as F


@torch.inference_mode()
def score_documents(model, tokenizer, documents, device):
    """Score every next-token target once, keeping one-token overlap at chunk boundaries."""
    was_training = model.training
    model.eval()
    loss_sum = known_loss_sum = 0.0
    count = known_count = correct = top5_correct = known_correct = 0
    unknown_input = input_count = 0
    per_document = []
    try:
        for text in documents:
            ids = tokenizer.encode(text, add_eos=True)
            unknown_input += sum(i == tokenizer.unk_token_id for i in ids[:-1])
            input_count += len(ids) - 1
            document_loss = 0.0
            document_targets = 0
            width = model.config.max_position_embeddings
            for start in range(0, len(ids) - 1, width - 1):
                batch = torch.tensor([ids[start:start + width]], device=device)
                logits = model(batch)["logits"][0, :-1].float()
                targets = batch[0, 1:]
                losses = F.cross_entropy(logits, targets, reduction="none")
                predictions = logits.argmax(-1)
                known = targets != tokenizer.unk_token_id
                loss_sum += losses.sum().item()
                document_loss += losses.sum().item()
                count += targets.numel()
                document_targets += targets.numel()
                correct += (predictions == targets).sum().item()
                top5_correct += (logits.topk(min(5, logits.size(-1)), dim=-1).indices == targets[:, None]).any(-1).sum().item()
                known_loss_sum += losses[known].sum().item()
                known_count += known.sum().item()
                known_correct += ((predictions == targets) & known).sum().item()
            per_document.append({"characters": len(text), "targets": document_targets,
                                 "loss": document_loss / document_targets if document_targets else None})
    finally:
        model.train(was_training)
    if count == 0:
        raise ValueError("评测文本中没有可预测的 token")
    loss = loss_sum / count
    return {
        "documents": len(documents), "target_tokens": count, "loss": loss,
        "ppl": math.exp(loss) if loss < 700 else None,
        "top1_accuracy": correct / count, "top5_accuracy": top5_correct / count,
        "unknown_input_rate": unknown_input / max(input_count, 1),
        "known_target_tokens": known_count,
        "known_target_loss": known_loss_sum / known_count if known_count else None,
        "known_top1_accuracy": known_correct / known_count if known_count else None,
        "per_document": per_document,
    }
