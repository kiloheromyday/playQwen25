import unittest

import torch

from qwen25.config import QwenConfig
from qwen25.data import deduplicate_documents, split_documents
from qwen25.model import QwenForCausalLM, RMSNorm
from qwen25.tokenizer import SimpleCharTokenizer


class ModelSmokeTests(unittest.TestCase):
    def setUp(self):
        self.config = QwenConfig(vocab_size=32, hidden_size=32, intermediate_size=88, num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=64)

    def test_config_head_dim(self):
        self.assertEqual(self.config.head_dim, 8)

    def test_rms_norm_shape_and_finite(self):
        x = torch.randn(2, 5, self.config.hidden_size)
        y = RMSNorm(self.config.hidden_size)(x)
        self.assertEqual(y.shape, x.shape)
        self.assertTrue(torch.isfinite(y).all())

    def test_forward_and_backward(self):
        model = QwenForCausalLM(self.config)
        ids = torch.randint(0, self.config.vocab_size, (2, 12))
        result = model(ids, labels=ids)
        self.assertEqual(result["logits"].shape, (2, 12, self.config.vocab_size))
        result["loss"].backward()
        self.assertTrue(all(parameter.grad is not None for parameter in model.parameters() if parameter.requires_grad))

    def test_gqa_shape(self):
        model = QwenForCausalLM(self.config)
        ids = torch.randint(0, self.config.vocab_size, (1, 8))
        self.assertEqual(model(ids)["logits"].shape, (1, 8, self.config.vocab_size))

    def test_causal_mask_blocks_future_tokens(self):
        torch.manual_seed(7)
        model = QwenForCausalLM(self.config).eval()
        prefix = torch.randint(0, self.config.vocab_size, (1, 5))
        future_a = torch.randint(0, self.config.vocab_size, (1, 4))
        future_b = torch.randint(0, self.config.vocab_size, (1, 4))
        first = model(torch.cat((prefix, future_a), dim=1))["logits"][:, : prefix.size(1)]
        second = model(torch.cat((prefix, future_b), dim=1))["logits"][:, : prefix.size(1)]
        self.assertTrue(torch.allclose(first, second, atol=1e-6, rtol=1e-6))

    def test_documents_are_cleaned_deduplicated_and_split(self):
        documents = deduplicate_documents(["a\r\n\r\nb", "a\n\nb", "c"])
        self.assertEqual(documents, ["a\n\nb", "c"])
        train, valid = split_documents(documents, valid_ratio=0.5, seed=0)
        self.assertEqual(len(train) + len(valid), 2)
        self.assertEqual(set(train).isdisjoint(valid), True)

    def test_hf_config_subset_and_tokenizer_round_trip(self):
        config = QwenConfig.from_hf_json("configs/tiny.json")
        self.assertEqual(config.vocab_size, 64)
        tokenizer = SimpleCharTokenizer.from_texts(["abc"])
        restored = SimpleCharTokenizer.from_dict(tokenizer.to_dict())
        self.assertEqual(restored.decode(restored.encode("abc")), "abc")

    def test_kv_cache_matches_full_forward(self):
        torch.manual_seed(11)
        model = QwenForCausalLM(self.config).eval()
        prompt = torch.randint(0, self.config.vocab_size, (1, 5))
        continuation = torch.randint(0, self.config.vocab_size, (1, 3))
        with torch.no_grad():
            full = model(torch.cat((prompt, continuation), dim=1))["logits"][:, -3:]
            result = model(prompt, use_cache=True)
            cached = []
            past = result["past_key_values"]
            for index in range(continuation.size(1)):
                result = model(
                    continuation[:, index : index + 1],
                    past_key_values=past,
                    use_cache=True,
                )
                cached.append(result["logits"])
                past = result["past_key_values"]
            cached = torch.cat(cached, dim=1)
        self.assertTrue(torch.allclose(full, cached, atol=1e-6, rtol=1e-6))


if __name__ == "__main__":
    unittest.main()
