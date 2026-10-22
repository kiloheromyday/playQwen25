import unittest

import torch

from qwen25.config import QwenConfig
from qwen25.model import QwenForCausalLM
from qwen25.train import evaluate_loss, train_loader_steps


class TrainingTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        config = QwenConfig(vocab_size=16, hidden_size=16, intermediate_size=32,
                            num_hidden_layers=1, num_attention_heads=2,
                            num_key_value_heads=1, max_position_embeddings=16)
        self.model = QwenForCausalLM(config)
        self.batches = [torch.randint(0, 16, (2, 8))]
        self.device = torch.device("cpu")

    def test_validation_observes_each_update_and_restores_training_mode(self):
        observed = []

        def observe(stats):
            loss, _ = evaluate_loss(self.model, self.batches, self.device)
            observed.append((stats.step, loss))
            self.assertTrue(self.model.training)

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=0.01)
        stats = train_loader_steps(self.model, self.batches, optimizer, 3,
                                   self.device, on_step=observe)
        self.assertEqual([step for step, _ in observed], [1, 2, 3])
        self.assertGreater(observed[0][1], observed[-1][1])
        final_loss, _ = evaluate_loss(self.model, self.batches, self.device)
        self.assertAlmostEqual(observed[-1][1], final_loss)
        self.assertEqual(sum(item.tokens for item in stats), 3 * 2 * 7)

    def test_evaluation_preserves_eval_mode(self):
        self.model.eval()
        evaluate_loss(self.model, self.batches, self.device)
        self.assertFalse(self.model.training)

    def test_evaluation_restores_mode_on_failure(self):
        self.model.train()
        with self.assertRaises(ValueError):
            evaluate_loss(self.model, [torch.ones(2, dtype=torch.long)], self.device)
        self.assertTrue(self.model.training)


if __name__ == "__main__":
    unittest.main()
