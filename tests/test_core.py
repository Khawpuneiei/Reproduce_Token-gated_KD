import math
import unittest

import torch

from tgkd.answers import (answer_confidence, expected_calibration_error, extract_answer, is_correct,
                          span_token_indices)
from tgkd.data import Example, clean_solution, collate, gold_answer, select_train_rows
from tgkd.gates import entropy_from_logits, sample_gate, tau_for_median_gate, token_gate
from tgkd.losses import TAID, kl_per_token, weighted_mean


class GateTests(unittest.TestCase):
    def test_entropy_uniform_and_onehot(self):
        v = 1000
        self.assertAlmostEqual(float(entropy_from_logits(torch.zeros(1, v))), math.log(v), places=4)
        peaked = torch.full((1, v), -1e4)
        peaked[0, 3] = 0
        self.assertAlmostEqual(float(entropy_from_logits(peaked)), 0.0, places=5)

    def test_entropy_stable_with_extreme_logits(self):
        z = torch.tensor([[1e4, -1e4, 0.0, float("-inf")]])
        h = entropy_from_logits(z)
        self.assertTrue(torch.isfinite(h).all())

    def test_literal_gate_bounds(self):
        v = 151665
        h = torch.tensor([0.0, math.log(v)])
        g = token_gate(h, v, tau=1.0, g_min=0.0)
        self.assertAlmostEqual(float(g[0]), 1.0)
        self.assertAlmostEqual(float(g[1]), math.exp(-1), places=5)

    def test_floor_applies(self):
        g = token_gate(torch.tensor([10.0]), 151665, tau=0.01, g_min=0.1)
        self.assertAlmostEqual(float(g), 0.1)

    def test_sample_gate_broadcasts_mean_over_mask(self):
        ent = torch.tensor([[1.0, 3.0, 99.0], [2.0, 2.0, 2.0]])
        mask = torch.tensor([[1, 1, 0], [1, 1, 1]], dtype=torch.bool)
        g = sample_gate(ent, mask, 100, tau=1.0, g_min=0.0)
        expect = math.exp(-2.0 / math.log(100))
        self.assertTrue(torch.allclose(g, torch.full_like(g, expect)))

    def test_tau_for_median(self):
        v = 1000
        ents = torch.tensor([0.1, 0.5, 2.0])
        tau = tau_for_median_gate(ents, v, 0.5)
        self.assertAlmostEqual(float(token_gate(torch.tensor([0.5]), v, tau, 0.0)), 0.5, places=5)


class LossTests(unittest.TestCase):
    def test_kl_zero_for_identical(self):
        z = torch.randn(4, 50)
        self.assertLess(float(kl_per_token(z, z).abs().max()), 1e-5)

    def test_weighted_mean_normalises_by_weight_sum(self):
        per = torch.tensor([1.0, 3.0])
        self.assertAlmostEqual(float(weighted_mean(per, torch.tensor([1.0, 0.0]))), 1.0)
        self.assertAlmostEqual(float(weighted_mean(per, None)), 2.0)

    def test_taid_target_at_t1_is_teacher(self):
        taid = TAID()
        taid.t = 1.0
        zs, zt = torch.randn(3, 20), torch.randn(3, 20)
        ce = taid.per_token(zs, zt)
        p = torch.softmax(zt, -1)
        expect = -(p * torch.log_softmax(zs, -1)).sum(-1)
        self.assertTrue(torch.allclose(ce, expect, atol=1e-5))

    def test_taid_schedule_monotone_and_reaches_end(self):
        taid = TAID(t_start=0.4, alpha=5e-4, beta=0.99)
        ts = []
        for step in range(1, 101):
            taid.update(10.0 / step, step, 100)
            ts.append(taid.t)
        self.assertTrue(all(b >= a for a, b in zip(ts, ts[1:])))
        self.assertAlmostEqual(ts[-1], 1.0)

    def test_taid_first_update_only_records_loss(self):
        taid = TAID()
        taid.update(5.0, 1, 10)
        self.assertEqual(taid.t, 0.4)
        self.assertEqual(taid.prev_loss, 5.0)


class AnswerTests(unittest.TestCase):
    def test_extract_prefers_hash(self):
        self.assertEqual(extract_answer("so 5 apples\n#### 1,234.")[0], "1234")
        self.assertEqual(extract_answer("total is 7 then 9")[0], "9")
        self.assertEqual(extract_answer("no numbers")[0], None)

    def test_is_correct_numeric(self):
        self.assertTrue(is_correct("18.0", "18"))
        self.assertFalse(is_correct(None, "18"))

    def test_span_tokens_and_confidence(self):
        pieces = ["####", " 1", "23", "\n"]
        text = "".join(pieces)
        _, span = extract_answer(text)
        idx = span_token_indices(pieces, span)
        self.assertEqual(idx, [1, 2])
        self.assertAlmostEqual(answer_confidence([0.0, math.log(0.5), math.log(0.5), 0.0], idx), 0.25)

    def test_ece(self):
        self.assertAlmostEqual(expected_calibration_error([1.0, 1.0], [True, True]), 0.0)
        self.assertAlmostEqual(expected_calibration_error([0.9, 0.9], [False, False]), 0.9)


class DataTests(unittest.TestCase):
    def test_clean_and_gold(self):
        self.assertEqual(clean_solution("a <<1+1=2>>2\n#### 2"), "a 2\n#### 2")
        self.assertEqual(gold_answer("x\n#### 1,000"), "1000")

    def test_collate_shifts_mask(self):
        b = collate([Example([1, 2, 3, 4], [0, 0, 1, 1]), Example([5, 6], [0, 1])], pad_id=0)
        self.assertEqual(b["target_mask"].tolist(), [[False, True, True, False], [True, False, False, False]])
        targets = b["input_ids"].roll(-1, dims=1)[b["target_mask"]].tolist()
        self.assertEqual(targets, [3, 4, 6])

    def test_subset_deterministic(self):
        rows = [{"i": i} for i in range(100)]
        self.assertEqual(select_train_rows(rows, 10, 0), select_train_rows(rows, 10, 0))
        self.assertNotEqual(select_train_rows(rows, 10, 0), select_train_rows(rows, 10, 1))


if __name__ == "__main__":
    unittest.main()
