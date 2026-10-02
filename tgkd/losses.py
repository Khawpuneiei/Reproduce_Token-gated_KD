"""Distillation losses on flattened reply positions.

Per-token functions take logits of shape (n, V) for the n supervised positions
only (the trainer never materialises full-sequence logits) and return (n,).
Gated losses are normalised by the gate sum so gating changes the *mix* of
tokens, not the effective learning rate.

TAID is ported from SakanaAI/TAID src/distil_losses/taid.py: logit-space
interpolation p_t = softmax((1-t)*stopgrad(z_s) + t*z_t), cross-entropy to the
student, and the momentum-adaptive monotone schedule for t.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def kl_per_token(student_logits: torch.Tensor, teacher_logits: torch.Tensor) -> torch.Tensor:
    """KL(p_teacher || q_student) for each row."""
    log_q = F.log_softmax(student_logits.float(), dim=-1)
    log_p = F.log_softmax(teacher_logits.float(), dim=-1)
    return (log_p.exp() * (log_p - log_q)).sum(dim=-1)


def ce_per_token(student_logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(student_logits.float(), targets, reduction="none")


def weighted_mean(per_token: torch.Tensor, weights: torch.Tensor | None) -> torch.Tensor:
    if weights is None:
        return per_token.mean()
    w = weights.detach().float()
    return (per_token * w).sum() / w.sum().clamp(min=1e-8)


class TAID:
    def __init__(self, t_start: float = 0.4, t_end: float = 1.0, alpha: float = 5e-4, beta: float = 0.99):
        assert 0.0 <= t_start < 1.0 and 0.0 < t_end <= 1.0 and 0.0 <= alpha <= 1.0
        self.t_start, self.t_end, self.alpha, self.beta = t_start, t_end, alpha, beta
        self.t = t_start
        self.prev_loss = math.inf
        self.momentum = 0.0

    def per_token(self, student_logits: torch.Tensor, teacher_logits: torch.Tensor) -> torch.Tensor:
        z_s = student_logits.float()
        p_t = torch.softmax((1 - self.t) * z_s.detach() + self.t * teacher_logits.float(), dim=-1)
        return -(p_t * F.log_softmax(z_s, dim=-1)).sum(dim=-1)

    def update(self, loss: float, step: int, total_steps: int) -> float:
        """Original update_t; called once per optimizer step with the step's mean loss."""
        if math.isinf(self.prev_loss):
            self.prev_loss = loss
            return 0.0
        rel = (self.prev_loss - loss) / (self.prev_loss + 1e-15)
        self.momentum = self.beta * self.momentum + (1 - self.beta) * rel
        adaptive = 1.0 / (1.0 + math.exp(-self.momentum))
        target = self.t_start + (self.t_end - self.t_start) * (step / total_steps)
        delta = self.alpha * adaptive * (1 - self.t)
        self.t = min(self.t_end, max(target, self.t + delta))
        self.prev_loss = loss
        return delta
