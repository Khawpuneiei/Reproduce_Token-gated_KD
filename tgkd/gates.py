"""GRACE-style confidence gates from teacher entropy.

g_i = max(g_min, exp(-H_i / (tau * log|V|)))

tau = 1 is the literal proposal formula. Because H_i <= log|V|, it is already
bounded below by e^-1, but on low-entropy math tokens it stays close to 1; tau < 1
sharpens it. g_min keeps a floor so high-entropy "pivot" tokens are never fully
suppressed (EGRSD pitfall).
"""

from __future__ import annotations

import math

import torch


def entropy_from_logits(logits: torch.Tensor) -> torch.Tensor:
    """Shannon entropy (nats) over the last dim, numerically stable in fp32.

    H = logsumexp(z) - sum softmax(z) * z avoids 0 * log 0 terms.
    """
    z = logits.float()
    lse = torch.logsumexp(z, dim=-1)
    p = torch.softmax(z, dim=-1)
    # p == 0 where z == -inf; mask so 0 * -inf does not become NaN
    h = lse - torch.where(p > 0, p * z, torch.zeros_like(z)).sum(dim=-1)
    return h.clamp(min=0.0, max=math.log(z.shape[-1]))


def token_gate(entropy: torch.Tensor, vocab_size: int, tau: float = 1.0, g_min: float = 0.1) -> torch.Tensor:
    if tau <= 0:
        raise ValueError("tau must be positive")
    g = torch.exp(-entropy / (tau * math.log(vocab_size)))
    return g.clamp(min=g_min, max=1.0)


def sample_gate(entropy: torch.Tensor, mask: torch.Tensor, vocab_size: int, tau: float = 1.0,
                g_min: float = 0.1) -> torch.Tensor:
    """One gate per sequence from its mean reply-token entropy, broadcast to tokens.

    entropy, mask: (batch, seq). Returns (batch, seq).
    """
    m = mask.float()
    mean_h = (entropy * m).sum(dim=1) / m.sum(dim=1).clamp(min=1.0)
    g = token_gate(mean_h, vocab_size, tau=tau, g_min=g_min)
    return g[:, None].expand_as(entropy)


def tau_for_median_gate(entropies: torch.Tensor, vocab_size: int, target: float = 0.5) -> float:
    """tau such that the median token's (unfloored) gate equals `target`."""
    med = float(entropies.float().median())
    return med / (-math.log(target) * math.log(vocab_size))
