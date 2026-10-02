"""GSM8K formatting shared by training and evaluation.

Teacher and student share one tokenizer family, so every sequence is built once
with the student tokenizer and an explicit system prompt (the Instruct
template's default system prompt differs from the base template's).
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass

SYSTEM_PROMPT = "You are a helpful assistant."
INSTRUCTION = "Solve the problem step by step. Put the final numeric answer on the last line as '#### <number>'."
_CALC = re.compile(r"<<[^>]*>>")


def clean_solution(answer: str) -> str:
    """Strip GSM8K calculator annotations, keep the '#### n' final line."""
    return _CALC.sub("", answer).strip()


def gold_answer(answer: str) -> str:
    return answer.split("####")[-1].strip().replace(",", "")


def prompt_messages(question: str) -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"{question.strip()}\n{INSTRUCTION}"},
    ]


def prompt_text(tokenizer, question: str) -> str:
    return tokenizer.apply_chat_template(prompt_messages(question), tokenize=False, add_generation_prompt=True)


@dataclass
class Example:
    input_ids: list[int]
    response_mask: list[int]  # 1 where the token is part of the assistant reply (incl. <|im_end|>)


def encode_example(tokenizer, question: str, answer: str, max_length: int) -> Example:
    prompt_ids = tokenizer(prompt_text(tokenizer, question), add_special_tokens=False).input_ids
    reply_ids = tokenizer(clean_solution(answer) + "<|im_end|>", add_special_tokens=False).input_ids
    ids = (prompt_ids + reply_ids)[:max_length]
    mask = ([0] * len(prompt_ids) + [1] * len(reply_ids))[:max_length]
    return Example(ids, mask)


def select_train_rows(rows: list[dict], size: int, seed: int) -> list[dict]:
    """Deterministic subset + order shared by every arm."""
    order = list(range(len(rows)))
    random.Random(seed).shuffle(order)
    return [rows[i] for i in order[: min(size, len(rows))]]


def collate(examples: list[Example], pad_id: int) -> dict:
    """Right-pad. labels_mask[j] marks positions whose *next* token is a reply token."""
    import torch

    width = max(len(e.input_ids) for e in examples)
    ids = torch.full((len(examples), width), pad_id, dtype=torch.long)
    attn = torch.zeros((len(examples), width), dtype=torch.long)
    target_mask = torch.zeros((len(examples), width), dtype=torch.bool)
    for row, ex in enumerate(examples):
        n = len(ex.input_ids)
        ids[row, :n] = torch.tensor(ex.input_ids)
        attn[row, :n] = 1
        # position j predicts token j+1
        target_mask[row, : n - 1] = torch.tensor(ex.response_mask[1:], dtype=torch.bool)
    return {"input_ids": ids, "attention_mask": attn, "target_mask": target_mask}
