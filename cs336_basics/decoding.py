from __future__ import annotations

import torch

from .model import TransformerLM, softmax
from .tokenizer import Tokenizer


@torch.no_grad()
def generate(
    model: TransformerLM,
    tokenizer: Tokenizer,
    prompt: str,
    max_new_tokens: int = 256,
    temperature: float = 1.0,
    top_p: float = 1.0,
    eos_token: str = "<|endoftext|>",
) -> str:
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if not 0 < top_p <= 1:
        raise ValueError("top_p must be in (0, 1]")

    ids = tokenizer.encode(prompt)
    if not ids:
        raise ValueError("prompt must produce at least one token")
    eos_id = tokenizer.token_to_id.get(eos_token.encode("utf-8"))
    device = next(model.parameters()).device
    was_training = model.training
    model.eval()

    for _ in range(max_new_tokens):
        context = ids[-model.context_length :]
        input_ids = torch.tensor(context, dtype=torch.long, device=device).unsqueeze(0)
        logits = model(input_ids)[0, -1] / temperature
        probabilities = softmax(logits, dim=-1)
        sorted_probabilities, sorted_indices = torch.sort(probabilities, descending=True)
        cumulative = sorted_probabilities.cumsum(dim=-1)
        remove = cumulative > top_p
        remove[1:] = remove[:-1].clone()
        remove[0] = False
        sorted_probabilities = sorted_probabilities.masked_fill(remove, 0)
        sorted_probabilities /= sorted_probabilities.sum()
        next_index = torch.multinomial(sorted_probabilities, num_samples=1)
        next_id = sorted_indices[next_index].item()
        ids.append(next_id)
        if next_id == eos_id:
            break

    if was_training:
        model.train()
    return tokenizer.decode(ids)
