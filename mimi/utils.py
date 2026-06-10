"""Internal helpers: embeddings, cosine similarity, logging, rich console."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
from rich.console import Console
from rich.theme import Theme

if TYPE_CHECKING:
    from transformers import PreTrainedModel, PreTrainedTokenizerBase

# Shared rich console used across the package for user-facing output.
console = Console(
    theme=Theme(
        {
            "info": "bold blue",
            "success": "bold green",
            "warning": "bold yellow",
            "error": "bold red",
            "dim": "dim white",
        }
    )
)


def get_logger(name: str) -> logging.Logger:
    """Return a module-level logger that writes to stderr."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
        )
        logger.addHandler(handler)
        logger.setLevel(logging.WARNING)
    return logger


def compute_embeddings(
    model: "PreTrainedModel",
    tokenizer: "PreTrainedTokenizerBase",
    text: str,
    device: str = "cpu",
    max_length: int = 256,
) -> torch.Tensor:
    """
    Compute a single embedding vector for *text* by mean-pooling the last hidden state.

    Works with both plain HuggingFace causal LMs and PEFT-wrapped models.
    The returned tensor is always on CPU and in float32.
    """
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
        padding=True,
    ).to(device)

    with torch.no_grad():
        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            output_hidden_states=True,
            return_dict=True,
        )

    # Last transformer layer hidden states: (batch, seq_len, hidden_dim)
    last_hidden: torch.Tensor = outputs.hidden_states[-1]

    # Mean-pool over non-padding token positions.
    mask = inputs["attention_mask"].unsqueeze(-1).float()  # (batch, seq_len, 1)
    pooled = (last_hidden.float() * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)

    return pooled.squeeze(0).cpu()


def cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    """Return cosine similarity between two 1-D embedding vectors, in range [-1, 1]."""
    return F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()


def safe_model_name(model_name: str) -> str:
    """Convert a HuggingFace model name to a filesystem-safe string."""
    return model_name.replace("/", "--").replace("\\", "--").replace(":", "-")
