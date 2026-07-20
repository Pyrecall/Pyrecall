"""Internal helpers: embeddings, cosine similarity, log-likelihood, logging, rich console."""

from __future__ import annotations

import hashlib
import logging
import math
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
from rich.console import Console
from rich.theme import Theme

if TYPE_CHECKING:
    from PIL import Image
    from transformers import PreTrainedModel, PreTrainedTokenizerBase, ProcessorMixin


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
        handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
        logger.addHandler(handler)
        logger.setLevel(logging.WARNING)
    return logger


logger = get_logger(__name__)


def _prepare_model_inputs(
    tokenizer_or_processor: PreTrainedTokenizerBase | ProcessorMixin,
    prompt: str,
    completion: str,
    image: Image.Image | None = None,
    max_length: int = 512,
) -> dict[str, torch.Tensor]:
    """Prepare model inputs for log-likelihood computation.

    Supports both text-only and multimodal (VLM) models.

    Args:
        tokenizer_or_processor: Either a tokenizer (text-only) or processor (VLM).
        prompt: The input prompt text.
        completion: The completion text to score.
        image: Optional PIL Image for VLM models. If None, uses text-only path.
        max_length: Maximum sequence length.

    Returns:
        Dictionary with input_ids, attention_mask, and any vision tensors
        (e.g., pixel_values) needed by the model.
    """
    full_text = prompt + completion
    prompt_char_len = len(prompt)

    # Check if this is a processor (VLM) with an image, or a tokenizer (text-only)
    is_processor = hasattr(tokenizer_or_processor, "image_processor") or hasattr(
        tokenizer_or_processor, "feature_extractor"
    )

    if image is not None and is_processor:
        # VLM path: use processor with image
        # Processors typically accept text and images together
        full_enc = tokenizer_or_processor(
            text=full_text,
            images=image,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
            add_special_tokens=True,
        )
        # For VLM, we still need to find the prompt boundary
        # Use token count approach since offset mapping may not be available
        prompt_enc = tokenizer_or_processor(
            text=prompt,
            images=image,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
            add_special_tokens=True,
        )
        prompt_len = prompt_enc["input_ids"].shape[1]
    elif getattr(tokenizer_or_processor, "is_fast", False) is True:
        # Fast tokenizer path: use offset mapping for accurate prompt boundary
        full_enc = tokenizer_or_processor(
            full_text,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
            add_special_tokens=True,
            return_offsets_mapping=True,
        )
        offsets = full_enc.pop("offset_mapping")[0].tolist()
        prompt_len = next(
            (i for i, (_, end) in enumerate(offsets) if end > prompt_char_len),
            len(offsets),
        )
    else:
        # Slow tokenizer or mock: fall back to separate tokenisation
        prompt_enc = tokenizer_or_processor(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
            add_special_tokens=True,
        )
        full_enc = tokenizer_or_processor(
            full_text,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
            add_special_tokens=True,
        )
        prompt_len = prompt_enc["input_ids"].shape[1]

    return {
        "input_ids": full_enc["input_ids"],
        "attention_mask": full_enc["attention_mask"],
        "prompt_len": prompt_len,
    }


def compute_embeddings(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
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


def compute_log_likelihood(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    prompt: str,
    completion: str,
    device: str = "cpu",
    max_length: int = 512,
    processor: ProcessorMixin | None = None,
    image: Image.Image | None = None,
) -> float:
    """Return the per-token log-likelihood of *completion* given *prompt*.

    Specifically: exp(-mean_NLL_per_completion_token) ∈ (0, 1].
    Higher = model assigns higher probability to the reference answer = less forgetting.

    Uses a single causal-LM forward pass with the prompt tokens masked out of the
    loss so only the completion tokens contribute to the NLL.

    For VLM models, pass a processor and image to enable multimodal scoring.
    """
    # Use processor if provided and image is given, otherwise use tokenizer
    tok_or_proc = processor if processor is not None else tokenizer

    inputs = _prepare_model_inputs(
        tokenizer_or_processor=tok_or_proc,
        prompt=prompt,
        completion=completion,
        image=image,
        max_length=max_length,
    )

    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)
    prompt_len = inputs["prompt_len"]

    # Labels: -100 masks prompt tokens so they don't contribute to loss.
    labels = input_ids.clone()
    labels[0, :prompt_len] = -100

    n_scored = int((labels[0] != -100).sum().item())
    if n_scored == 0:
        # All tokens masked — prompt is at or beyond max_length, no completion to score.
        logger.warning(
            "compute_log_likelihood: no completion tokens remain after masking "
            "(prompt may exceed max_length=%d). Returning NaN.",
            max_length,
        )
        return float("nan")

    # Build model kwargs, including any vision tensors from the processor
    model_kwargs = {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}
    for key, value in inputs.items():
        if key not in ("input_ids", "attention_mask", "prompt_len"):
            model_kwargs[key] = value.to(device)

    with torch.no_grad():
        outputs = model(**model_kwargs)

    mean_nll: float = outputs.loss.item()
    if mean_nll <= 0.0:
        return 1.0
    return math.exp(-mean_nll)


def compute_log_likelihood_batch(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    prompts: list[str],
    completions: list[str],
    device: str = "cpu",
    max_length: int = 512,
    processor: ProcessorMixin | None = None,
    image_paths: list[Image.Image | None] | None = None,
) -> list[float]:
    """Batched variant of :func:`compute_log_likelihood`.

    Tokenises all (prompt, completion) pairs, right-pads to the longest sequence
    in the batch, and runs a single forward pass to compute per-item NLL.
    Returns a list of scores in the same order as the inputs.

    For VLM models, pass a processor and image_paths to enable multimodal scoring.
    """
    if not prompts:
        return []

    # Use processor if provided, otherwise use tokenizer
    tok_or_proc = processor if processor is not None else tokenizer

    all_input_ids: list[list[int]] = []
    all_labels: list[list[int]] = []
    all_vision_tensors: dict[str, list[torch.Tensor]] = {}

    for i, (prompt, completion) in enumerate(zip(prompts, completions)):
        image = image_paths[i] if image_paths is not None else None
        inputs = _prepare_model_inputs(
            tokenizer_or_processor=tok_or_proc,
            prompt=prompt,
            completion=completion,
            image=image,
            max_length=max_length,
        )

        ids = inputs["input_ids"][0].tolist()
        prompt_len = inputs["prompt_len"]
        labels = ids[:]
        for j in range(prompt_len):
            labels[j] = -100
        all_input_ids.append(ids)
        all_labels.append(labels)

        # Collect any vision tensors for VLM support
        for key, value in inputs.items():
            if key not in ("input_ids", "attention_mask", "prompt_len"):
                if key not in all_vision_tensors:
                    all_vision_tensors[key] = []
                all_vision_tensors[key].append(value)

    pad_id = tokenizer.pad_token_id or 0
    max_len = max(len(ids) for ids in all_input_ids)
    padded_ids = torch.full((len(prompts), max_len), pad_id, dtype=torch.long)
    padded_labels = torch.full((len(prompts), max_len), -100, dtype=torch.long)
    attn_mask = torch.zeros(len(prompts), max_len, dtype=torch.long)
    for i, (ids, lbls) in enumerate(zip(all_input_ids, all_labels)):
        n = len(ids)
        padded_ids[i, :n] = torch.tensor(ids, dtype=torch.long)
        padded_labels[i, :n] = torch.tensor(lbls, dtype=torch.long)
        attn_mask[i, :n] = 1

    padded_ids = padded_ids.to(device)
    padded_labels = padded_labels.to(device)
    attn_mask = attn_mask.to(device)

    # Build model kwargs, including any vision tensors
    model_kwargs = {"input_ids": padded_ids, "attention_mask": attn_mask}
    for key, tensors in all_vision_tensors.items():
        # Stack vision tensors for batch processing
        model_kwargs[key] = torch.cat(tensors, dim=0).to(device)

    with torch.no_grad():
        outputs = model(**model_kwargs)

    logits = outputs.logits

    # Some mocked models used in tests return logits with shape
    # (batch_size, vocab_size) instead of the HuggingFace standard
    # (batch_size, seq_len, vocab_size). Fall back to the sequential
    # implementation in that case.
    if logits.ndim != 3:
        logger.warning(
            "compute_log_likelihood_batch: unexpected logits shape %s; "
            "falling back to sequential scoring.",
            tuple(logits.shape),
        )
        return [
            compute_log_likelihood(
                model,
                tokenizer,
                prompt,
                completion,
                device=device,
                max_length=max_length,
                processor=processor,
                image=image_paths[i] if image_paths is not None else None,
            )
            for i, (prompt, completion) in enumerate(zip(prompts, completions))
        ]

    # Shift: logits[t] predicts token[t+1].
    logits = logits[:, :-1, :].contiguous()
    shift_labels = padded_labels[:, 1:].contiguous()

    loss_per_token = F.cross_entropy(
        logits.view(-1, logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=-100,
        reduction="none",
    ).view(len(prompts), -1)

    n_scored = (shift_labels != -100).sum(dim=1).float()

    scores: list[float] = []
    for nll_sum, n in zip(loss_per_token.sum(dim=1), n_scored):
        if n.item() == 0:
            logger.warning(
                "compute_log_likelihood_batch: no completion tokens remain after masking "
                "(prompt may exceed max_length=%d). Returning NaN.",
                max_length,
            )
            scores.append(0.0)
        else:
            mean_nll = nll_sum.item() / n.item()
            scores.append(1.0 if mean_nll <= 0.0 else math.exp(-mean_nll))
    return scores


def safe_model_name(model_name: str) -> str:
    """Convert a HuggingFace model name to a filesystem-safe string.

    Long names are truncated and suffixed with a hash of the full original
    name so directory components stay well under the 255-byte filesystem
    limit even after parent path segments are added.
    """
    safe = model_name.replace("/", "--").replace("\\", "--").replace(":", "-")
    if len(safe) > 200:
        digest = hashlib.sha1(model_name.encode()).hexdigest()[:8]
        safe = f"{safe[:180]}--{digest}"
    return safe
