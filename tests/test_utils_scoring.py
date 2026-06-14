"""Tests for compute_log_likelihood in pyrecall.utils."""

from __future__ import annotations

import math
from unittest.mock import MagicMock

import pytest
import torch

from pyrecall.utils import compute_log_likelihood


def _make_model_mock(loss_value: float) -> MagicMock:
    output = MagicMock()
    output.loss = torch.tensor(loss_value)
    model = MagicMock()
    model.return_value = output
    return model


def _make_slow_tokenizer_mock(prompt_len: int = 3, total_len: int = 6) -> MagicMock:
    """Mock tokenizer without is_fast=True — uses separate-tokenisation fallback (2 calls)."""
    prompt_ids = torch.ones(1, prompt_len, dtype=torch.long)
    full_ids = torch.ones(1, total_len, dtype=torch.long)
    tok = MagicMock()
    tok.is_fast = False  # forces fallback path
    tok.side_effect = [
        {"input_ids": prompt_ids, "attention_mask": torch.ones(1, prompt_len)},
        {"input_ids": full_ids, "attention_mask": torch.ones(1, total_len)},
    ]
    return tok


def _make_fast_tokenizer_mock(prompt_len: int = 3, total_len: int = 6) -> MagicMock:
    """Mock tokenizer with is_fast=True — uses offset_mapping path (1 call).

    Offsets: first prompt_len tokens cover chars [0, 6], rest cover [6+, ...].
    prompt = "prompt" → len 6.
    """
    full_ids = torch.ones(1, total_len, dtype=torch.long)

    chars_per_prompt_tok = 2  # 3 tokens × 2 chars = 6 chars (prompt)
    chars_per_comp_tok = 2
    offsets = []
    for i in range(prompt_len):
        offsets.append([i * chars_per_prompt_tok, (i + 1) * chars_per_prompt_tok])
    for i in range(total_len - prompt_len):
        start = 6 + i * chars_per_comp_tok
        offsets.append([start, start + chars_per_comp_tok])

    tok = MagicMock()
    tok.is_fast = True
    tok.return_value = {
        "input_ids": full_ids,
        "attention_mask": torch.ones(1, total_len),
        "offset_mapping": torch.tensor([offsets]),
    }
    return tok


class TestComputeLogLikelihood:
    """Tests for the compute_log_likelihood helper."""

    def test_returns_float_in_zero_one(self) -> None:
        model = _make_model_mock(loss_value=1.0)
        tokenizer = _make_slow_tokenizer_mock()
        score = compute_log_likelihood(model, tokenizer, "prompt", "completion")
        assert 0.0 < score <= 1.0

    def test_lower_nll_yields_higher_score(self) -> None:
        """exp(-low_NLL) > exp(-high_NLL)."""
        model_good = _make_model_mock(loss_value=0.5)
        model_bad = _make_model_mock(loss_value=3.0)
        tok_good = _make_slow_tokenizer_mock()
        tok_bad = _make_slow_tokenizer_mock()
        score_good = compute_log_likelihood(model_good, tok_good, "p", "c")
        score_bad = compute_log_likelihood(model_bad, tok_bad, "p", "c")
        assert score_good > score_bad

    def test_zero_nll_returns_one(self) -> None:
        model = _make_model_mock(loss_value=0.0)
        tokenizer = _make_slow_tokenizer_mock()
        score = compute_log_likelihood(model, tokenizer, "prompt", "completion")
        assert score == pytest.approx(1.0)

    def test_prompt_tokens_masked_in_labels_slow_path(self) -> None:
        """Fallback (slow tokenizer): first prompt_len labels must be -100."""
        captured_labels: list[torch.Tensor] = []

        output = MagicMock()
        output.loss = torch.tensor(1.0)

        def fake_model(**kwargs: object) -> MagicMock:
            captured_labels.append(kwargs["labels"].clone())  # type: ignore[arg-type]
            return output

        prompt_ids = torch.tensor([[10, 20, 30]])
        full_ids = torch.tensor([[10, 20, 30, 40, 50]])
        tok = MagicMock()
        tok.is_fast = False
        tok.side_effect = [
            {"input_ids": prompt_ids, "attention_mask": torch.ones(1, 3)},
            {"input_ids": full_ids, "attention_mask": torch.ones(1, 5)},
        ]

        compute_log_likelihood(fake_model, tok, "prompt", "completion")  # type: ignore[arg-type]
        labels = captured_labels[0]
        assert (labels[0, :3] == -100).all(), "Prompt tokens must be masked"
        assert (labels[0, 3:] != -100).all(), "Completion tokens must not be masked"

    def test_prompt_tokens_masked_in_labels_fast_path(self) -> None:
        """Fast tokenizer (offset mapping): prompt tokens are masked using char offsets (#99)."""
        captured_labels: list[torch.Tensor] = []

        output = MagicMock()
        output.loss = torch.tensor(1.0)

        def fake_model(**kwargs: object) -> MagicMock:
            captured_labels.append(kwargs["labels"].clone())  # type: ignore[arg-type]
            return output

        full_ids = torch.tensor([[10, 20, 30, 40, 50]])
        # prompt = "prompt" → 6 chars. Offsets: tokens 0-2 end at ≤6, tokens 3-4 end at >6.
        offsets = torch.tensor([[[0, 2], [2, 4], [4, 6], [6, 8], [8, 10]]])
        tok = MagicMock()
        tok.is_fast = True
        tok.return_value = {
            "input_ids": full_ids,
            "attention_mask": torch.ones(1, 5),
            "offset_mapping": offsets,
        }

        compute_log_likelihood(fake_model, tok, "prompt", "completion")  # type: ignore[arg-type]
        labels = captured_labels[0]
        assert (labels[0, :3] == -100).all(), "Prompt tokens (end ≤ 6) must be masked"
        assert (labels[0, 3:] != -100).all(), "Completion tokens must not be masked"

    def test_all_tokens_masked_returns_nan(self) -> None:
        """When prompt fills the entire sequence, no completion to score → return nan (#100)."""
        model = _make_model_mock(loss_value=0.0)
        full_ids = torch.ones(1, 3, dtype=torch.long)
        # All 3 tokens end at ≤ 6 (prompt_char_len=6), so prompt_len = 3 → all masked.
        offsets = torch.tensor([[[0, 2], [2, 4], [4, 6]]])
        tok = MagicMock()
        tok.is_fast = True
        tok.return_value = {
            "input_ids": full_ids,
            "attention_mask": torch.ones(1, 3),
            "offset_mapping": offsets,
        }
        score = compute_log_likelihood(model, tok, "prompt", "completion")
        assert math.isnan(score), "All-masked input must return NaN"

    def test_score_is_deterministic(self) -> None:
        model = _make_model_mock(loss_value=2.0)
        tok1 = _make_slow_tokenizer_mock()
        tok2 = _make_slow_tokenizer_mock()
        s1 = compute_log_likelihood(model, tok1, "x", "y")
        model.reset_mock()
        model.return_value.loss = torch.tensor(2.0)
        s2 = compute_log_likelihood(model, tok2, "x", "y")
        assert s1 == pytest.approx(s2)

    def test_high_nll_yields_near_zero_score(self) -> None:
        model = _make_model_mock(loss_value=20.0)
        tokenizer = _make_slow_tokenizer_mock()
        score = compute_log_likelihood(model, tokenizer, "p", "c")
        assert score < 0.01

    def test_fast_tokenizer_uses_single_call(self) -> None:
        """Fast tokenizer path should make exactly one tokenizer call (#99)."""
        model = _make_model_mock(loss_value=1.0)
        tok = _make_fast_tokenizer_mock(prompt_len=3, total_len=6)
        compute_log_likelihood(model, tok, "prompt", "completion")
        assert tok.call_count == 1, "Fast path must use a single tokenizer call"


class TestScoringMethodInSkillScore:
    """Tests for the scoring_method field on SkillScore."""

    def test_default_scoring_method_is_log_likelihood(self) -> None:
        from pyrecall.snapshot import SkillScore

        s = SkillScore(category="c", prompt="p", response="r", score=0.5)
        assert s.scoring_method == "log_likelihood"

    def test_to_dict_includes_scoring_method(self) -> None:
        from pyrecall.snapshot import SkillScore

        s = SkillScore(category="c", prompt="p", response="r", score=0.5, scoring_method="cosine")
        d = s.to_dict()
        assert d["scoring_method"] == "cosine"

    def test_from_dict_reads_scoring_method(self) -> None:
        from pyrecall.snapshot import SkillScore

        d = {
            "category": "c",
            "prompt": "p",
            "response": "r",
            "score": 0.5,
            "scoring_method": "cosine",
        }
        s = SkillScore.from_dict(d)
        assert s.scoring_method == "cosine"

    def test_from_dict_defaults_to_cosine_for_old_snapshots(self) -> None:
        from pyrecall.snapshot import SkillScore

        d = {"category": "c", "prompt": "p", "response": "r", "score": 0.5}
        s = SkillScore.from_dict(d)
        assert s.scoring_method == "cosine"
