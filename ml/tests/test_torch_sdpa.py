from __future__ import annotations

import torch
import torch.nn.functional as F

from ml.src.models.torch_sdpa import apply_sdpa_to_estimator, apply_sdpa_to_model


def test_apply_sdpa_to_model_defaults_multihead_attention_to_sdpa(monkeypatch) -> None:
    calls = []
    original_sdpa = F.scaled_dot_product_attention

    def spy_scaled_dot_product_attention(*args, **kwargs):
        calls.append((args, kwargs))
        return original_sdpa(*args, **kwargs)

    monkeypatch.setattr(F, "scaled_dot_product_attention", spy_scaled_dot_product_attention)

    attention = torch.nn.MultiheadAttention(embed_dim=8, num_heads=2, dropout=0.0, batch_first=True)
    report = apply_sdpa_to_model(attention)

    x = torch.randn(3, 4, 8, requires_grad=True)
    output, weights = attention(x, x, x)

    assert report.enabled
    assert report.patched_attention_modules == 1
    assert output.shape == x.shape
    assert weights is None
    assert calls


def test_apply_sdpa_to_model_preserves_explicit_attention_weights(monkeypatch) -> None:
    calls = []
    original_sdpa = F.scaled_dot_product_attention

    def spy_scaled_dot_product_attention(*args, **kwargs):
        calls.append((args, kwargs))
        return original_sdpa(*args, **kwargs)

    monkeypatch.setattr(F, "scaled_dot_product_attention", spy_scaled_dot_product_attention)

    attention = torch.nn.MultiheadAttention(embed_dim=8, num_heads=2, dropout=0.0, batch_first=True)
    apply_sdpa_to_model(attention)

    x = torch.randn(3, 4, 8, requires_grad=True)
    output, weights = attention(x, x, x, need_weights=True)

    assert output.shape == x.shape
    assert weights is not None
    assert not calls


def test_apply_sdpa_to_estimator_finds_nested_torch_modules() -> None:
    class Estimator:
        def __init__(self) -> None:
            self.model_ = torch.nn.Sequential(
                torch.nn.MultiheadAttention(embed_dim=8, num_heads=2, dropout=0.0, batch_first=True)
            )

    estimator = Estimator()
    report = apply_sdpa_to_estimator(estimator)

    assert report.enabled
    assert report.patched_attention_modules == 1
    assert report.inspected_torch_modules == 1
