"""Utilities for enabling PyTorch SDPA in wrapped tabular torch models."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import logging
from types import MethodType
from typing import Any


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class SdpaPatchReport:
    """Result of attempting to patch torch attention modules for SDPA."""

    enabled: bool
    patched_attention_modules: int = 0
    inspected_torch_modules: int = 0
    reason: str = ""

    @classmethod
    def disabled(cls, reason: str) -> "SdpaPatchReport":
        return cls(enabled=False, reason=reason)

    def merge(self, other: "SdpaPatchReport") -> "SdpaPatchReport":
        return SdpaPatchReport(
            enabled=self.enabled or other.enabled,
            patched_attention_modules=self.patched_attention_modules + other.patched_attention_modules,
            inspected_torch_modules=self.inspected_torch_modules + other.inspected_torch_modules,
            reason=self.reason or other.reason,
        )


@dataclass(frozen=True)
class SdpaBackendLogSpec:
    module: Any
    batch_size: int
    tokens: int
    dtype: Any
    device: str
    training: bool
    label: str


@dataclass(frozen=True)
class SdpaBackendChoice:
    module: Any
    query: Any
    key: Any
    value: Any
    args: tuple[Any, ...]
    kwargs: dict[str, Any]
    label: str


def apply_sdpa_to_model(module: Any) -> SdpaPatchReport:
    """Patch all ``nn.MultiheadAttention`` children to default to SDPA.

    PyTorch's ``nn.MultiheadAttention`` uses ``scaled_dot_product_attention``
    when attention weights are not requested. Several third-party tabular
    transformer blocks call MHA as ``self.attn(x, x, x)``, which keeps the
    PyTorch default ``need_weights=True`` and can bypass SDPA. This patch keeps
    explicit caller choices intact while making the no-weights path the default.
    """

    torch, functional = _torch_with_sdpa()
    if torch is None or functional is None:
        return SdpaPatchReport.disabled("torch SDPA is not available")
    if not isinstance(module, torch.nn.Module):
        return SdpaPatchReport.disabled("object is not a torch.nn.Module")

    patched = 0
    for child in module.modules():
        if isinstance(child, torch.nn.MultiheadAttention) and not getattr(child, "_secanday_sdpa_patched", False):
            _patch_multihead_attention(child)
            patched += 1
    return SdpaPatchReport(
        enabled=True,
        patched_attention_modules=patched,
        inspected_torch_modules=1,
    )


def apply_sdpa_to_estimator(estimator: Any) -> SdpaPatchReport:
    """Find torch modules inside an estimator or pipeline and patch MHA modules."""

    torch, functional = _torch_with_sdpa()
    if torch is None or functional is None:
        return SdpaPatchReport.disabled("torch SDPA is not available")

    report = SdpaPatchReport(enabled=False)
    visited: set[int] = set()
    stack: list[tuple[Any, int]] = [(estimator, 0)]

    while stack:
        obj, depth = stack.pop()
        if obj is None or id(obj) in visited or depth > 5:
            continue
        visited.add(id(obj))

        if isinstance(obj, torch.nn.Module):
            report = report.merge(apply_sdpa_to_model(obj))
            continue

        for child in _iter_children(obj, torch):
            stack.append((child, depth + 1))

    return report if report.inspected_torch_modules else SdpaPatchReport.disabled("no torch.nn.Module was found")


def maybe_apply_sdpa(target: Any, enabled: bool = True) -> SdpaPatchReport:
    """Apply SDPA patching when enabled, returning a report either way."""

    if not enabled:
        return SdpaPatchReport.disabled("SDPA patching is disabled")
    return apply_sdpa_to_estimator(target)


def log_sdpa_backend_for_model(spec: SdpaBackendLogSpec) -> None:
    """Log expected SDPA backend for all MHA modules using representative metadata."""

    torch, functional = _torch_with_sdpa()
    if torch is None or functional is None or not isinstance(spec.module, torch.nn.Module):
        LOGGER.info("%s SDPA backend: unavailable", spec.label)
        return
    for index, child in enumerate(child for child in spec.module.modules() if isinstance(child, torch.nn.MultiheadAttention)):
        shape = (
            (spec.batch_size, spec.tokens, int(child.embed_dim))
            if bool(getattr(child, "batch_first", False))
            else (spec.tokens, spec.batch_size, int(child.embed_dim))
        )
        query = torch.empty(shape, device=spec.device, dtype=spec.dtype)
        was_training = child.training
        child.train(spec.training)
        _log_sdpa_backend_choice(
            SdpaBackendChoice(
                module=child,
                query=query,
                key=query,
                value=query,
                args=(),
                kwargs={"need_weights": False},
                label=f"{spec.label}.mha{index}",
            )
        )
        child.train(was_training)
        child._secanday_sdpa_backend_logged = True


def _patch_multihead_attention(module: Any) -> None:
    original_forward = module.forward

    def forward_with_sdpa_default(self: Any, query: Any, key: Any, value: Any, *args: Any, **kwargs: Any) -> Any:
        if len(args) < 2 and "need_weights" not in kwargs:
            kwargs["need_weights"] = False
        need_weights = _resolve_need_weights(args, kwargs)
        torch, _ = _torch_with_sdpa()
        if not need_weights and not getattr(self, "_secanday_sdpa_backend_logged", False) and not _torch_is_compiling(torch):
            _log_sdpa_backend_choice(SdpaBackendChoice(self, query, key, value, args, kwargs, type(self).__name__))
            self._secanday_sdpa_backend_logged = True
        return original_forward(query, key, value, *args, **kwargs)

    module._secanday_original_forward = original_forward
    module._secanday_sdpa_patched = True
    module._secanday_sdpa_backend_logged = False
    module.forward = MethodType(forward_with_sdpa_default, module)


def _resolve_need_weights(args: tuple[Any, ...], kwargs: dict[str, Any]) -> bool:
    if "need_weights" in kwargs:
        return bool(kwargs["need_weights"])
    if len(args) >= 2:
        return bool(args[1])
    return True


def _log_sdpa_backend_choice(choice: SdpaBackendChoice) -> None:
    torch, functional = _torch_with_sdpa()
    if torch is None or functional is None:
        LOGGER.info("%s SDPA backend: unavailable; torch SDPA is not present.", choice.label)
        return
    if not getattr(choice.query, "is_cuda", False):
        LOGGER.info(
            "%s SDPA backend: expected=math device=%s dtype=%s reason=non-CUDA query",
            choice.label,
            getattr(choice.query, "device", "unknown"),
            getattr(choice.query, "dtype", "unknown"),
        )
        return

    batch, q_tokens = _batch_and_tokens(choice.module, choice.query)
    _, k_tokens = _batch_and_tokens(choice.module, choice.key)
    heads = int(getattr(choice.module, "num_heads", 1))
    head_dim = int(getattr(choice.module, "head_dim", choice.query.shape[-1] // max(heads, 1)))
    dropout_p = float(getattr(choice.module, "dropout", 0.0)) if bool(getattr(choice.module, "training", False)) else 0.0
    is_causal = bool(choice.kwargs.get("is_causal", choice.args[4] if len(choice.args) >= 5 else False))

    representative_query = torch.empty((batch, heads, q_tokens, head_dim), device=choice.query.device, dtype=choice.query.dtype)
    representative_key = torch.empty((batch, heads, k_tokens, head_dim), device=choice.key.device, dtype=choice.key.dtype)
    representative_value = torch.empty((batch, heads, k_tokens, head_dim), device=choice.value.device, dtype=choice.value.dtype)
    params = torch.backends.cuda.SDPAParams(
        representative_query,
        representative_key,
        representative_value,
        None,
        dropout_p,
        is_causal,
        False,
    )
    candidates = _sdpa_candidates(torch, params)
    expected = _expected_sdpa_backend(torch, candidates)
    LOGGER.info(
        "%s SDPA backend: expected=%s candidates=%s priority=%s shape=(batch=%s, heads=%s, q_tokens=%s, k_tokens=%s, head_dim=%s) dtype=%s dropout=%s causal=%s",
        choice.label,
        expected,
        candidates,
        _sdp_priority_order(torch),
        batch,
        heads,
        q_tokens,
        k_tokens,
        head_dim,
        choice.query.dtype,
        dropout_p,
        is_causal,
    )


def _batch_and_tokens(module: Any, tensor: Any) -> tuple[int, int]:
    if tensor.dim() < 3:
        return 1, int(tensor.shape[0])
    if bool(getattr(module, "batch_first", False)):
        return int(tensor.shape[0]), int(tensor.shape[1])
    return int(tensor.shape[1]), int(tensor.shape[0])


def _sdpa_candidates(torch: Any, params: Any) -> dict[str, bool]:
    cuda = torch.backends.cuda
    return {
        "flash": bool(cuda.flash_sdp_enabled() and cuda.can_use_flash_attention(params, debug=False)),
        "efficient": bool(cuda.mem_efficient_sdp_enabled() and cuda.can_use_efficient_attention(params, debug=False)),
        "cudnn": bool(cuda.cudnn_sdp_enabled() and cuda.can_use_cudnn_attention(params, debug=False)),
        "math": bool(cuda.math_sdp_enabled()),
    }


def _expected_sdpa_backend(torch: Any, candidates: dict[str, bool]) -> str:
    name_by_value = {
        int(torch.backends.cuda.SDPBackend.FLASH_ATTENTION): "flash",
        int(torch.backends.cuda.SDPBackend.EFFICIENT_ATTENTION): "efficient",
        int(torch.backends.cuda.SDPBackend.MATH): "math",
        int(torch.backends.cuda.SDPBackend.CUDNN_ATTENTION): "cudnn",
    }
    for value in torch._C._get_sdp_priority_order():
        name = name_by_value.get(int(value))
        if name and candidates.get(name, False):
            return name
    return "unavailable"


def _sdp_priority_order(torch: Any) -> list[str]:
    name_by_value = {
        int(torch.backends.cuda.SDPBackend.FLASH_ATTENTION): "flash",
        int(torch.backends.cuda.SDPBackend.EFFICIENT_ATTENTION): "efficient",
        int(torch.backends.cuda.SDPBackend.MATH): "math",
        int(torch.backends.cuda.SDPBackend.CUDNN_ATTENTION): "cudnn",
        int(torch.backends.cuda.SDPBackend.OVERRIDEABLE): "overrideable",
    }
    return [name_by_value.get(int(value), str(value)) for value in torch._C._get_sdp_priority_order()]


def _torch_is_compiling(torch: Any) -> bool:
    if torch is None:
        return False
    compiler = getattr(torch, "compiler", None)
    is_compiling = getattr(compiler, "is_compiling", None)
    if callable(is_compiling):
        return bool(is_compiling())
    dynamo = getattr(torch, "_dynamo", None)
    is_compiling = getattr(dynamo, "is_compiling", None)
    return bool(is_compiling()) if callable(is_compiling) else False


def _torch_with_sdpa() -> tuple[Any | None, Any | None]:
    try:
        import torch
        import torch.nn.functional as functional
    except ImportError:
        return None, None
    if not hasattr(functional, "scaled_dot_product_attention"):
        return None, None
    return torch, functional


def _iter_children(obj: Any, torch: Any) -> list[Any]:
    children: list[Any] = []

    if isinstance(obj, Mapping):
        children.extend(obj.values())
        return children

    if isinstance(obj, Sequence) and not isinstance(obj, (str, bytes, bytearray)):
        children.extend(obj)
        return children

    named_steps = getattr(obj, "named_steps", None)
    if isinstance(named_steps, Mapping):
        children.extend(named_steps.values())

    for attr in (
        "model_",
        "model",
        "network_",
        "network",
        "module_",
        "module",
        "executor_",
        "estimator",
        "estimator_",
    ):
        try:
            value = getattr(obj, attr)
        except Exception:
            continue
        if value is not obj:
            children.append(value)

    try:
        values = vars(obj).values()
    except TypeError:
        values = ()
    for value in values:
        if isinstance(value, torch.nn.Module):
            children.append(value)
        elif _looks_like_model_container(value):
            children.append(value)

    return children


def _looks_like_model_container(value: Any) -> bool:
    if value is None or isinstance(value, (str, bytes, bytearray, int, float, bool)):
        return False
    if isinstance(value, (Mapping, Sequence)) and not isinstance(value, (str, bytes, bytearray)):
        return True
    module_name = type(value).__module__.lower()
    return any(
        token in module_name
        for token in (
            "deepctr_torch",
            "pytabkit",
            "pytorch_tabnet",
            "pytorch_tabular",
            "tabicl",
            "tabpfn",
            "tabtransformer",
            "fttransformer",
        )
    )
