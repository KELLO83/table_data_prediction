from __future__ import annotations

from typing import Final

EXPERIMENT_MODEL_GROUPS: Final[dict[str, tuple[str, ...]]] = {
    "gbdt_traditional_ml": (
        "lightgbm",
        "catboost",
    ),
    "neural_tabular": (
        "realmlp",
        "tabm",
        "tabr",
        "dcnv2",
        "node",
    ),
    "transformer_attention": (
        "ft_transformer",
        "tab_transformer",
        "tabnet",
    ),
    "pretrained_foundation": (
        "tabpfn",
        "tabiclv2",
    ),
}

EXPERIMENT_MODEL_NAMES: Final[tuple[str, ...]] = tuple(
    model_name
    for model_group in EXPERIMENT_MODEL_GROUPS.values()
    for model_name in model_group
)

SANITY_MODEL_NAMES: Final[tuple[str, ...]] = (
    "dummy_mean",
    "dummy_median",
    "ridge",
)

GBDT_MODEL_NAMES: Final[tuple[str, ...]] = EXPERIMENT_MODEL_GROUPS["gbdt_traditional_ml"]
NEURAL_MODEL_NAMES: Final[tuple[str, ...]] = EXPERIMENT_MODEL_GROUPS["neural_tabular"]
TRANSFORMER_MODEL_NAMES: Final[tuple[str, ...]] = EXPERIMENT_MODEL_GROUPS["transformer_attention"]
FOUNDATION_MODEL_NAMES: Final[tuple[str, ...]] = EXPERIMENT_MODEL_GROUPS["pretrained_foundation"]

ALL_SUPPORTED_MODEL_NAMES: Final[tuple[str, ...]] = SANITY_MODEL_NAMES + EXPERIMENT_MODEL_NAMES
