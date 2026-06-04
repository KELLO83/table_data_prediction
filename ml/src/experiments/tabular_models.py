from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from ml.src.models.base import BaseModel
from ml.src.models.registry import create_model

ROOT = Path(__file__).resolve().parents[3]


def build_model(
    model_name: str,
    params: dict[str, Any],
    feature_set_name: str,
) -> BaseModel:
    if model_name == "tabpfn":
        load_tabpfn_token()
    return create_model(model_name, feature_set=feature_set_name, params=params)


def load_tabpfn_token() -> None:
    if os.environ.get("TABPFN_TOKEN"):
        return
    token_path = ROOT / ".secrets" / "tabpfn_token"
    if token_path.exists():
        token = token_path.read_text(encoding="utf-8-sig").strip().lstrip("\ufeff")
        if token:
            os.environ["TABPFN_TOKEN"] = token
