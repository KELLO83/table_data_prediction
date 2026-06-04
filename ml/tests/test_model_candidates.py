from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

import train
from ml.scripts import smoke_model_wrappers
from ml.src.experiments import tabular_runner
from ml.src.models.candidates import (
    EXPERIMENT_MODEL_GROUPS,
    EXPERIMENT_MODEL_NAMES,
    FOUNDATION_MODEL_NAMES,
    GBDT_MODEL_NAMES,
    NEURAL_MODEL_NAMES,
    TRANSFORMER_MODEL_NAMES,
)


def test_experiment_model_candidates_exclude_sanity_wrappers() -> None:
    candidates = set(EXPERIMENT_MODEL_NAMES)

    assert len(EXPERIMENT_MODEL_NAMES) == 12
    assert {"dummy_mean", "dummy_median", "ridge"}.isdisjoint(candidates)
    assert {"lightgbm", "catboost", "tabm", "tabpfn", "tabiclv2"}.issubset(candidates)


def test_experiment_model_candidates_are_grouped_then_priority_ordered() -> None:
    assert EXPERIMENT_MODEL_GROUPS == {
        "gbdt_traditional_ml": ("lightgbm", "catboost"),
        "neural_tabular": ("realmlp", "tabm", "tabr", "dcnv2", "node"),
        "transformer_attention": ("ft_transformer", "tab_transformer", "tabnet"),
        "pretrained_foundation": ("tabpfn", "tabiclv2"),
    }
    assert EXPERIMENT_MODEL_NAMES == (
        GBDT_MODEL_NAMES
        + NEURAL_MODEL_NAMES
        + TRANSFORMER_MODEL_NAMES
        + FOUNDATION_MODEL_NAMES
    )


def test_smoke_script_defaults_to_experiment_model_candidates() -> None:
    assert smoke_model_wrappers.DEFAULT_MODELS == list(EXPERIMENT_MODEL_NAMES)


def test_train_cli_exposes_only_experiment_model_candidates() -> None:
    assert train.MODEL_CHOICES == list(EXPERIMENT_MODEL_NAMES)


def test_tabular_regression_runner_accepts_only_experiment_candidates() -> None:
    model_action = next(
        action
        for action in tabular_runner.build_parser()._actions
        if "--model" in action.option_strings
    )

    choices = model_action.choices

    assert choices is not None
    assert tuple(choices) == EXPERIMENT_MODEL_NAMES


def test_tabular_regression_script_stays_thin_cli_wrapper() -> None:
    script = Path("ml/scripts/run_tabular_regression.py")
    pure_lines = [
        line
        for line in script.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]

    assert len(pure_lines) <= 80


def test_train_entrypoint_uses_experiment_runner_directly() -> None:
    train_source = Path("train.py").read_text(encoding="utf-8")
    cli_source = Path("ml/scripts/run_tabular_regression.py").read_text(encoding="utf-8")

    assert "importlib" not in train_source
    assert "spec_from_file_location" not in train_source
    assert "from sklearn.model_selection import train_test_split" not in cli_source


def test_tabular_runner_passes_validation_split_to_model(monkeypatch, tmp_path) -> None:
    captured: dict[str, int] = {}

    class CapturingModel:
        def fit(
            self,
            x_train: pd.DataFrame,
            y_train: pd.Series,
            x_valid: pd.DataFrame | None = None,
            y_valid: pd.Series | None = None,
        ) -> None:
            assert x_valid is not None
            assert y_valid is not None
            captured["train_rows"] = len(x_train)
            captured["valid_rows"] = len(x_valid)
            captured["valid_target_rows"] = len(y_valid)

        def predict(self, x_valid: pd.DataFrame) -> list[float]:
            return [1.0] * len(x_valid)

    monkeypatch.setattr(
        tabular_runner,
        "read_csv",
        lambda _path: pd.DataFrame({"feature": [1, 2, 3, 4, 5], "target": [1, 1, 1, 1, 1]}),
    )
    monkeypatch.setattr(tabular_runner, "build_model", lambda **_kwargs: CapturingModel())
    args = argparse.Namespace(
        config=None,
        csv=tmp_path / "input.csv",
        target="target",
        model="lightgbm",
        features="",
        exclude="",
        categorical="",
        test_size=0.4,
        seed=42,
        output=tmp_path / "results.csv",
        save_predictions=False,
        prediction_dir=tmp_path / "predictions",
        device=None,
        model_param=[],
    )

    tabular_runner.run_experiment(args)

    assert captured == {"train_rows": 3, "valid_rows": 2, "valid_target_rows": 2}
