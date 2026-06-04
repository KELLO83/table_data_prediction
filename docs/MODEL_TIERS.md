# Model Tiers

## Tier 0: Serious Tree Baselines

LightGBM is the first serious baseline for generic tabular regression.

Reason:

- strong default performance
- fast on small and medium tables
- robust with mixed feature scales

## Tier 1: Alternative GBDT

CatBoost is useful when categorical columns are important or when LightGBM is unstable.

## Tier 2: Foundation / In-Context

TabPFN v3 is used as a pretrained tabular foundation model.

Rules:

- do not train from scratch
- use CPU for quick checks if GPU is unnecessary
- ensure token/checkpoint access is available before unattended runs

## Tier 3: Neural / Transformer Models

Use only after the CSV contract and baseline metrics are stable.

These models are more sensitive to sample size, scaling, batch size, and overfitting.

Available names:

```text
realmlp, tabm, tabr, dcnv2, node,
ft_transformer, tab_transformer, tabnet
```

## Additional Foundation Model

TabICLv2 is exposed through the model registry. Runtime depends on package/checkpoint availability.
