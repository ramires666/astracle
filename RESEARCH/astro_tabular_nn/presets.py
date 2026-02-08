"""Named presets for recommended astro tabular NN runs."""

from __future__ import annotations

from dataclasses import replace

from .config import ModelConfig, TrainConfig


# Recommended from best-grid post-fix broad trial (2026-02-08).
TRIAL_TOP_RECALL_MODEL = ModelConfig(
    model_type="dcn",
    hidden_dims=(512, 256, 128),
    cross_layers=2,
    cross_rank=32,
    embed_dim=32,
    dropout=0.25,
    activation="gelu",
)

TRIAL_TOP_RECALL_TRAIN = replace(
    TrainConfig(),
    learning_rate=5e-4,
    weight_decay=5e-4,
    class_weight_power=1.4,
    label_smoothing=0.05,
    batch_size=768,
)


def recommended_trial_preset() -> tuple[ModelConfig, TrainConfig]:
    """Return model+train preset from latest broad trial leader."""
    return TRIAL_TOP_RECALL_MODEL, TRIAL_TOP_RECALL_TRAIN
