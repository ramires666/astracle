"""Configuration dataclasses for astro tabular neural experiments."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Sequence, Tuple


DEFAULT_DATASET = Path(
    "data/market/processed/"
    "btc_dataset_oracle_gauss_h1_pmraw_s1p0_thr100p000500_tmfixed__om1p00_geo.parquet"
)


@dataclass(frozen=True)
class DatasetConfig:
    """Dataset loading options."""

    dataset_path: Path = DEFAULT_DATASET
    date_col: str = "date"
    target_col: str = "target"
    drop_cols: Tuple[str, ...] = (
        "close",
        "next_ret",
        "turning_direction",
        "sample_weight",
        "target_mode",
        "event_index",
        "segment_index",
        "smoothed_close",
        "smooth_slope",
    )
    fillna_value: float = 0.0


@dataclass(frozen=True)
class SplitConfig:
    """Simple chronological split ratios."""

    train_ratio: float = 0.70
    val_ratio: float = 0.15


@dataclass(frozen=True)
class ModelConfig:
    """Tabular model settings (supports multiple architectures)."""

    model_type: str = "dcn"
    hidden_dims: Tuple[int, ...] = (384, 192, 96)
    cross_layers: int = 3
    cross_rank: int = 64
    embed_dim: int = 32
    dropout: float = 0.15
    activation: str = "gelu"


@dataclass(frozen=True)
class TrainConfig:
    """Training settings with CUDA-first defaults."""

    epochs: int = 24
    batch_size: int = 512
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    label_smoothing: float = 0.02
    class_weight_power: float = 1.2
    grad_clip_norm: float = 2.0
    early_stopping_patience: int = 5
    num_workers: int = 6
    amp_enabled: bool = True
    amp_dtype: str = "float16"
    compile_model: bool = False
    device: str = "cuda"
    seed: int = 42


@dataclass(frozen=True)
class ScoutConfig:
    """Small scout experiment settings used in notebook and scripts."""

    split: SplitConfig = field(default_factory=SplitConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    margin_grid: Tuple[float, ...] = (
        0.00,
        0.01,
        0.02,
        0.03,
        0.05,
        0.08,
        0.12,
        0.16,
        0.22,
    )
    threshold_grid: Tuple[float, ...] = (
        0.35,
        0.40,
        0.45,
        0.50,
        0.55,
        0.60,
        0.65,
    )
    margin_gap_penalty: float = 0.35
    margin_prior_penalty: float = 0.10
    cutoff_objective: str = "recall_balance"
    segment_score_gamma: float = 1.5
    segment_min_days: int = 5
    segment_include_open_tail: bool = True
    segment_metric: str = "weighted_hit_rate"
    top_k_for_bounds: int = 3


def as_path(path_like: Path | str) -> Path:
    """Normalize config path inputs."""
    return path_like if isinstance(path_like, Path) else Path(path_like)


def with_dataset(cfg: DatasetConfig, dataset_path: Path | str) -> DatasetConfig:
    """Return DatasetConfig with a new dataset path."""
    return replace(cfg, dataset_path=as_path(dataset_path))


def with_seed(train_cfg: TrainConfig, seed: int) -> TrainConfig:
    """Return TrainConfig with a specific random seed."""
    return replace(train_cfg, seed=int(seed))


def with_epochs(train_cfg: TrainConfig, epochs: int) -> TrainConfig:
    """Return TrainConfig with an overridden epoch count."""
    return replace(train_cfg, epochs=int(epochs))


def with_batch_size(train_cfg: TrainConfig, batch_size: int) -> TrainConfig:
    """Return TrainConfig with an overridden batch size."""
    return replace(train_cfg, batch_size=int(batch_size))


def with_model_dims(model_cfg: ModelConfig, hidden_dims: Sequence[int]) -> ModelConfig:
    """Return ModelConfig with modified MLP width/depth."""
    return replace(model_cfg, hidden_dims=tuple(int(v) for v in hidden_dims))


def with_dropout(model_cfg: ModelConfig, dropout: float) -> ModelConfig:
    """Return ModelConfig with a different dropout rate."""
    return replace(model_cfg, dropout=float(dropout))


def with_cross(model_cfg: ModelConfig, cross_layers: int, cross_rank: int) -> ModelConfig:
    """Return ModelConfig with modified cross network shape."""
    return replace(model_cfg, cross_layers=int(cross_layers), cross_rank=int(cross_rank))


def with_model_type(model_cfg: ModelConfig, model_type: str) -> ModelConfig:
    """Return ModelConfig with a different architecture type."""
    return replace(model_cfg, model_type=str(model_type))


def with_embed_dim(model_cfg: ModelConfig, embed_dim: int) -> ModelConfig:
    """Return ModelConfig with a different embedding width."""
    return replace(model_cfg, embed_dim=int(embed_dim))
