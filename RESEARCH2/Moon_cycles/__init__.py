"""Utilities for Moon-cycle-only research notebook."""

from .moon_data import (
    MoonLabelConfig,
    build_balanced_labels_for_gauss,
    build_moon_dataset_for_gauss,
    build_moon_phase_features,
    get_moon_feature_columns,
    load_market_slice,
)
from .splits import (
    SplitDefinition,
    describe_split,
    describe_splits_table,
    make_classic_split,
    make_walk_forward_splits,
)
from .eval_utils import (
    compute_binary_metrics,
    compute_rolling_metrics,
    compute_statistical_significance,
    make_coin_flip_baseline,
    make_majority_baseline,
)
from .eval_visuals import VisualizationConfig, evaluate_with_visuals
from .threshold_utils import predict_proba_up_safe, tune_threshold_with_balance
from .bakeoff_utils import SkModelSpec, default_model_specs, run_moon_model_bakeoff
from .search_utils import (
    WalkForwardConfig,
    XgbConfig,
    evaluate_fixed_gauss,
    run_gauss_search,
)

__all__ = [
    "MoonLabelConfig",
    "build_balanced_labels_for_gauss",
    "build_moon_dataset_for_gauss",
    "build_moon_phase_features",
    "get_moon_feature_columns",
    "load_market_slice",
    "SplitDefinition",
    "describe_split",
    "describe_splits_table",
    "make_classic_split",
    "make_walk_forward_splits",
    "compute_binary_metrics",
    "compute_rolling_metrics",
    "compute_statistical_significance",
    "make_coin_flip_baseline",
    "make_majority_baseline",
    "VisualizationConfig",
    "evaluate_with_visuals",
    "predict_proba_up_safe",
    "tune_threshold_with_balance",
    "SkModelSpec",
    "default_model_specs",
    "run_moon_model_bakeoff",
    "WalkForwardConfig",
    "XgbConfig",
    "evaluate_fixed_gauss",
    "run_gauss_search",
]
