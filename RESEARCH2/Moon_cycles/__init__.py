"""Utilities for Moon-cycle-only research notebook."""

from .moon_data import (
    MoonLabelConfig,
    build_balanced_labels_for_gauss,
    build_moon_dataset_for_gauss,
    build_moon_phase_features,
    get_moon_feature_columns,
    load_market_slice,
)
from .ephemeris_data import EphemerisFeatureConfig, build_ephemeris_feature_set
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
from .trading_utils import (
    TradingConfig,
    backtest_long_flat_signals,
    build_signal_from_proba,
    plot_backtest_price_and_equity,
    sweep_trading_params,
)
from .trading_utils_short import backtest_long_short_signals
from .turning_astro_features import (
    TurningAstroFeatureConfig,
    build_transit_to_natal_feature_set,
    build_turning_astro_feature_set,
    classify_feature_group,
    summarize_feature_groups,
)
from .turning_targets import (
    build_point_only_targets,
    build_segment_midpoint_targets,
    build_turning_target_frame,
    build_window_kernel_targets,
    merge_features_with_turning_target,
)
from .turning_targets_numba import NUMBA_AVAILABLE

__all__ = [
    "MoonLabelConfig",
    "build_balanced_labels_for_gauss",
    "build_moon_dataset_for_gauss",
    "build_moon_phase_features",
    "EphemerisFeatureConfig",
    "build_ephemeris_feature_set",
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
    "TradingConfig",
    "backtest_long_flat_signals",
    "backtest_long_short_signals",
    "build_signal_from_proba",
    "plot_backtest_price_and_equity",
    "sweep_trading_params",
    "TurningAstroFeatureConfig",
    "build_transit_to_natal_feature_set",
    "build_turning_astro_feature_set",
    "classify_feature_group",
    "summarize_feature_groups",
    "build_point_only_targets",
    "build_segment_midpoint_targets",
    "build_turning_target_frame",
    "build_window_kernel_targets",
    "merge_features_with_turning_target",
    "NUMBA_AVAILABLE",
]
