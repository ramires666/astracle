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
try:
    # Optional dependency: visual reports require seaborn/matplotlib.
    from .eval_visuals import VisualizationConfig, evaluate_with_visuals
except ModuleNotFoundError as exc:
    if str(exc.name) not in {"seaborn", "matplotlib"}:
        raise

    class VisualizationConfig:  # type: ignore[override]
        """Placeholder when visualization dependencies are not installed."""

        def __init__(self, *args, **kwargs) -> None:
            raise ModuleNotFoundError(
                "VisualizationConfig requires matplotlib and seaborn. "
                "Install with: pip install matplotlib seaborn"
            )

    def evaluate_with_visuals(*args, **kwargs):  # type: ignore[override]
        """Placeholder when visualization dependencies are not installed."""
        raise ModuleNotFoundError(
            "evaluate_with_visuals requires matplotlib and seaborn. "
            "Install with: pip install matplotlib seaborn"
        )
from .threshold_utils import predict_proba_up_safe, tune_threshold_with_balance
from .bakeoff_utils import SkModelSpec, default_model_specs, run_moon_model_bakeoff
from .search_utils import (
    WalkForwardConfig,
    XgbConfig,
    evaluate_fixed_gauss,
    run_gauss_search,
)
try:
    # Optional dependency: trading plots require matplotlib.
    from .trading_utils import (
        TradingConfig,
        backtest_long_flat_signals,
        build_signal_from_proba,
        plot_backtest_price_and_equity,
        sweep_trading_params,
    )
except ModuleNotFoundError as exc:
    if str(exc.name) != "matplotlib":
        raise

    class TradingConfig:  # type: ignore[override]
        """Placeholder when matplotlib is not installed."""

        def __init__(self, *args, **kwargs) -> None:
            raise ModuleNotFoundError(
                "TradingConfig requires matplotlib. "
                "Install with: pip install matplotlib"
            )

    def backtest_long_flat_signals(*args, **kwargs):  # type: ignore[override]
        """Placeholder when matplotlib is not installed."""
        raise ModuleNotFoundError(
            "backtest_long_flat_signals requires matplotlib. "
            "Install with: pip install matplotlib"
        )

    def build_signal_from_proba(*args, **kwargs):  # type: ignore[override]
        """Placeholder when matplotlib is not installed."""
        raise ModuleNotFoundError(
            "build_signal_from_proba requires matplotlib. "
            "Install with: pip install matplotlib"
        )

    def plot_backtest_price_and_equity(*args, **kwargs):  # type: ignore[override]
        """Placeholder when matplotlib is not installed."""
        raise ModuleNotFoundError(
            "plot_backtest_price_and_equity requires matplotlib. "
            "Install with: pip install matplotlib"
        )

    def sweep_trading_params(*args, **kwargs):  # type: ignore[override]
        """Placeholder when matplotlib is not installed."""
        raise ModuleNotFoundError(
            "sweep_trading_params requires matplotlib. "
            "Install with: pip install matplotlib"
        )
try:
    # Optional dependency: short strategy helpers rely on trading_utils/matplotlib.
    from .trading_utils_short import backtest_long_short_signals
except ModuleNotFoundError as exc:
    if str(exc.name) != "matplotlib":
        raise

    def backtest_long_short_signals(*args, **kwargs):  # type: ignore[override]
        """Placeholder when matplotlib is not installed."""
        raise ModuleNotFoundError(
            "backtest_long_short_signals requires matplotlib. "
            "Install with: pip install matplotlib"
        )
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
