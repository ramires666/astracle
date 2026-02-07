"""
=============================================================================
MODEL EVALUATION UTILITIES FOR RESEARCH
=============================================================================

This module provides comprehensive model evaluation tools with:
- Confusion matrix visualization (heatmap with percentages)
- Predicted vs True labels comparison plot (time series)
- Full metrics report (accuracy, precision, recall, F1, MCC)

WHY THIS MODULE?
----------------
After training a model, we need to understand HOW WELL it performs.
Just looking at accuracy is not enough - we need to see:
1. Does the model confuse UP with DOWN? (confusion matrix)
2. When does it make mistakes? (time series plot)
3. Is it balanced between classes? (per-class metrics)

USAGE EXAMPLE:
--------------
    from RESEARCH.evaluation import evaluate_model_full
    
    metrics = evaluate_model_full(
        y_true=y_test,           # Actual labels [0, 1, 0, 1, ...]
        y_pred=y_pred,           # Predicted labels
        dates=test_dates,        # Dates for time plot (optional)
        title="My Model",        # Title for plots
        show_plot=True           # Display plots
    )
    
    print(f"Recall MIN: {metrics['recall_min']}")

=============================================================================
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    recall_score,
    precision_score,
    matthews_corrcoef,
    classification_report,
    confusion_matrix,
)

from .evaluation_stats import compute_accuracy_significance


# =============================================================================
# MAIN EVALUATION FUNCTION
# =============================================================================

def evaluate_model_full(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray] = None,
    dates: Optional[pd.Series] = None,
    title: str = "Model Evaluation",
    label_names: List[str] = ["DOWN", "UP"],
    figsize: Tuple[int, int] = (14, 5),
    show_plot: bool = True,
    prices: Optional[pd.Series] = None,
    random_baseline: float = 0.5,
    price_color: str = "#1f77b4",
    up_color: str = "green",
    down_color: str = "red",
    shade_alpha: float = 0.20,
) -> Dict:
    """
    Comprehensive model evaluation with visualizations.
    
    THIS FUNCTION DOES THREE THINGS:
    --------------------------------
    1. Calculates ALL important metrics (accuracy, recall, F1, MCC, etc.)
    2. Creates a CONFUSION MATRIX heatmap (shows what model confuses)
    3. Creates a TIME SERIES plot (shows predictions over time)
    
    PARAMETERS EXPLAINED:
    ---------------------
    y_true : np.ndarray
        The ACTUAL labels from test data. Example: [0, 1, 0, 0, 1, 1]
        These are the "ground truth" - what actually happened.
        
    y_pred : np.ndarray
        What the MODEL predicted. Example: [0, 1, 1, 0, 1, 0]
        We compare this to y_true to see how accurate the model is.
        
    y_proba : np.ndarray, optional
        Probability scores from model (0.0 to 1.0).
        Not used for metrics, but can be useful for analysis.
        
    dates : pd.Series, optional
        Dates corresponding to each prediction.
        If provided, we create a time series plot showing
        predictions vs actual values over time.
        
    title : str
        Title for the plots. Example: "XGBoost Model Evaluation"
        
    label_names : List[str]
        Names for classes. Default ["DOWN", "UP"] for binary classification.
        Class 0 = DOWN (price went down)
        Class 1 = UP (price went up)
        
    figsize : Tuple[int, int]
        Size of the figure in inches. Default (14, 5).
        
    show_plot : bool
        Whether to display plots. Set False in grid search to save time.
    
    RETURNS:
    --------
    Dict with all calculated metrics:
        - accuracy: Overall percentage of correct predictions
        - balanced_accuracy: Average of recall for each class (handles imbalance)
        - f1_macro: Harmonic mean of precision and recall (averaged)
        - mcc: Matthews Correlation Coefficient (-1 to 1, 0 = random)
        - recall_down, recall_up: How many DOWN/UP cases we correctly identified
        - recall_min: The WORSE of the two recalls (our target metric!)
        - recall_gap: Difference between recalls (should be small for balance)
        - f1_down, f1_up, f1_min, f1_gap: Same but for F1 score
        - precision_down, precision_up: Precision per class
        - p_value_vs_random: One-sided p-value vs random baseline (0.5 by default)
        - accuracy_ci95_low / accuracy_ci95_high: Wilson 95% CI for accuracy
    
    WHY WE USE RECALL_MIN AS TARGET:
    ---------------------------------
    Recall = "Of all actual DOWNs, how many did we predict correctly?"
    
    If recall_down=0.8 and recall_up=0.4, the model is biased toward DOWN.
    Taking MIN(0.8, 0.4) = 0.4 forces us to optimize the WEAKER class.
    
    A model with recall_min=0.6 is BETTER than one with
    recall_down=0.9 and recall_up=0.3 (recall_min=0.3).
    """
    
    # -------------------------------------------------------------------------
    # STEP 1: Calculate basic metrics
    # -------------------------------------------------------------------------
    # Accuracy = correct predictions / total predictions
    acc = accuracy_score(y_true, y_pred)
    
    # Balanced accuracy = average of per-class recall
    # This is better than accuracy when classes are imbalanced
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    
    # F1 macro = average F1 across classes
    # F1 = harmonic mean of precision and recall
    f1_macro = f1_score(y_true, y_pred, average='macro')
    
    # MCC = Matthews Correlation Coefficient
    # Ranges from -1 (total disagreement) to +1 (perfect prediction)
    # 0 means random prediction. This is considered the best single metric.
    mcc = matthews_corrcoef(y_true, y_pred)
    
    # -------------------------------------------------------------------------
    # STEP 2: Calculate per-class metrics using classification_report
    # -------------------------------------------------------------------------
    # classification_report gives us precision, recall, f1 for each class
    report = classification_report(
        y_true, y_pred,
        labels=[0, 1],              # Our classes: 0=DOWN, 1=UP
        target_names=label_names,   # Human-readable names
        output_dict=True,           # Return as dictionary (not string)
        zero_division=0             # Return 0 if division by zero
    )
    
    # Extract recall (also called "sensitivity" or "true positive rate")
    # Recall = TP / (TP + FN)
    # "Of all actual DOWNs, how many did we correctly predict as DOWN?"
    recall_down = report[label_names[0]]["recall"]  # Recall for class 0
    recall_up = report[label_names[1]]["recall"]    # Recall for class 1
    
    # Our KEY METRIC: the minimum of the two recalls
    # This ensures both classes are predicted well
    recall_min = min(recall_down, recall_up)
    
    # Gap shows how "balanced" the model is
    # Large gap = model favors one class over another
    recall_gap = abs(recall_down - recall_up)
    
    # Same analysis but for F1 score
    f1_down = report[label_names[0]]["f1-score"]
    f1_up = report[label_names[1]]["f1-score"]
    f1_min = min(f1_down, f1_up)
    f1_gap = abs(f1_down - f1_up)
    
    # -------------------------------------------------------------------------
    # STEP 3: Build metrics dictionary
    # -------------------------------------------------------------------------
    metrics = {
        "accuracy": acc,
        "balanced_accuracy": bal_acc,
        "f1_macro": f1_macro,
        "mcc": mcc,
        "recall_down": recall_down,
        "recall_up": recall_up,
        "recall_min": recall_min,
        "recall_gap": recall_gap,
        "f1_down": f1_down,
        "f1_up": f1_up,
        "f1_min": f1_min,
        "f1_gap": f1_gap,
        "precision_down": report[label_names[0]]["precision"],
        "precision_up": report[label_names[1]]["precision"],
    }

    # -------------------------------------------------------------------------
    # STEP 3.1: Statistical significance vs random baseline
    # -------------------------------------------------------------------------
    significance = compute_accuracy_significance(
        y_true=y_true,
        y_pred=y_pred,
        random_baseline=random_baseline,
    )
    metrics.update(significance)
    
    # -------------------------------------------------------------------------
    # STEP 4: Print metrics summary
    # -------------------------------------------------------------------------
    print("=" * 60)
    print(f"📊 {title}")
    print("=" * 60)
    print(f"Accuracy:          {acc:.4f}")
    print(f"Balanced Acc:      {bal_acc:.4f}")
    print(f"F1 Macro:          {f1_macro:.4f}")
    print(f"MCC:               {mcc:.4f}")
    print("-" * 40)
    print(f"R_DOWN: {recall_down:.4f}  |  R_UP: {recall_up:.4f}  |  R_MIN: {recall_min:.4f}  |  GAP: {recall_gap:.4f}")
    print(f"F1_DOWN: {f1_down:.4f} |  F1_UP: {f1_up:.4f} |  F1_MIN: {f1_min:.4f} |  GAP: {f1_gap:.4f}")
    print("-" * 40)
    print(
        f"Significance vs random({random_baseline:.2f}): "
        f"p-value={significance['p_value_vs_random']:.6g}"
    )
    print(
        f"Accuracy 95% CI (Wilson): "
        f"[{significance['accuracy_ci95_low']:.4f}, {significance['accuracy_ci95_high']:.4f}]"
    )
    print("=" * 60)
    
    # -------------------------------------------------------------------------
    # STEP 5: Create visualizations (if requested)
    # -------------------------------------------------------------------------
    if show_plot:
        # Determine number of plots and layout
        if dates is not None and prices is not None:
            # 3 plots total: 1 CM on top, 2 Time Series below? 
            # Or just 2 rows: Top=CM, Bottom=Time Series (Predicted & True sharing axis?)
            # Let's do:
            # Row 1: Confusion Matrix
            # Row 2: Predicted Labels (background color)
            # Row 3: True Labels (background color)
            
            fig = plt.figure(figsize=(12, 12))
            gs = fig.add_gridspec(3, 1, height_ratios=[1, 1, 1])
            
            ax_cm = fig.add_subplot(gs[0])
            ax_pred = fig.add_subplot(gs[1])
            ax_true = fig.add_subplot(gs[2], sharex=ax_pred)
            
            axes_list = [ax_cm, ax_pred, ax_true]
            
        elif dates is not None:
             # Fallback: CM + Scatter
            fig, axes = plt.subplots(1, 2, figsize=figsize)
            ax_cm = axes[0]
            ax_ts = axes[1]
            axes_list = [ax_cm, ax_ts]
            
        else:
            # Just CM
            fig, ax_cm = plt.subplots(1, 1, figsize=(6, 5))
            axes_list = [ax_cm]

        # ---------------------------------------------------------------------
        # PLOT 1: Confusion Matrix Heatmap (Always on ax_cm)
        # ---------------------------------------------------------------------
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        cm_pct = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
        
        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=label_names, yticklabels=label_names,
            ax=ax_cm, cbar=False
        )
        
        for i in range(2):
            for j in range(2):
                ax_cm.text(j + 0.5, i + 0.7, f"({cm_pct[i,j]:.1f}%)",
                        ha='center', va='center', fontsize=9, color='gray')
        
        ax_cm.set_xlabel('Predicted')
        ax_cm.set_ylabel('True')
        ax_cm.set_title(f'Confusion Matrix\nR_MIN={recall_min:.3f}, MCC={mcc:.3f}')
        
        # ---------------------------------------------------------------------
        # PLOT 2 & 3: Time Series
        # ---------------------------------------------------------------------
        if dates is not None and prices is not None:
            # --- Predicted Labels ---
            title_metrics = f"PREDICTED (R_DN={recall_down:.2f} R_UP={recall_up:.2f} MCC={mcc:.3f} ACC={acc:.3f})"
            ax_pred.set_title(title_metrics, color='black')
            # Use BLUE/BLACK for price on white background
            ax_pred.plot(dates, prices, color=price_color, linewidth=1.5, label='Price')
            
            p_min, p_max = prices.min(), prices.max()
            margin = (p_max - p_min) * 0.05
            fill_min, fill_max = p_min - margin, p_max + margin
            
            # Standard colors work well on white
            ax_pred.fill_between(dates, fill_min, fill_max, where=(y_pred==1), 
                           color=up_color, alpha=shade_alpha, label='UP')
            ax_pred.fill_between(dates, fill_min, fill_max, where=(y_pred==0), 
                           color=down_color, alpha=shade_alpha, label='DOWN')
            ax_pred.set_ylabel("Price")
            ax_pred.grid(True, alpha=0.3, color='gray', linestyle=':')
            
            # --- True Labels ---
            ax_true.set_title("TRUE Labels (Green=UP, Red=DOWN)", color='black')
            ax_true.plot(dates, prices, color=price_color, linewidth=1.5, label='Price')
            
            ax_true.fill_between(dates, fill_min, fill_max, where=(y_true==1), 
                           color=up_color, alpha=shade_alpha, label='UP')
            ax_true.fill_between(dates, fill_min, fill_max, where=(y_true==0), 
                           color=down_color, alpha=shade_alpha, label='DOWN')
            ax_true.set_ylabel("Price")
            ax_true.set_xlabel("Date")
            ax_true.grid(True, alpha=0.3, color='gray', linestyle=':')
            
        elif dates is not None:
            # Fallback Scatter Plot
            df_plot = pd.DataFrame({
                'date': dates.values if hasattr(dates, 'values') else dates,
                'true': y_true,
                'pred': y_pred,
            }).sort_values('date')
            
            ax_ts.plot(df_plot['date'], df_plot['true'], 'b-', alpha=0.3, label='True')
            ax_ts.plot(df_plot['date'], df_plot['pred'], 'r--', alpha=0.3, label='Pred')
            
            correct = df_plot['true'] == df_plot['pred']
            ax_ts.scatter(df_plot.loc[correct, 'date'], df_plot.loc[correct, 'true'], c='g', s=10)
            ax_ts.scatter(df_plot.loc[~correct, 'date'], df_plot.loc[~correct, 'true'], c='r', s=10)
            ax_ts.set_title(f"Accuracy: {acc:.3f}")
            plt.setp(ax_ts.get_xticklabels(), rotation=45)

        plt.suptitle(title, fontsize=14, fontweight='bold', y=0.98, color='black')
        plt.tight_layout()
        plt.show()

    
    return metrics


# =============================================================================
# MODEL COMPARISON FUNCTION
# =============================================================================

def compare_models(
    results: List[Dict],
    names: List[str],
    metric: str = "recall_min",
    figsize: Tuple[int, int] = (10, 6),
) -> pd.DataFrame:
    """
    Compare multiple model results visually with a bar chart.
    
    USE CASE:
    ---------
    After running grid search with different configurations, you want
    to compare them side by side. This function creates a bar chart
    showing key metrics for each configuration.
    
    PARAMETERS:
    -----------
    results : List[Dict]
        List of metric dictionaries from evaluate_model_full().
        Each dict should have keys like 'recall_min', 'mcc', etc.
        
    names : List[str]
        Names for each model/configuration.
        Example: ["Baseline", "No MeanNode", "No Pluto"]
        
    metric : str
        Which metric to use for ranking. Default "recall_min".
        
    figsize : Tuple[int, int]
        Size of the figure in inches.
    
    RETURNS:
    --------
    pd.DataFrame with all models and their metrics side by side.
    
    EXAMPLE:
    --------
        results = [baseline_metrics, model1_metrics, model2_metrics]
        names = ["Baseline", "Model 1", "Model 2"]
        df = compare_models(results, names)
    """
    # Convert list of dicts to DataFrame
    df = pd.DataFrame(results, index=names)
    
    # Select key metrics for comparison (most important ones)
    key_metrics = ['recall_min', 'balanced_accuracy', 'mcc', 'f1_macro', 'recall_gap']
    df_plot = df[key_metrics]
    
    # Create bar chart
    fig, ax = plt.subplots(figsize=figsize)
    df_plot.plot(kind='bar', ax=ax, width=0.8)
    
    ax.set_ylabel('Score')
    ax.set_title('Model Comparison')
    ax.legend(loc='upper right')
    ax.set_xticklabels(names, rotation=45, ha='right')
    
    # Annotate best model
    best_idx = df[metric].idxmax()
    ax.annotate(f'Best: {best_idx}', xy=(0.02, 0.98), xycoords='axes fraction',
                fontsize=10, fontweight='bold', va='top')
    
    plt.tight_layout()
    plt.show()
    
    return df
