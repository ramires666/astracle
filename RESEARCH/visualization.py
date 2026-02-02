"""
Visualization module for RESEARCH pipeline.
All plotting functions in one place.
"""
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import Optional, List, Dict
from sklearn.metrics import confusion_matrix

# Set default style
sns.set_theme(style="whitegrid")
plt.rcParams["figure.figsize"] = (12, 4)


def plot_price_distribution(
    df_market: pd.DataFrame,
    price_mode: str = "log",
    figsize: tuple = (12, 6),
):
    """
    Plot price and daily return distribution.
    
    Args:
        df_market: Market DataFrame with date and close columns
        price_mode: 'log' or 'raw'
        figsize: Figure size
    """
    fig, ax = plt.subplots(2, 1, figsize=figsize, sharex=False)
    
    if price_mode == "log":
        price_series = np.log(df_market["close"])
        price_label = "log(close)"
    else:
        price_series = df_market["close"]
        price_label = "close"
    
    ax[0].plot(df_market["date"], price_series, color="tab:blue", linewidth=1)
    ax[0].set_title("BTC close (daily)")
    ax[0].set_xlabel("Date")
    ax[0].set_ylabel(price_label)
    
    log_ret = np.log(df_market["close"]).diff().dropna()
    ax[1].hist(log_ret, bins=80, color="tab:gray")
    ax[1].set_title("Daily log return distribution")
    ax[1].set_xlabel("log_return")
    ax[1].set_ylabel("frequency")
    
    plt.tight_layout()
    plt.show()


def plot_class_distribution(
    df_labels: pd.DataFrame,
    figsize: tuple = (6, 4),
):
    """Plot class distribution of labels."""
    label_map = {0: "DOWN", 1: "UP"}
    counts = df_labels["target"].value_counts(normalize=True).sort_index() * 100
    colors = ["#d62728", "#2ca02c"]
    
    plt.figure(figsize=figsize)
    plt.bar([label_map[i] for i in counts.index], counts.values, color=colors)
    plt.title("Class share (balanced labels)")
    plt.ylabel("%")
    plt.show()


def shade_up_down(
    ax,
    dates,
    close,
    up_mask,
    down_mask,
    title: str,
    y_label: str,
):
    """Helper: shade UP/DOWN zones on price chart."""
    ax.plot(dates, close, color="black", linewidth=1.2, label="Price")
    ax.fill_between(
        dates, 0, 1, where=up_mask,
        transform=ax.get_xaxis_transform(),
        color="green", alpha=0.15, label="UP"
    )
    ax.fill_between(
        dates, 0, 1, where=down_mask,
        transform=ax.get_xaxis_transform(),
        color="red", alpha=0.15, label="DOWN"
    )
    ax.set_title(title)
    ax.set_ylabel(y_label)
    ax.legend(loc="upper left")


def plot_price_with_labels(
    df_market: pd.DataFrame,
    df_labels: pd.DataFrame,
    price_mode: str = "raw",
    smooth: Optional[pd.Series] = None,
    gauss_window: Optional[int] = None,
    gauss_std: Optional[float] = None,
    figsize: tuple = (14, 5),
):
    """
    Plot price with label shading (forward-filled, no gaps).
    
    Args:
        df_market: Full market DataFrame
        df_labels: Labels DataFrame
        price_mode: 'log' or 'raw'
        smooth: Optional pre-computed smooth series
        gauss_window, gauss_std: Gaussian params (for label)
        figsize: Figure size
    """
    # Merge labels onto all market dates, then forward-fill
    df_plot = df_market[["date", "close"]].copy()
    df_plot["date"] = pd.to_datetime(df_plot["date"])
    
    df_labels_copy = df_labels[["date", "target"]].copy()
    df_labels_copy["date"] = pd.to_datetime(df_labels_copy["date"])
    
    df_plot = df_plot.merge(df_labels_copy, on="date", how="left")
    df_plot = df_plot.sort_values("date").reset_index(drop=True)
    df_plot["target"] = df_plot["target"].ffill()  # Forward-fill: state continues until change
    
    if price_mode == "log":
        price_series = np.log(df_plot["close"]).astype(float)
        y_label = "log(price)"
    else:
        price_series = df_plot["close"].astype(float)
        y_label = "price"
    
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.plot(df_plot["date"], price_series, label="price", linewidth=0.8, color="black")
    
    if smooth is not None and smooth.notna().any():
        label = f"Gauss(w={gauss_window}, std={gauss_std})" if gauss_window else "Gaussian"
        ax.plot(df_market["date"], smooth, label=label, linewidth=1.2)
    
    # Shade with axvspan - each day extends to the next
    dates = df_plot["date"].reset_index(drop=True)
    targets = df_plot["target"].values
    
    up_added = False
    down_added = False
    
    for i in range(len(dates)):
        target = targets[i]
        if pd.isna(target):
            continue
        
        # Extend to next date (or +1 day for last)
        if i < len(dates) - 1:
            end_date = dates.iloc[i + 1]
        else:
            end_date = dates.iloc[i] + pd.Timedelta(days=1)
        
        if target == 1:
            ax.axvspan(dates.iloc[i], end_date, color="green", alpha=0.15,
                      label="UP" if not up_added else None)
            up_added = True
        else:
            ax.axvspan(dates.iloc[i], end_date, color="red", alpha=0.15,
                      label="DOWN" if not down_added else None)
            down_added = True
    
    ax.set_title("Price with labels")
    ax.set_xlabel("Date")
    ax.set_ylabel(y_label)
    ax.legend()
    plt.show()


def plot_last_n_days(
    df_market: pd.DataFrame,
    df_labels: pd.DataFrame,
    n_days: int = 30,
    price_mode: str = "raw",
    figsize: tuple = (14, 5),
):
    """Plot last N days with labels (forward-filled to cover all days)."""
    # First merge labels onto ALL market dates, then ffill, then slice
    df_full = df_market[["date", "close"]].copy()
    df_full["date"] = pd.to_datetime(df_full["date"])
    
    df_labels_copy = df_labels[["date", "target"]].copy()
    df_labels_copy["date"] = pd.to_datetime(df_labels_copy["date"])
    
    df_full = df_full.merge(df_labels_copy, on="date", how="left")
    df_full = df_full.sort_values("date").reset_index(drop=True)
    df_full["target"] = df_full["target"].ffill()  # Forward-fill on full history
    
    # Now slice to last N days
    df_plot = df_full.tail(n_days).copy()
    
    if price_mode == "log":
        price_series = np.log(df_plot["close"]).astype(float)
        y_label = "log(price)"
    else:
        price_series = df_plot["close"].astype(float)
        y_label = "price"
    
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.plot(df_plot["date"], price_series, label="price", linewidth=1.2, color="tab:blue")
    
    # Shade labels with full background (no gaps)
    dates = pd.to_datetime(df_plot["date"]).reset_index(drop=True)
    targets = df_plot["target"].values
    
    for i in range(len(dates)):
        target = targets[i]
        if pd.isna(target):
            continue
        color = "green" if target == 1 else "red"
        # Extend to next date (or +1 day for last point)
        if i < len(dates) - 1:
            end_date = dates.iloc[i + 1]
        else:
            end_date = dates.iloc[i] + pd.Timedelta(days=1)
        ax.axvspan(dates.iloc[i], end_date, color=color, alpha=0.3)
    
    ax.set_title(f"Last {n_days} days")
    ax.set_xlabel("Date")
    ax.set_ylabel(y_label)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    
    n_labeled = df_plot["target"].notna().sum()
    print(f"Last {n_days} days: {n_labeled} labeled (forward-filled) out of {len(df_plot)}")


def plot_future_return_distribution(
    future_ret: pd.Series,
    title: str = "Future return distribution",
    figsize: tuple = (7, 4),
):
    """Plot future return distribution."""
    plt.figure(figsize=figsize)
    plt.hist(future_ret.dropna(), bins=80, color="tab:gray", alpha=0.8)
    plt.title(title)
    plt.xlabel("future return")
    plt.ylabel("count")
    plt.show()


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label_names: List[str] = ["DOWN", "UP"],
    figsize: tuple = (5, 4),
):
    """Plot confusion matrix."""
    labels = list(range(len(label_names)))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    
    plt.figure(figsize=figsize)
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=label_names, yticklabels=label_names
    )
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()


def plot_feature_importance(
    importance_df: pd.DataFrame,
    title: str = "Top Features by Importance",
    figsize: tuple = (8, 6),
):
    """Plot feature importance bar chart."""
    plt.figure(figsize=figsize)
    sns.barplot(data=importance_df, x="importance", y="feature", color="tab:blue")
    plt.title(title)
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.show()


def plot_predictions(
    df_test: pd.DataFrame,
    y_pred: np.ndarray,
    y_true: Optional[np.ndarray] = None,
    price_mode: str = "log",
    figsize: tuple = (12, 7),
):
    """
    Plot predictions with price (continuous shading, no gaps).
    
    Args:
        df_test: Test DataFrame with date and close
        y_pred: Predicted labels
        y_true: True labels (optional, for second panel)
        price_mode: 'log' or 'raw'
        figsize: Figure size
    """
    dates = pd.to_datetime(df_test["date"]).reset_index(drop=True)
    
    if price_mode == "log":
        close = np.log(df_test["close"]).values
        y_label = "log(price)"
    else:
        close = df_test["close"].values
        y_label = "price"
    
    def shade_continuous(ax, dates, labels, title):
        """Shade with axvspan - each label extends to next date."""
        ax.plot(dates, close, color="black", linewidth=1.2, label="Price")
        
        up_added = False
        down_added = False
        
        for i in range(len(dates)):
            label = labels[i]
            # Extend to next date (or +1 day for last)
            if i < len(dates) - 1:
                end_date = dates.iloc[i + 1]
            else:
                end_date = dates.iloc[i] + pd.Timedelta(days=1)
            
            if label == 1:
                ax.axvspan(dates.iloc[i], end_date, color="green", alpha=0.15, 
                          label="UP" if not up_added else None)
                up_added = True
            else:
                ax.axvspan(dates.iloc[i], end_date, color="red", alpha=0.15,
                          label="DOWN" if not down_added else None)
                down_added = True
        
        ax.set_title(title)
        ax.set_ylabel(y_label)
        ax.legend(loc="upper left")
    
    if y_true is not None:
        fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=True)
        shade_continuous(axes[0], dates, y_pred, "Predicted")
        shade_continuous(axes[1], dates, y_true, "True Labels")
        axes[1].set_xlabel("Date")
    else:
        fig, ax = plt.subplots(figsize=(12, 4))
        shade_continuous(ax, dates, y_pred, "Predicted")
        ax.set_xlabel("Date")
    
    plt.tight_layout()
    plt.show()


def plot_grid_search_results(
    results_df: pd.DataFrame,
    metric: str = "bal_acc",
    top_n: int = 15,
    figsize: tuple = (10, 6),
):
    """Plot grid search results as bar chart."""
    df = results_df.head(top_n).copy()
    df["label"] = df.apply(
        lambda r: f"orb={r['orb_mult']}, w={int(r['gauss_window'])}, s={r['gauss_std']}", axis=1
    )
    
    plt.figure(figsize=figsize)
    colors = plt.cm.viridis(np.linspace(0, 1, len(df)))
    plt.barh(df["label"], df[metric], color=colors)
    plt.xlabel(metric)
    plt.title(f"Top {top_n} Grid Search Results by {metric}")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()
