"""
Visualization utilities for flower species classification project.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from typing import Optional, List
import config


def setup_plotting_style():
    """Set up the plotting style."""
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (10, 6)


def plot_feature_distributions(df: pd.DataFrame, 
                               feature_columns: Optional[List[str]] = None,
                               target_column: Optional[str] = None,
                               save_path: Optional[str] = None) -> None:
    """
    Plot distribution of features by target class.
    
    Args:
        df: DataFrame containing the data
        feature_columns: List of feature columns to plot
        target_column: Name of the target column
        save_path: Path to save the plot (optional)
    """
    if feature_columns is None:
        feature_columns = config.FEATURE_COLUMNS
    if target_column is None:
        target_column = config.TARGET_COLUMN
    
    setup_plotting_style()
    
    n_features = len(feature_columns)
    fig, axes = plt.subplots(n_features, 1, figsize=(10, 4 * n_features))
    
    if n_features == 1:
        axes = [axes]
    
    for idx, feature in enumerate(feature_columns):
        sns.violinplot(data=df, x=target_column, y=feature, ax=axes[idx])
        axes[idx].set_title(f'Distribution of {feature} by Species')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Feature distribution plot saved to {save_path}")
    else:
        plt.show()


def plot_correlation_matrix(df: pd.DataFrame,
                           feature_columns: Optional[List[str]] = None,
                           save_path: Optional[str] = None) -> None:
    """
    Plot correlation matrix of features.
    
    Args:
        df: DataFrame containing the data
        feature_columns: List of feature columns
        save_path: Path to save the plot (optional)
    """
    if feature_columns is None:
        feature_columns = config.FEATURE_COLUMNS
    
    setup_plotting_style()
    
    correlation = df[feature_columns].corr()
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation, annot=True, cmap='coolwarm', center=0, 
                square=True, linewidths=1, cbar_kws={"shrink": 0.8})
    plt.title('Feature Correlation Matrix')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Correlation matrix plot saved to {save_path}")
    else:
        plt.show()


def plot_pairplot(df: pd.DataFrame,
                 feature_columns: Optional[List[str]] = None,
                 target_column: Optional[str] = None,
                 save_path: Optional[str] = None) -> None:
    """
    Create pairplot of features colored by target class.
    
    Args:
        df: DataFrame containing the data
        feature_columns: List of feature columns
        target_column: Name of the target column
        save_path: Path to save the plot (optional)
    """
    if feature_columns is None:
        feature_columns = config.FEATURE_COLUMNS
    if target_column is None:
        target_column = config.TARGET_COLUMN
    
    setup_plotting_style()
    
    plot_df = df[feature_columns + [target_column]]
    pairplot = sns.pairplot(plot_df, hue=target_column, diag_kind='kde')
    pairplot.fig.suptitle('Pairplot of Features by Species', y=1.01)
    
    if save_path:
        plt.savefig(save_path)
        print(f"Pairplot saved to {save_path}")
    else:
        plt.show()


def plot_confusion_matrix(y_true: pd.Series,
                         y_pred: np.ndarray,
                         labels: Optional[List[str]] = None,
                         save_path: Optional[str] = None) -> None:
    """
    Plot confusion matrix.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        labels: Class labels
        save_path: Path to save the plot (optional)
    """
    setup_plotting_style()
    
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap='Blues', values_format='d')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Confusion matrix plot saved to {save_path}")
    else:
        plt.show()


def plot_feature_importance(model, 
                           feature_names: List[str],
                           save_path: Optional[str] = None) -> None:
    """
    Plot feature importance for tree-based models.
    
    Args:
        model: Trained model with feature_importances_ attribute
        feature_names: Names of features
        save_path: Path to save the plot (optional)
    """
    if not hasattr(model, 'feature_importances_'):
        print(f"Model {type(model).__name__} does not have feature importances")
        return
    
    setup_plotting_style()
    
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    plt.figure(figsize=(10, 6))
    plt.title('Feature Importances')
    plt.bar(range(len(importances)), importances[indices])
    plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=45)
    plt.xlabel('Features')
    plt.ylabel('Importance')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Feature importance plot saved to {save_path}")
    else:
        plt.show()


def plot_model_comparison(results: dict, save_path: Optional[str] = None) -> None:
    """
    Plot comparison of multiple models.
    
    Args:
        results: Dictionary with model names as keys and metrics as values
        save_path: Path to save the plot (optional)
    """
    setup_plotting_style()
    
    models = list(results.keys())
    metrics = ['accuracy', 'precision', 'recall', 'f1_score']
    
    x = np.arange(len(models))
    width = 0.2
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for i, metric in enumerate(metrics):
        values = [results[model][metric] for model in models]
        ax.bar(x + i * width, values, width, label=metric.replace('_', ' ').title())
    
    ax.set_xlabel('Models')
    ax.set_ylabel('Score')
    ax.set_title('Model Comparison')
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(models, rotation=45, ha='right')
    ax.legend()
    ax.set_ylim([0, 1])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Model comparison plot saved to {save_path}")
    else:
        plt.show()
