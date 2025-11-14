import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Rectangle


def plot_or_forest(or_df: pd.DataFrame, output_path: str = None, 
                   figsize: tuple = (10, 8), title: str = "Forest Plot of Odds Ratios") -> str:
    """Create a forest plot of odds ratios with confidence intervals.
    
    Args:
        or_df: DataFrame with odds ratio data (must have columns: feature, odds_ratio, 
               odds_ratio_ci_lower, odds_ratio_ci_upper, p_value)
        output_path: Path to save the plot (optional)
        figsize: Figure size tuple
        title: Plot title
        
    Returns:
        Path where plot was saved, or "displayed" if no output_path
    """
    # Filter out intercept if present
    or_df = or_df[~or_df['feature'].str.contains('Intercept', na=False)].copy()
    
    # Sort by distance from 1.0 (null effect)
    or_df['distance_from_null'] = abs(or_df['odds_ratio'] - 1.0)
    or_df = or_df.sort_values('distance_from_null', ascending=False)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Set style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Plot odds ratios and confidence intervals
    y_positions = np.arange(len(or_df))
    
    # Plot confidence intervals
    for i, (_, row) in enumerate(or_df.iterrows()):
        # Confidence interval line
        ax.plot([row['odds_ratio_ci_lower'], row['odds_ratio_ci_upper']], 
                [y_positions[i], y_positions[i]], 
                color='black', linewidth=1.5, alpha=0.7)
        
        # Odds ratio point
        color = 'red' if row['odds_ratio'] > 1 else 'blue'
        marker = 'o' if row['p_value'] < 0.05 else 's'
        ax.scatter(row['odds_ratio'], y_positions[i], 
                  color=color, s=60, marker=marker, alpha=0.8, zorder=5)
    
    # Add vertical line at null effect (OR = 1.0)
    ax.axvline(x=1.0, color='black', linestyle='--', alpha=0.5, linewidth=1)
    
    # Customize y-axis
    ax.set_yticks(y_positions)
    ax.set_yticklabels(or_df['feature'], fontsize=10)
    
    # Customize x-axis
    ax.set_xlabel('Odds Ratio (log scale)', fontsize=12, fontweight='bold')
    ax.set_xscale('log')
    
    # Add grid
    ax.grid(True, alpha=0.3, axis='x')
    
    # Add title
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    
    # Add legend for significance
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', 
                  markersize=8, label='OR > 1 (p < 0.05)'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', 
                  markersize=8, label='OR < 1 (p < 0.05)'),
        plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='gray', 
                  markersize=8, label='Not significant (p ≥ 0.05)')
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=9)
    
    # Add p-value annotations
    for i, (_, row) in enumerate(or_df.iterrows()):
        if row['p_value'] < 0.001:
            p_text = '***'
        elif row['p_value'] < 0.01:
            p_text = '**'
        elif row['p_value'] < 0.05:
            p_text = '*'
        else:
            p_text = 'ns'
        
        # Position text to the right of the confidence interval
        x_pos = row['odds_ratio_ci_upper'] * 1.1
        ax.text(x_pos, y_positions[i], p_text, 
                fontsize=8, ha='left', va='center', fontweight='bold')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save or display
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        return output_path
    else:
        plt.show()
        return "displayed"


def plot_pr_curve(precision: np.ndarray, recall: np.ndarray, 
                  output_path: str = None, figsize: tuple = (8, 6),
                  title: str = "Precision-Recall Curve", 
                  auc_pr: float = None) -> str:
    """Create a precision-recall curve plot.
    
    Args:
        precision: Precision values
        recall: Recall values
        output_path: Path to save the plot (optional)
        figsize: Figure size tuple
        title: Plot title
        auc_pr: AUC-PR value to display (optional)
        
    Returns:
        Path where plot was saved, or "displayed" if no output_path
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Set style
    plt.style.use('default')
    
    # Plot PR curve
    ax.plot(recall, precision, linewidth=2, color='blue', alpha=0.8)
    
    # Add baseline (random classifier)
    baseline = np.mean(precision) if len(precision) > 0 else 0.5
    ax.axhline(y=baseline, color='red', linestyle='--', alpha=0.7, 
               label=f'Random Classifier (PR = {baseline:.3f})')
    
    # Customize axes
    ax.set_xlabel('Recall', fontsize=12, fontweight='bold')
    ax.set_ylabel('Precision', fontsize=12, fontweight='bold')
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    # Add title
    if auc_pr is not None:
        title += f" (AUC-PR = {auc_pr:.3f})"
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    
    # Add legend
    ax.legend(fontsize=10)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save or display
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        return output_path
    else:
        plt.show()
        return "displayed"


def plot_roc_curve(fpr: np.ndarray, tpr: np.ndarray, 
                   output_path: str = None, figsize: tuple = (8, 6),
                   title: str = "ROC Curve", auc_roc: float = None) -> str:
    """Create a ROC curve plot.
    
    Args:
        fpr: False positive rate values
        tpr: True positive rate values
        output_path: Path to save the plot (optional)
        figsize: Figure size tuple
        title: Plot title
        auc_roc: AUC-ROC value to display (optional)
        
    Returns:
        Path where plot was saved, or "displayed" if no output_path
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Set style
    plt.style.use('default')
    
    # Plot ROC curve
    ax.plot(fpr, tpr, linewidth=2, color='blue', alpha=0.8)
    
    # Add diagonal line (random classifier)
    ax.plot([0, 1], [0, 1], color='red', linestyle='--', alpha=0.7, 
            label='Random Classifier (AUC = 0.5)')
    
    # Customize axes
    ax.set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
    ax.set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    # Add title
    if auc_roc is not None:
        title += f" (AUC-ROC = {auc_roc:.3f})"
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    
    # Add legend
    ax.legend(fontsize=10)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save or display
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        return output_path
    else:
        plt.show()
        return "displayed"


def plot_threshold_metrics(threshold_df: pd.DataFrame, 
                          output_path: str = None, figsize: tuple = (12, 8),
                          title: str = "Metrics by Threshold") -> str:
    """Create a plot showing various metrics across different thresholds.
    
    Args:
        threshold_df: DataFrame with threshold metrics
        output_path: Path to save the plot (optional)
        figsize: Figure size tuple
        title: Plot title
        
    Returns:
        Path where plot was saved, or "displayed" if no output_path
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=figsize)
    
    # Set style
    plt.style.use('default')
    
    # Plot 1: Precision, Recall, F1
    ax1.plot(threshold_df['threshold'], threshold_df['precision'], 
             label='Precision', marker='o', alpha=0.8)
    ax1.plot(threshold_df['threshold'], threshold_df['recall'], 
             label='Recall', marker='s', alpha=0.8)
    ax1.plot(threshold_df['threshold'], threshold_df['f1'], 
             label='F1', marker='^', alpha=0.8)
    ax1.set_xlabel('Threshold')
    ax1.set_ylabel('Score')
    ax1.set_title('Precision, Recall, F1')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Specificity and Accuracy
    ax2.plot(threshold_df['threshold'], threshold_df['specificity'], 
             label='Specificity', marker='o', alpha=0.8)
    ax2.plot(threshold_df['threshold'], threshold_df['accuracy'], 
             label='Accuracy', marker='s', alpha=0.8)
    ax2.set_xlabel('Threshold')
    ax2.set_ylabel('Score')
    ax2.set_title('Specificity and Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Positive Rate
    ax3.plot(threshold_df['threshold'], threshold_df['positive_rate'], 
             label='Positive Rate', marker='o', color='green', alpha=0.8)
    ax3.set_xlabel('Threshold')
    ax3.set_ylabel('Positive Rate')
    ax3.set_title('Positive Rate')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Confusion Matrix Components
    ax4.plot(threshold_df['threshold'], threshold_df['tp'], 
             label='True Positives', marker='o', alpha=0.8)
    ax4.plot(threshold_df['threshold'], threshold_df['fp'], 
             label='False Positives', marker='s', alpha=0.8)
    ax4.plot(threshold_df['threshold'], threshold_df['tn'], 
             label='True Negatives', marker='^', alpha=0.8)
    ax4.plot(threshold_df['threshold'], threshold_df['fn'], 
             label='False Negatives', marker='d', alpha=0.8)
    ax4.set_xlabel('Threshold')
    ax4.set_ylabel('Count')
    ax4.set_title('Confusion Matrix Components')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # Add overall title
    fig.suptitle(title, fontsize=16, fontweight='bold')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save or display
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        return output_path
    else:
        plt.show()
        return "displayed"
