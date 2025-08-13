from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    precision_recall_curve, roc_auc_score, roc_curve
)


def compute_pr_metrics(y_true: np.ndarray, y_score: np.ndarray) -> Dict:
    """Compute precision-recall metrics.
    
    Args:
        y_true: True binary labels
        y_score: Predicted probabilities for positive class
        
    Returns:
        Dictionary with PR curve data and AUC-PR
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    
    # Calculate AUC-PR (area under precision-recall curve)
    auc_pr = np.trapz(precision, recall)
    
    # Calculate F1 score at each threshold
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
    
    return {
        'precision': precision,
        'recall': recall,
        'thresholds': thresholds,
        'auc_pr': auc_pr,
        'f1_scores': f1_scores
    }


def compute_roc_metrics(y_true: np.ndarray, y_score: np.ndarray) -> Dict:
    """Compute ROC curve metrics.
    
    Args:
        y_true: True binary labels
        y_score: Predicted probabilities for positive class
        
    Returns:
        Dictionary with ROC curve data and AUC-ROC
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    auc_roc = roc_auc_score(y_true, y_score)
    
    return {
        'fpr': fpr,
        'tpr': tpr,
        'thresholds': thresholds,
        'auc_roc': auc_roc
    }


def threshold_table(y_true: np.ndarray, y_score: np.ndarray, 
                   percentiles: List[float] = None) -> pd.DataFrame:
    """Generate threshold table with metrics at different probability thresholds.
    
    Args:
        y_true: True binary labels
        y_score: Predicted probabilities for positive class
        percentiles: List of percentiles to evaluate (default: 10 percentiles from 0.1 to 1.0)
        
    Returns:
        DataFrame with metrics at each threshold
    """
    if percentiles is None:
        percentiles = np.linspace(0.1, 1.0, 10)
    
    # Calculate thresholds based on percentiles
    thresholds = np.percentile(y_score, [p * 100 for p in percentiles])
    
    results = []
    
    for threshold in thresholds:
        # Make predictions at this threshold
        y_pred = (y_score >= threshold).astype(int)
        
        # Calculate metrics
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        # Positive rate (predicted positive / total)
        positive_rate = (tp + fp) / len(y_true)
        
        results.append({
            'threshold': threshold,
            'precision': precision,
            'recall': recall,
            'specificity': specificity,
            'accuracy': accuracy,
            'f1': f1,
            'positive_rate': positive_rate,
            'tp': tp,
            'fp': fp,
            'tn': tn,
            'fn': fn
        })
    
    return pd.DataFrame(results)


def compute_summary_metrics(y_true: np.ndarray, y_pred: np.ndarray, 
                           y_score: np.ndarray) -> Dict:
    """Compute comprehensive summary metrics.
    
    Args:
        y_true: True binary labels
        y_pred: Predicted binary labels
        y_score: Predicted probabilities for positive class
        
    Returns:
        Dictionary with summary metrics
    """
    # Basic classification metrics
    accuracy = accuracy_score(y_true, y_pred)
    
    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    # Derived metrics
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    # Prevalence
    prevalence = np.mean(y_true)
    
    # Positive and negative predictive values
    ppv = precision  # same as precision
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0
    
    # Likelihood ratios
    lr_positive = recall / (1 - specificity) if specificity < 1 else np.inf
    lr_negative = (1 - recall) / specificity if specificity > 0 else np.inf
    
    # AUC metrics
    auc_roc = roc_auc_score(y_true, y_score)
    
    # PR metrics
    precision_pr, recall_pr, _ = precision_recall_curve(y_true, y_score)
    auc_pr = np.trapz(precision_pr, recall_pr)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'specificity': specificity,
        'f1': f1,
        'prevalence': prevalence,
        'ppv': ppv,
        'npv': npv,
        'lr_positive': lr_positive,
        'lr_negative': lr_negative,
        'auc_roc': auc_roc,
        'auc_pr': auc_pr,
        'tp': tp,
        'fp': fp,
        'tn': tn,
        'fn': fn
    }


def find_optimal_threshold(y_true: np.ndarray, y_score: np.ndarray, 
                          metric: str = 'f1') -> Tuple[float, float]:
    """Find optimal threshold based on specified metric.
    
    Args:
        y_true: True binary labels
        y_score: Predicted probabilities for positive class
        metric: Metric to optimize ('f1', 'precision', 'recall', 'f1_balanced')
        
    Returns:
        Tuple of (optimal_threshold, optimal_metric_value)
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    
    if metric == 'f1':
        # F1 score
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
        optimal_idx = np.argmax(f1_scores)
        optimal_threshold = thresholds[optimal_idx] if optimal_idx < len(thresholds) else 0.5
        optimal_value = f1_scores[optimal_idx]
        
    elif metric == 'precision':
        # Maximum precision
        optimal_idx = np.argmax(precision)
        optimal_threshold = thresholds[optimal_idx] if optimal_idx < len(thresholds) else 0.5
        optimal_value = precision[optimal_idx]
        
    elif metric == 'recall':
        # Maximum recall
        optimal_idx = np.argmax(recall)
        optimal_threshold = thresholds[optimal_idx] if optimal_idx < len(thresholds) else 0.5
        optimal_value = recall[optimal_idx]
        
    elif metric == 'f1_balanced':
        # Balanced F1 (equal weight to precision and recall)
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
        # Find threshold where precision and recall are most balanced
        balance_scores = 1 - abs(precision - recall)
        optimal_idx = np.argmax(f1_scores * balance_scores)
        optimal_threshold = thresholds[optimal_idx] if optimal_idx < len(thresholds) else 0.5
        optimal_value = f1_scores[optimal_idx]
        
    else:
        raise ValueError(f"Unknown metric: {metric}. Use 'f1', 'precision', 'recall', or 'f1_balanced'")
    
    return optimal_threshold, optimal_value
