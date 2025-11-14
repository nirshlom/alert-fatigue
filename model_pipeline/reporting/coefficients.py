import numpy as np
import pandas as pd
from typing import Dict, Optional


def coefficients_to_or(coef_df: pd.DataFrame, 
                      confidence_level: float = 0.95) -> pd.DataFrame:
    """Convert coefficient DataFrame to odds ratio format for reporting.
    
    Args:
        coef_df: DataFrame with coefficient information from fitted model
        confidence_level: Confidence level for intervals (default: 0.95)
        
    Returns:
        DataFrame with odds ratios and confidence intervals
    """
    # Create a copy to avoid modifying original
    or_df = coef_df.copy()
    
    # Ensure required columns exist
    required_cols = ['feature', 'coefficient', 'std_error', 'p_value']
    missing_cols = [col for col in required_cols if col not in or_df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Calculate odds ratios if not already present
    if 'odds_ratio' not in or_df.columns:
        or_df['odds_ratio'] = np.exp(or_df['coefficient'])
    
    # Calculate confidence intervals if not already present
    if 'odds_ratio_ci_lower' not in or_df.columns or 'odds_ratio_ci_upper' not in or_df.columns:
        # Calculate confidence intervals for coefficients
        z_score = 1.96  # For 95% CI, approximately 1.96
        or_df['coefficient_ci_lower'] = or_df['coefficient'] - z_score * or_df['std_error']
        or_df['coefficient_ci_upper'] = or_df['coefficient'] + z_score * or_df['std_error']
        
        # Convert to odds ratio confidence intervals
        or_df['odds_ratio_ci_lower'] = np.exp(or_df['coefficient_ci_lower'])
        or_df['odds_ratio_ci_upper'] = np.exp(or_df['coefficient_ci_upper'])
    
    # Add significance indicators
    if 'significant' not in or_df.columns:
        or_df['significant'] = or_df['p_value'] < 0.05
    
    if 'significance_level' not in or_df.columns:
        or_df['significance_level'] = or_df['p_value'].apply(_get_significance_level)
    
    # Add interpretation
    or_df['interpretation'] = or_df.apply(_interpret_odds_ratio, axis=1)
    
    # Format for display
    or_df['odds_ratio_formatted'] = or_df['odds_ratio'].apply(lambda x: f"{x:.3f}")
    or_df['ci_formatted'] = or_df.apply(
        lambda row: f"({row['odds_ratio_ci_lower']:.3f}, {row['odds_ratio_ci_upper']:.3f})", 
        axis=1
    )
    or_df['p_value_formatted'] = or_df['p_value'].apply(_format_p_value)
    
    # Select and reorder columns for reporting
    report_cols = [
        'feature', 'odds_ratio', 'odds_ratio_formatted', 'ci_formatted',
        'odds_ratio_ci_lower', 'odds_ratio_ci_upper', 'coefficient',
        'std_error', 'p_value', 'p_value_formatted', 'significant', 
        'significance_level', 'interpretation'
    ]
    
    # Only include columns that exist
    available_cols = [col for col in report_cols if col in or_df.columns]
    or_df = or_df[available_cols]
    
    return or_df


def _get_significance_level(p_value: float) -> str:
    """Get significance level string based on p-value.
    
    Args:
        p_value: P-value
        
    Returns:
        Significance level string
    """
    if p_value < 0.001:
        return "***"
    elif p_value < 0.01:
        return "**"
    elif p_value < 0.05:
        return "*"
    else:
        return "ns"


def _interpret_odds_ratio(row: pd.Series) -> str:
    """Interpret odds ratio for reporting.
    
    Args:
        row: Row from coefficient DataFrame
        
    Returns:
        Interpretation string
    """
    or_val = row['odds_ratio']
    p_val = row['p_value']
    
    if p_val >= 0.05:
        return "Not significant"
    
    if or_val > 1:
        if or_val < 1.5:
            strength = "slightly"
        elif or_val < 3:
            strength = "moderately"
        else:
            strength = "strongly"
        return f"{strength} increases odds"
    else:
        if or_val > 0.67:
            strength = "slightly"
        elif or_val > 0.33:
            strength = "moderately"
        else:
            strength = "strongly"
        return f"{strength} decreases odds"


def _format_p_value(p_value: float) -> str:
    """Format p-value for display.
    
    Args:
        p_value: P-value
        
    Returns:
        Formatted p-value string
    """
    if p_value < 0.001:
        return "< 0.001"
    elif p_value < 0.01:
        return f"{p_value:.3f}"
    elif p_value < 0.05:
        return f"{p_value:.3f}"
    else:
        return f"{p_value:.3f}"


def create_coefficient_summary(coef_df: pd.DataFrame) -> Dict:
    """Create a summary of coefficient analysis.
    
    Args:
        coef_df: DataFrame with coefficient information
        
    Returns:
        Dictionary with summary statistics
    """
    summary = {}
    
    # Count significant features
    if 'significant' in coef_df.columns:
        summary['total_features'] = len(coef_df)
        summary['significant_features'] = coef_df['significant'].sum()
        summary['non_significant_features'] = (~coef_df['significant']).sum()
    
    # Count by significance level
    if 'significance_level' in coef_df.columns:
        significance_counts = coef_df['significance_level'].value_counts()
        summary['significance_breakdown'] = significance_counts.to_dict()
    
    # Odds ratio statistics
    if 'odds_ratio' in coef_df.columns:
        # Filter out intercept if present
        or_values = coef_df[~coef_df['feature'].str.contains('Intercept', na=False)]['odds_ratio']
        
        summary['odds_ratio_stats'] = {
            'min': float(or_values.min()),
            'max': float(or_values.max()),
            'mean': float(or_values.mean()),
            'median': float(or_values.median())
        }
        
        # Count protective vs risk factors
        protective = (or_values < 1).sum()
        risk = (or_values > 1).sum()
        summary['factor_types'] = {
            'protective_factors': int(protective),
            'risk_factors': int(risk)
        }
    
    # P-value statistics
    if 'p_value' in coef_df.columns:
        p_values = coef_df['p_value']
        summary['p_value_stats'] = {
            'min': float(p_values.min()),
            'max': float(p_values.max()),
            'mean': float(p_values.mean()),
            'median': float(p_values.median())
        }
    
    return summary


def filter_significant_coefficients(coef_df: pd.DataFrame, 
                                  alpha: float = 0.05) -> pd.DataFrame:
    """Filter coefficients to only include significant ones.
    
    Args:
        coef_df: DataFrame with coefficient information
        alpha: Significance level (default: 0.05)
        
    Returns:
        DataFrame with only significant coefficients
    """
    if 'p_value' not in coef_df.columns:
        raise ValueError("p_value column not found in coefficient DataFrame")
    
    significant = coef_df[coef_df['p_value'] < alpha].copy()
    
    # Sort by p-value
    significant = significant.sort_values('p_value')
    
    return significant


def sort_coefficients_by_importance(coef_df: pd.DataFrame, 
                                   method: str = 'odds_ratio') -> pd.DataFrame:
    """Sort coefficients by importance.
    
    Args:
        coef_df: DataFrame with coefficient information
        method: Sorting method ('odds_ratio', 'p_value', 'coefficient')
        
    Returns:
        Sorted DataFrame
    """
    if method == 'odds_ratio':
        # Sort by distance from 1.0 (null effect)
        coef_df = coef_df.copy()
        coef_df['distance_from_null'] = abs(coef_df['odds_ratio'] - 1.0)
        return coef_df.sort_values('distance_from_null', ascending=False)
    
    elif method == 'p_value':
        return coef_df.sort_values('p_value')
    
    elif method == 'coefficient':
        # Sort by absolute coefficient value
        coef_df = coef_df.copy()
        coef_df['abs_coefficient'] = abs(coef_df['coefficient'])
        return coef_df.sort_values('abs_coefficient', ascending=False)
    
    else:
        raise ValueError(f"Unknown sorting method: {method}. Use 'odds_ratio', 'p_value', or 'coefficient'")


# Import numpy for calculations
import numpy as np
