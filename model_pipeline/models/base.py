from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd


class BaseBinaryClassifier(ABC):
    """Base interface for binary classification models."""
    
    def __init__(self):
        self.is_fitted = False
        self.feature_names_ = None
        self.target_name_ = None
    
    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'BaseBinaryClassifier':
        """Fit the model to the training data.
        
        Args:
            X: Training features DataFrame
            y: Training target Series
            
        Returns:
            self: Fitted model instance
        """
        pass
    
    @abstractmethod
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict class probabilities.
        
        Args:
            X: Features DataFrame
            
        Returns:
            Array of shape (n_samples, 2) with probabilities for [class_0, class_1]
        """
        pass
    
    def predict(self, X: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        """Predict binary classes using a threshold.
        
        Args:
            X: Features DataFrame
            threshold: Probability threshold for positive class (default: 0.5)
            
        Returns:
            Array of binary predictions (0 or 1)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        proba = self.predict_proba(X)
        return (proba[:, 1] >= threshold).astype(int)
    
    @abstractmethod
    def get_coefficients(self) -> pd.DataFrame:
        """Get model coefficients and statistics.
        
        Returns:
            DataFrame with coefficient names, values, standard errors, p-values, and CIs
        """
        pass
    
    def get_feature_importance(self) -> pd.Series:
        """Get feature importance scores (absolute coefficient values for linear models).
        
        Returns:
            Series with feature names as index and importance scores as values
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before getting feature importance")
        
        coef_df = self.get_coefficients()
        return pd.Series(
            data=abs(coef_df['coefficient'].values),
            index=coef_df['feature'].values
        ).sort_values(ascending=False)
