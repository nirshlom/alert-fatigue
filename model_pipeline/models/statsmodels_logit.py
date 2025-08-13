import warnings
from typing import List, Optional

import numpy as np
import pandas as pd
import patsy
import statsmodels.api as sm
from statsmodels.genmod.generalized_linear_model import GLMResults
from statsmodels.regression.linear_model import RegressionResults

from .base import BaseBinaryClassifier


class StatsmodelsLogitModel(BaseBinaryClassifier):
    """Logistic regression model using statsmodels with patsy formula interface."""
    
    def __init__(self, use_glm: bool = True):
        """Initialize the model.
        
        Args:
            use_glm: If True, use GLM with binomial family; if False, use Logit
        """
        super().__init__()
        self.use_glm = use_glm
        self.model = None
        self.result = None
        self.formula = None
        self.feature_mapping = {}
        
    def _build_formula(self, X: pd.DataFrame, y: pd.Series) -> str:
        """Build patsy formula from features and target.
        
        Args:
            X: Features DataFrame
            y: Target Series
            
        Returns:
            Patsy formula string
        """
        target_name = y.name
        feature_terms = []
        
        for col in X.columns:
            if X[col].dtype in ['object', 'category', 'string']:
                # Categorical feature with frozen levels
                levels = list(X[col].cat.categories) if hasattr(X[col], 'cat') else list(X[col].unique())
                feature_terms.append(f"C({col}, levels={levels})")
            else:
                # Numeric feature
                feature_terms.append(col)
        
        formula = f"{target_name} ~ " + " + ".join(feature_terms)
        return formula
    
    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'StatsmodelsLogitModel':
        """Fit the logistic regression model.
        
        Args:
            X: Training features DataFrame
            y: Training target Series
            
        Returns:
            self: Fitted model instance
        """
        # Store feature and target names
        self.feature_names_ = list(X.columns)
        self.target_name_ = y.name
        
        # Build formula
        self.formula = self._build_formula(X, y)
        
        # Prepare data for statsmodels
        data = pd.concat([X, y], axis=1)
        
        # Fit model
        if self.use_glm:
            self.model = sm.GLM.from_formula(
                self.formula, 
                data=data, 
                family=sm.families.Binomial()
            )
        else:
            self.model = sm.Logit.from_formula(self.formula, data=data)
        
        # Suppress convergence warnings for now
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.result = self.model.fit(disp=0)
        
        self.is_fitted = True
        return self
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict class probabilities.
        
        Args:
            X: Features DataFrame
            
        Returns:
            Array of shape (n_samples, 2) with probabilities for [class_0, class_1]
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        # Prepare data for prediction
        data = X.copy()
        if self.target_name_ not in data.columns:
            # Add dummy target column for patsy formula
            data[self.target_name_] = 0
        
        # Get prediction matrix
        try:
            # For statsmodels formula API, we need to use the result's predict method
            # with the same data structure that was used during training
            if self.use_glm:
                # GLM returns probabilities directly
                probs = self.result.predict(data)
            else:
                # Logit returns logits, need to convert to probabilities
                logits = self.result.predict(data)
                probs = 1 / (1 + np.exp(-logits))
            
            # Return as [class_0, class_1] probabilities
            return np.column_stack([1 - probs, probs])
            
        except Exception as e:
            raise ValueError(f"Error in prediction: {e}. Formula: {self.formula}")
    
    def get_coefficients(self) -> pd.DataFrame:
        """Get model coefficients and statistics.
        
        Returns:
            DataFrame with coefficient names, values, standard errors, p-values, and CIs
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before getting coefficients")
        
        # Extract coefficient information
        params = self.result.params
        conf_int = self.result.conf_int()
        pvalues = self.result.pvalues
        
        # Create coefficient DataFrame
        coef_df = pd.DataFrame({
            'feature': params.index,
            'coefficient': params.values,
            'std_error': self.result.bse.values,
            'p_value': pvalues.values,
            'ci_lower': conf_int.iloc[:, 0].values,
            'ci_upper': conf_int.iloc[:, 1].values
        })
        
        # Calculate odds ratios and their CIs
        coef_df['odds_ratio'] = np.exp(coef_df['coefficient'])
        coef_df['odds_ratio_ci_lower'] = np.exp(coef_df['ci_lower'])
        coef_df['odds_ratio_ci_upper'] = np.exp(coef_df['ci_upper'])
        
        # Add significance indicators
        coef_df['significant'] = coef_df['p_value'] < 0.05
        coef_df['significance_level'] = coef_df['p_value'].apply(self._get_significance_level)
        
        return coef_df
    
    def _get_significance_level(self, p_value: float) -> str:
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
    
    def get_model_summary(self) -> str:
        """Get model summary as string.
        
        Returns:
            Model summary string
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before getting summary")
        
        return str(self.result.summary())
    
    def get_aic_bic(self) -> dict:
        """Get AIC and BIC values.
        
        Returns:
            Dictionary with AIC and BIC values
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before getting AIC/BIC")
        
        return {
            'aic': self.result.aic,
            'bic': self.result.bic
        }
