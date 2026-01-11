import warnings
import re
from typing import List, Optional, Dict, Union

import numpy as np
import pandas as pd

try:
    import rpy2.robjects as ro
    from rpy2.robjects import pandas2ri
    from rpy2.robjects.conversion import localconverter
    from rpy2.robjects.packages import importr
    from rpy2.rinterface_lib.embedded import RRuntimeError
    
    # Import R packages
    try:
        lme4 = importr('lme4')
        stats = importr('stats')
        base = importr('base')
    except RRuntimeError as e:
        raise ImportError(
            f"Failed to import R packages. Make sure R and lme4 are installed.\n"
            f"Install lme4 in R with: install.packages('lme4')\n"
            f"Original error: {e}"
        )
    
    R_AVAILABLE = True
except ImportError as e:
    R_AVAILABLE = False
    R_IMPORT_ERROR = str(e)

from .base import BaseBinaryClassifier


class LME4GLMMModel(BaseBinaryClassifier):
    """Generalized Linear Mixed Model using R's lme4 package via rpy2.
    
    This model supports random effects (e.g., random intercepts by patient, unit, etc.)
    which are essential for hierarchical/multilevel data structures.
    
    Example:
        # Simple random intercept model
        model = LME4GLMMModel(random_effects="(1|patient_id)")
        model.fit(X, y, groups=df['patient_id'])
        
        # Random intercept and slope
        model = LME4GLMMModel(random_effects="(1 + age|patient_id)")
        model.fit(X, y, groups=df[['patient_id']])
    """
    
    def __init__(
        self,
        random_effects: Optional[str] = None,
        family: str = "binomial",
        link: str = "logit",
        control: Optional[Dict] = None
    ):
        """Initialize the GLMM model.
        
        Args:
            random_effects: R formula string for random effects (e.g., "(1|patient_id)").
                           If None, model will be a regular GLM (no random effects).
            family: Distribution family ("binomial", "poisson", "gaussian", etc.)
            link: Link function ("logit", "probit", "identity", etc.)
            control: Optional dict of lme4 control parameters (e.g., {'optimizer': 'bobyqa'})
        """
        if not R_AVAILABLE:
            raise ImportError(
                f"rpy2 or R packages not available. {R_IMPORT_ERROR}\n"
                f"Please install: pip install rpy2\n"
                f"And install lme4 in R: install.packages('lme4')"
            )
        
        super().__init__()
        self.random_effects = random_effects
        self.family = family
        self.link = link
        self.control = control or {}
        
        self.model = None
        self.formula = None
        self.group_columns = []
        self.feature_mapping = {}
        
    def _build_formula(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        groups: Optional[Union[pd.Series, pd.DataFrame]] = None
    ) -> str:
        """Build R formula string from features, target, and random effects.
        
        Args:
            X: Features DataFrame
            y: Target Series
            groups: Optional grouping variable(s) for random effects
            
        Returns:
            R formula string
        """
        target_name = y.name
        
        # Build fixed effects part
        feature_terms = []
        for col in X.columns:
            # For categorical features, R will handle them automatically
            # We can add explicit factor() if needed, but R handles it well
            feature_terms.append(col)
        
        fixed_effects = " + ".join(feature_terms)
        
        # Add random effects if specified
        if self.random_effects:
            # If random_effects contains placeholder, replace with actual group column
            if groups is not None:
                if isinstance(groups, pd.Series):
                    group_col = groups.name
                    # Replace common placeholders
                    random_effects = self.random_effects.replace("(1|group)", f"(1|{group_col})")
                    random_effects = random_effects.replace("(1|GROUP)", f"(1|{group_col})")
                elif isinstance(groups, pd.DataFrame):
                    # Multiple grouping variables - use first one or let user specify
                    if len(groups.columns) == 1:
                        group_col = groups.columns[0]
                        random_effects = self.random_effects.replace("(1|group)", f"(1|{group_col})")
                    else:
                        # User should specify the exact formula
                        random_effects = self.random_effects
                else:
                    random_effects = self.random_effects
            else:
                random_effects = self.random_effects
            
            formula = f"{target_name} ~ {fixed_effects} + {random_effects}"
        else:
            formula = f"{target_name} ~ {fixed_effects}"
        
        return formula
    
    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        groups: Optional[Union[pd.Series, pd.DataFrame]] = None
    ) -> 'LME4GLMMModel':
        """Fit the GLMM model.
        
        Args:
            X: Training features DataFrame
            y: Training target Series
            groups: Optional grouping variable(s) for random effects.
                   Can be a Series (single grouping) or DataFrame (multiple groups).
                   If random_effects is specified but groups is None, assumes
                   grouping columns are already in X.
            
        Returns:
            self: Fitted model instance
        """
        # Store feature and target names
        self.feature_names_ = list(X.columns)
        self.target_name_ = y.name
        
        # Prepare data for R
        data = X.copy()
        data[self.target_name_] = y
        
        # Add grouping variables to data if provided
        if groups is not None:
            if isinstance(groups, pd.Series):
                data[groups.name] = groups
                self.group_columns = [groups.name]
            elif isinstance(groups, pd.DataFrame):
                for col in groups.columns:
                    data[col] = groups[col]
                self.group_columns = list(groups.columns)
        
        # Build formula
        self.formula = self._build_formula(X, y, groups)
        
        # Convert pandas DataFrame to R data.frame using new conversion context
        with localconverter(ro.default_converter + pandas2ri.converter):
            r_data = ro.conversion.py2rpy(data)
        
        # Build R formula object
        # ro.Formula can take a string directly
        r_formula = ro.Formula(self.formula)
        
        # Set up family
        if self.family == "binomial":
            if self.link == "logit":
                r_family = stats.binomial(link="logit")
            elif self.link == "probit":
                r_family = stats.binomial(link="probit")
            else:
                r_family = stats.binomial(link=self.link)
        else:
            # For other families, use stats::family
            r_family = getattr(stats, self.family)(link=self.link)
        
        # Set up control parameters
        r_control = None
        if self.control:
            # Create lmerControl object if needed
            if 'optimizer' in self.control or 'optCtrl' in self.control:
                try:
                    lmer_control = lme4.lmerControl
                    control_args = {}
                    if 'optimizer' in self.control:
                        control_args['optimizer'] = self.control['optimizer']
                    if 'optCtrl' in self.control:
                        control_args['optCtrl'] = ro.ListVector(self.control['optCtrl'])
                    r_control = lmer_control(**control_args)
                except:
                    # Fallback: pass as list
                    r_control = ro.ListVector(self.control)
        
        # Fit model using glmer (for binomial) or lmer (for gaussian)
        try:
            if self.family == "binomial":
                if r_control:
                    self.model = lme4.glmer(
                        r_formula,
                        data=r_data,
                        family=r_family,
                        control=r_control
                    )
                else:
                    self.model = lme4.glmer(
                        r_formula,
                        data=r_data,
                        family=r_family
                    )
            else:
                # For other families, use appropriate function
                if r_control:
                    self.model = lme4.glmer(
                        r_formula,
                        data=r_data,
                        family=r_family,
                        control=r_control
                    )
                else:
                    self.model = lme4.glmer(
                        r_formula,
                        data=r_data,
                        family=r_family
                    )
        except RRuntimeError as e:
            raise ValueError(
                f"Failed to fit GLMM model. R error: {e}\n"
                f"Formula: {self.formula}\n"
                f"Check your data and random effects specification."
            )
        
        self.is_fitted = True
        return self
    
    def predict_proba(
        self,
        X: pd.DataFrame,
        groups: Optional[Union[pd.Series, pd.DataFrame]] = None,
        allow_new_levels: bool = False
    ) -> np.ndarray:
        """Predict class probabilities.
        
        Args:
            X: Features DataFrame
            groups: Optional grouping variable(s) for random effects.
                   Must match structure used in fit().
            allow_new_levels: If True, allow new grouping levels (uses population-level predictions)
            
        Returns:
            Array of shape (n_samples, 2) with probabilities for [class_0, class_1]
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        # Prepare data for R
        data = X.copy()
        
        # Add dummy target column for formula
        if self.target_name_ not in data.columns:
            data[self.target_name_] = 0
        
        # Add grouping variables if provided
        if groups is not None:
            if isinstance(groups, pd.Series):
                data[groups.name] = groups
            elif isinstance(groups, pd.DataFrame):
                for col in groups.columns:
                    data[col] = groups[col]
        
        # Convert to R data.frame using new conversion context
        with localconverter(ro.default_converter + pandas2ri.converter):
            r_data = ro.conversion.py2rpy(data)
        
        # Make predictions
        try:
            # Use R's predict() function
            # For lme4 models, we need to handle re.form parameter specially
            if allow_new_levels:
                # For new levels, use re.form=NA to get population-level predictions
                # In R: predict(model, newdata=data, type="response", allow.new.levels=TRUE, re.form=NA)
                # Use ro.r() to evaluate R code directly for complex parameter handling
                ro.r.assign('model', self.model)
                ro.r.assign('newdata', r_data)
                probs = np.array(ro.r(
                    'predict(model, newdata=newdata, type="response", '
                    'allow.new.levels=TRUE, re.form=NA)'
                ))
            else:
                # Standard prediction with random effects
                r_predict = ro.r['predict']
                probs = np.array(r_predict(
                    self.model,
                    newdata=r_data,
                    type="response",
                    allow_new_levels=False
                ))
        except RRuntimeError as e:
            raise ValueError(
                f"Error in prediction: {e}\n"
                f"Formula: {self.formula}\n"
                f"Try setting allow_new_levels=True if you have new grouping levels."
            )
        
        # Return as [class_0, class_1] probabilities
        return np.column_stack([1 - probs, probs])
    
    def _clean_term_name(self, term: str) -> str:
        """Return a concise, readable term name.
        
        Examples:
            "genderMALE" -> "gender_MALE"
            "unit_category_udInternal" -> "unit_category_ud_Internal"
            "(Intercept)" -> "Intercept"
        """
        # Handle intercept
        if term == "(Intercept)":
            return "Intercept"
        
        # Try to split on capital letters (R's default factor encoding)
        # This handles cases like "genderMALE" -> "gender_MALE"
        parts = re.split(r'([A-Z][a-z]*)', term, 1)
        if len(parts) > 1 and parts[1]:
            return f"{parts[0]}_{parts[1]}"
        
        return term
    
    def get_coefficients(self) -> pd.DataFrame:
        """Get fixed effects coefficients and statistics.
        
        Returns:
            DataFrame with coefficient names, values, standard errors, p-values, and CIs
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before getting coefficients")
        
        try:
            # Extract fixed effects summary
            summary = lme4.summary(self.model)
            
            # Get coefficients table
            coef_table = summary.rx2('coefficients')
            
            # Convert to numpy array
            coef_array = np.array(coef_table)
            coef_names = list(coef_table.rownames)
            
            # Extract statistics
            # R's summary gives: Estimate, Std. Error, z value, Pr(>|z|)
            estimates = coef_array[:, 0]
            std_errors = coef_array[:, 1]
            z_values = coef_array[:, 2]
            p_values = coef_array[:, 3]
            
            # Calculate confidence intervals (Wald CIs)
            # 95% CI: estimate ± 1.96 * SE
            ci_lower = estimates - 1.96 * std_errors
            ci_upper = estimates + 1.96 * std_errors
            
            # Clean term names
            clean_names = [self._clean_term_name(name) for name in coef_names]
            
            # Create coefficient DataFrame
            coef_df = pd.DataFrame({
                'feature': clean_names,
                'coefficient': estimates,
                'std_error': std_errors,
                'p_value': p_values,
                'ci_lower': ci_lower,
                'ci_upper': ci_upper
            })
            
            # Calculate odds ratios and their CIs (for binomial models)
            if self.family == "binomial":
                coef_df['odds_ratio'] = np.exp(coef_df['coefficient'])
                coef_df['odds_ratio_ci_lower'] = np.exp(coef_df['ci_lower'])
                coef_df['odds_ratio_ci_upper'] = np.exp(coef_df['ci_upper'])
            else:
                coef_df['odds_ratio'] = np.nan
                coef_df['odds_ratio_ci_lower'] = np.nan
                coef_df['odds_ratio_ci_upper'] = np.nan
            
            # Add significance indicators
            coef_df['significant'] = coef_df['p_value'] < 0.05
            coef_df['significance_level'] = coef_df['p_value'].apply(self._get_significance_level)
            
            return coef_df
            
        except Exception as e:
            raise ValueError(f"Error extracting coefficients: {e}")
    
    def get_random_effects(self) -> pd.DataFrame:
        """Get random effects variances and standard deviations.
        
        Returns:
            DataFrame with random effects information
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before getting random effects")
        
        if not self.random_effects:
            return pd.DataFrame()  # No random effects
        
        try:
            # Extract variance-covariance matrix of random effects
            varcorr = lme4.VarCorr(self.model)
            
            # Get summary of random effects
            summary = lme4.summary(self.model)
            re_summary = summary.rx2('varcor')
            
            # This is complex - R stores it as a list of matrices
            # For now, return a simplified version
            # Users can access full R object via self.model if needed
            
            return pd.DataFrame({
                'note': ['Use model.model to access full R random effects object']
            })
            
        except Exception as e:
            warnings.warn(f"Could not extract random effects summary: {e}")
            return pd.DataFrame()
    
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
        """Return a readable summary of the model.
        
        Returns:
            Formatted string with model summary
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before getting summary")
        
        try:
            # Get R summary
            summary = lme4.summary(self.model)
            
            # Extract key information
            coef_df = self.get_coefficients()
            
            # Build summary string
            lines = [
                "Generalized Linear Mixed Model (lme4) Summary",
                "=" * 79,
                f"Formula: {self.formula}",
                f"Family: {self.family} ({self.link} link)",
                "",
                "Fixed Effects:",
                "-" * 79
            ]
            
            # Format coefficients table
            summary_df = coef_df[['feature', 'coefficient', 'std_error', 'p_value', 'ci_lower', 'ci_upper']].copy()
            summary_df.columns = ['term', 'coef', 'std err', 'P>|z|', '[0.025', '0.975]']
            
            # Format for display
            for col in ['coef', 'std err', 'P>|z|', '[0.025', '0.975]']:
                summary_df[col] = summary_df[col].apply(lambda x: f"{x:.4f}")
            
            lines.append(summary_df.to_string(index=False))
            
            # Add AIC/BIC
            aic_bic = self.get_aic_bic()
            lines.extend([
                "",
                f"AIC: {aic_bic['aic']:.2f}",
                f"BIC: {aic_bic['bic']:.2f}"
            ])
            
            return "\n".join(lines)
            
        except Exception as e:
            return f"Error generating summary: {e}"
    
    def get_aic_bic(self) -> dict:
        """Get AIC and BIC values.
        
        Returns:
            Dictionary with AIC and BIC values
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before getting AIC/BIC")
        
        try:
            # Extract AIC and BIC from model
            aic = float(lme4.AIC(self.model)[0])
            bic = float(lme4.BIC(self.model)[0])
            
            return {
                'aic': aic,
                'bic': bic
            }
        except Exception as e:
            raise ValueError(f"Error extracting AIC/BIC: {e}")
    
    def get_r_model(self):
        """Get the underlying R model object for advanced operations.
        
        Returns:
            R model object (can be used with rpy2 for advanced operations)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before accessing R model")
        return self.model
