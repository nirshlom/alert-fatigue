from .base import BaseBinaryClassifier
from .statsmodels_logit import StatsmodelsLogitModel

# Conditionally import LME4GLMMModel (requires rpy2 and R)
try:
    from .lme4_glmm import LME4GLMMModel
    __all__ = ['BaseBinaryClassifier', 'StatsmodelsLogitModel', 'LME4GLMMModel']
except ImportError:
    # rpy2 or R packages not available
    __all__ = ['BaseBinaryClassifier', 'StatsmodelsLogitModel']
