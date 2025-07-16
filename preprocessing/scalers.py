import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
import warnings
from typing import Optional, Any

class SymlogScaler(BaseEstimator, TransformerMixin):
    """
    Scaler that applies a symmetric logarithmic (symlog) transformation.
    Useful for data with both positive and negative values spanning several orders of magnitude.

    Parameters
    ----------
    linthresh : float, optional
        The linear threshold for the symlog transform. If None, will be determined from data using percentile.
    percentile : float, optional
        Percentile (0-100) of |x| to use for linthresh. If both linthresh and percentile are given, percentile takes precedence.
        Default is 90 (i.e., 90th percentile).

    Notes
    -----
    - Zeros are ignored when computing linthresh from data.
    - If all data is zero, a fallback linthresh of 1e-3 is used and a warning is issued.
    - After fitting, the chosen linthresh is stored as an attribute.
    """
    def __init__(self, linthresh: Optional[float] = None, percentile: Optional[float] = 90):
        self.linthresh = linthresh
        self.percentile = percentile
        self.fitted_ = False

    @staticmethod
    def compute_linthresh(X: np.ndarray, percentile: float) -> float:
        X = np.asarray(X)
        abs_nonzero = np.abs(X[X != 0])
        if abs_nonzero.size == 0:
            warnings.warn("All data is zero; using fallback linthresh=1e-3.")
            return 1e-3
        return np.percentile(abs_nonzero, percentile)

    def fit(self, X: Any, y: Any = None) -> 'SymlogScaler':
        X = np.asarray(X)
        if self.linthresh is not None:
            self.linthresh_ = self.linthresh
        elif self.percentile is not None:
            self.linthresh_ = self.compute_linthresh(X, self.percentile)
        self.fitted_ = True
        return self

    def transform(self, X: Any) -> np.ndarray:
        if not getattr(self, 'fitted_', False):
            raise RuntimeError("SymlogScaler instance is not fitted yet. Call 'fit' before using this method.")
        X = np.asarray(X)
        return np.sign(X) * np.log1p(np.abs(X) / self.linthresh_)

    def inverse_transform(self, X: Any) -> np.ndarray:
        if not getattr(self, 'fitted_', False):
            raise RuntimeError("SymlogScaler instance is not fitted yet. Call 'fit' before using this method.")
        X = np.asarray(X)
        return np.sign(X) * (np.expm1(np.abs(X)) * self.linthresh_)

def get_scaler(name: str, **kwargs) -> Any:
    """
    Return a scaler instance by name. Supported: 'standard', 'symlog'.
    """
    name = name.lower()
    if name == 'standard':
        return StandardScaler(**kwargs)
    elif name == 'symlog':
        return SymlogScaler(**kwargs)
    else:
        raise ValueError(f"Unknown scaler: {name}. Supported: 'standard', 'symlog'") 