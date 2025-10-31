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

class AsinhScaler(BaseEstimator, TransformerMixin):
    """
    Scaler that applies an inverse hyperbolic sine (asinh) transformation with a data-driven scale.

    Transformation: x_scaled = asinh(x / C)
    Fitting: C is set to the median of absolute values in the data (median(|x|)).

    Notes
    -----
    - If median(|x|) == 0, a small fallback constant of 1e-6 is used and a warning is issued.
    - After fitting, the chosen constant is stored as `scale_constant_`.
    """
    def __init__(self, scale_constant: Optional[float] = None):
        self.scale_constant = scale_constant
        self.fitted_ = False

    @staticmethod
    def compute_scale_constant(X: np.ndarray) -> float:
        X = np.asarray(X)
        median_abs = np.median(np.abs(X))
        if median_abs == 0:
            warnings.warn("Median of absolute values is zero; using fallback scale_constant=1e-6.")
            return 1e-6
        return float(median_abs)

    def fit(self, X: Any, y: Any = None) -> 'AsinhScaler':
        X = np.asarray(X)
        if self.scale_constant is not None:
            self.scale_constant_ = float(self.scale_constant)
        else:
            self.scale_constant_ = self.compute_scale_constant(X)
        self.fitted_ = True
        return self

    def transform(self, X: Any) -> np.ndarray:
        if not getattr(self, 'fitted_', False):
            raise RuntimeError("AsinhScaler instance is not fitted yet. Call 'fit' before using this method.")
        X = np.asarray(X)
        return np.arcsinh(X / self.scale_constant_)

    def inverse_transform(self, X: Any) -> np.ndarray:
        if not getattr(self, 'fitted_', False):
            raise RuntimeError("AsinhScaler instance is not fitted yet. Call 'fit' before using this method.")
        X = np.asarray(X)
        return np.sinh(X) * self.scale_constant_

def get_scaler(name: str, **kwargs) -> Any:
    """
    Return a scaler instance by name. Supported: 'standard', 'symlog', 'asinh'.
    """
    name = name.lower()
    if name == 'standard':
        return StandardScaler(**kwargs)
    elif name == 'symlog':
        return SymlogScaler(**kwargs)
    elif name == 'asinh':
        return AsinhScaler(**kwargs)
    else:
        raise ValueError(f"Unknown scaler: {name}. Supported: 'standard', 'symlog', 'asinh'")

def get_fitted_attributes(scaler):
    """
    Return a dictionary of all public fitted attributes (ending with '_') for a scaler/estimator.
    This includes attributes like mean_, scale_, var_, linthresh_, etc.
    """
    return {k: getattr(scaler, k) for k in dir(scaler)
            if k.endswith('_') and not k.startswith('_') and not callable(getattr(scaler, k))} 