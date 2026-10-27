
"""
Regression subpackage containing linear, Cauchy,
and logistic regression models.
"""

from .regression import LinearRegression, CauchyRegression, CauchyModel
from .logit import LogisticRegression

__all__ = [
    "LinearRegression",
    "CauchyModel",
    "CauchyRegression",
    "LogisticRegression",
]

