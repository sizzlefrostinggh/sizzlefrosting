"""
Distributions module containing uniform and exponentialdist functions
"""

import secrets

def uniform(a: float = 0.0, b: float = 1.0) -> float:
    """Cryptographically secure uniform sample."""
    # 53 random bits gives 53-bit precision double
    u = secrets.randbits(53) / (1 << 53)   # in [0, 1)
    return a + (b - a) * u

import math

def exponentialdist(lam: float) -> float:
    """Cryptographically secure exponential sample."""
    # Generate uniform sample in [0, 1)
    u = secrets.randbits(53) / (1 << 53)
    # Inverse transform sampling formula: X = - (1 / lambda) * ln(U)
    return - (1.0 / lam) * math.log(u)