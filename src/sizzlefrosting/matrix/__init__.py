"""
Arithmetic subpackage containing matrix operations.
"""

from .elementary import rowswap, rowscale, rowreplacement, initial_swap, find_scalar, rref

__all__ = ['rowswap', 'rowscale', 'rowreplacement', 'initial_swap', 'find_scalar', 'rref']
