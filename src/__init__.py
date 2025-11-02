"""
Flower Species Classification Package

A machine learning package for classifying flower species using pandas and scikit-learn.
"""

__version__ = '1.0.0'
__author__ = 'Francisco G. Silverio'

from . import data_utils
from . import model_utils
from . import visualization

__all__ = ['data_utils', 'model_utils', 'visualization']
