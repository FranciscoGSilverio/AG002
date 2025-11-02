"""
Configuration file for the flower species classification project.
"""

import os

# Project paths
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results')

# Data file paths
RAW_DATA_PATH = os.path.join(DATA_DIR, 'iris.csv')
PROCESSED_DATA_PATH = os.path.join(DATA_DIR, 'processed_iris.csv')

# Model paths
MODEL_PATH = os.path.join(MODELS_DIR, 'flower_classifier.pkl')

# Model parameters
TEST_SIZE = 0.2
RANDOM_STATE = 42
N_SPLITS = 5  # for cross-validation

# Feature columns (for Iris dataset)
FEATURE_COLUMNS = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
TARGET_COLUMN = 'species'

# Create directories if they don't exist
for directory in [DATA_DIR, MODELS_DIR, RESULTS_DIR]:
    os.makedirs(directory, exist_ok=True)
