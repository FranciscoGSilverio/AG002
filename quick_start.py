#!/usr/bin/env python
"""
Quick Start Example

This script demonstrates the basic usage of the flower classification project.
"""

import os
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import config
from src.data_utils import load_data, preprocess_data, split_data, create_sample_data
from src.model_utils import create_classifier, train_model, evaluate_model, save_model, load_model


def quick_start():
    """Quick start guide for the flower classification project."""
    
    print("=" * 60)
    print("FLOWER CLASSIFICATION - QUICK START GUIDE")
    print("=" * 60)
    
    # Step 1: Create sample data
    print("\n[1/5] Creating sample dataset...")
    if not os.path.exists(config.RAW_DATA_PATH):
        create_sample_data(config.RAW_DATA_PATH)
    else:
        print(f"Dataset already exists at {config.RAW_DATA_PATH}")
    
    # Step 2: Load and preprocess data
    print("\n[2/5] Loading and preprocessing data...")
    df = load_data(config.RAW_DATA_PATH)
    X, y, scaler = preprocess_data(df)
    X_train, X_test, y_train, y_test = split_data(X, y)
    print(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")
    
    # Step 3: Train a simple model
    print("\n[3/5] Training a Random Forest classifier...")
    model = create_classifier('random_forest')
    model = train_model(model, X_train, y_train)
    
    # Step 4: Evaluate the model
    print("\n[4/5] Evaluating model performance...")
    metrics = evaluate_model(model, X_test, y_test, detailed=True)
    
    # Step 5: Save the model
    print("\n[5/5] Saving the trained model...")
    save_model(model, config.MODEL_PATH)
    
    print("\n" + "=" * 60)
    print("QUICK START COMPLETED!")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Run 'python train_model.py' to train and compare multiple models")
    print("2. Run 'python predict.py --help' to see prediction options")
    print("3. Open 'flower_classification_example.ipynb' for interactive examples")
    print("\nExample prediction command:")
    print("  python predict.py --sepal-length 5.1 --sepal-width 3.5 --petal-length 1.4 --petal-width 0.2")


if __name__ == "__main__":
    try:
        quick_start()
    except Exception as e:
        print(f"\nError: {e}")
        print("\nPlease make sure all dependencies are installed:")
        print("  pip install -r requirements.txt")
        sys.exit(1)
