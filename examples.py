"""
Advanced Examples

This script demonstrates advanced usage of the flower classification project,
including hyperparameter tuning and model comparison.
"""

import os
import sys

# Add project root to path - works regardless of where script is run from
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import config
from src.data_utils import load_data, preprocess_data, split_data, create_sample_data
from src.model_utils import (
    create_classifier, train_model, evaluate_model, 
    cross_validate_model, hyperparameter_tuning
)
from src.visualization import plot_model_comparison, plot_feature_importance


def example_1_basic_workflow():
    """Example 1: Basic machine learning workflow."""
    print("\n" + "=" * 60)
    print("EXAMPLE 1: Basic Workflow")
    print("=" * 60)
    
    # Create sample data if needed
    if not os.path.exists(config.RAW_DATA_PATH):
        create_sample_data(config.RAW_DATA_PATH)
    
    # Load and preprocess
    df = load_data(config.RAW_DATA_PATH)
    X, y, scaler = preprocess_data(df)
    X_train, X_test, y_train, y_test = split_data(X, y)
    
    # Train a model
    model = create_classifier('random_forest')
    model = train_model(model, X_train, y_train)
    
    # Evaluate
    metrics = evaluate_model(model, X_test, y_test)
    
    print(f"\nModel Accuracy: {metrics['accuracy']:.2%}")


def example_2_model_comparison():
    """Example 2: Compare multiple models."""
    print("\n" + "=" * 60)
    print("EXAMPLE 2: Model Comparison")
    print("=" * 60)
    
    # Load data
    df = load_data(config.RAW_DATA_PATH)
    X, y, scaler = preprocess_data(df)
    X_train, X_test, y_train, y_test = split_data(X, y)
    
    # Models to compare
    models = {
        'Random Forest': create_classifier('random_forest'),
        'Logistic Regression': create_classifier('logistic_regression'),
        'SVM': create_classifier('svm')
    }
    
    results = {}
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        model = train_model(model, X_train, y_train)
        metrics = evaluate_model(model, X_test, y_test, detailed=False)
        results[name] = metrics
    
    # Print comparison
    print("\n" + "-" * 60)
    print("Model Comparison:")
    print("-" * 60)
    for name, metrics in results.items():
        print(f"{name:20} - Accuracy: {metrics['accuracy']:.4f}")


def example_3_cross_validation():
    """Example 3: Cross-validation for robust evaluation."""
    print("\n" + "=" * 60)
    print("EXAMPLE 3: Cross-Validation")
    print("=" * 60)
    
    # Load data
    df = load_data(config.RAW_DATA_PATH)
    X, y, scaler = preprocess_data(df)
    
    # Create model
    model = create_classifier('random_forest')
    
    # Perform cross-validation
    cv_results = cross_validate_model(model, X, y, cv=5)
    
    print(f"\nCross-validation accuracy: {cv_results['mean_score']:.2%} "
          f"(+/- {cv_results['std_score']:.2%})")


def example_4_hyperparameter_tuning():
    """Example 4: Hyperparameter tuning with GridSearchCV."""
    print("\n" + "=" * 60)
    print("EXAMPLE 4: Hyperparameter Tuning")
    print("=" * 60)
    
    # Load data
    df = load_data(config.RAW_DATA_PATH)
    X, y, scaler = preprocess_data(df)
    X_train, X_test, y_train, y_test = split_data(X, y)
    
    # Define parameter grid for Random Forest
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [3, 5, 10, None],
        'min_samples_split': [2, 5, 10]
    }
    
    print("\nSearching for best hyperparameters...")
    best_model, best_params = hyperparameter_tuning(
        'random_forest',
        X_train,
        y_train,
        param_grid
    )
    
    # Evaluate best model
    metrics = evaluate_model(best_model, X_test, y_test, detailed=False)
    
    print(f"\nBest parameters: {best_params}")
    print(f"Test accuracy: {metrics['accuracy']:.2%}")


def example_5_custom_model_params():
    """Example 5: Creating models with custom parameters."""
    print("\n" + "=" * 60)
    print("EXAMPLE 5: Custom Model Parameters")
    print("=" * 60)
    
    # Load data
    df = load_data(config.RAW_DATA_PATH)
    X, y, scaler = preprocess_data(df)
    X_train, X_test, y_train, y_test = split_data(X, y)
    
    # Create Random Forest with custom parameters
    rf_model = create_classifier('random_forest', n_estimators=200, max_depth=5)
    rf_model = train_model(rf_model, X_train, y_train)
    
    # Create KNN with custom parameters
    knn_model = create_classifier('knn', n_neighbors=3)
    knn_model = train_model(knn_model, X_train, y_train)
    
    # Evaluate both
    print("\nRandom Forest (n_estimators=200, max_depth=5):")
    rf_metrics = evaluate_model(rf_model, X_test, y_test, detailed=False)
    print(f"Accuracy: {rf_metrics['accuracy']:.2%}")
    
    print("\nKNN (n_neighbors=3):")
    knn_metrics = evaluate_model(knn_model, X_test, y_test, detailed=False)
    print(f"Accuracy: {knn_metrics['accuracy']:.2%}")


def main():
    """Run all examples."""
    print("=" * 60)
    print("FLOWER CLASSIFICATION - ADVANCED EXAMPLES")
    print("=" * 60)
    
    # Ensure data exists
    if not os.path.exists(config.RAW_DATA_PATH):
        print("\nCreating sample dataset...")
        create_sample_data(config.RAW_DATA_PATH)
    
    # Run examples
    try:
        example_1_basic_workflow()
        example_2_model_comparison()
        example_3_cross_validation()
        example_4_hyperparameter_tuning()
        example_5_custom_model_params()
        
        print("\n" + "=" * 60)
        print("ALL EXAMPLES COMPLETED!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\nError running examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
