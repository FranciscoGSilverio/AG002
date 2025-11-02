"""
Main script for training flower species classification model.

This script loads the iris dataset, preprocesses it, trains multiple classifiers,
and saves the best performing model.
"""

import sys
import os

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from src.data_utils import (
    load_data, explore_data, preprocess_data, split_data, create_sample_data
)
from src.model_utils import (
    create_classifier, train_model, evaluate_model, 
    cross_validate_model, save_model
)
from src.visualization import (
    plot_feature_distributions, plot_correlation_matrix,
    plot_pairplot, plot_confusion_matrix, plot_feature_importance
)


def main():
    """Main training pipeline."""
    print("=" * 60)
    print("FLOWER SPECIES CLASSIFICATION - MODEL TRAINING")
    print("=" * 60)
    
    # Step 1: Create sample data if it doesn't exist
    if not os.path.exists(config.RAW_DATA_PATH):
        print("\nCreating sample iris dataset...")
        create_sample_data(config.RAW_DATA_PATH)
    
    # Step 2: Load data
    print("\nLoading data...")
    df = load_data(config.RAW_DATA_PATH)
    
    # Step 3: Explore data
    explore_data(df)
    
    # Step 4: Create visualizations
    print("\n" + "=" * 60)
    print("Creating visualizations...")
    try:
        plot_correlation_matrix(
            df, 
            save_path=os.path.join(config.RESULTS_DIR, 'correlation_matrix.png')
        )
        plot_feature_distributions(
            df,
            save_path=os.path.join(config.RESULTS_DIR, 'feature_distributions.png')
        )
        print("Visualizations saved to results directory")
    except Exception as e:
        print(f"Warning: Could not create visualizations: {e}")
    
    # Step 5: Preprocess data
    print("\n" + "=" * 60)
    print("Preprocessing data...")
    X, y, scaler = preprocess_data(df, scale_features=True)
    
    # Step 6: Split data
    X_train, X_test, y_train, y_test = split_data(X, y)
    
    # Step 7: Train multiple models and compare
    print("\n" + "=" * 60)
    print("Training multiple models...")
    
    models_to_try = {
        'Random Forest': 'random_forest',
        'Logistic Regression': 'logistic_regression',
        'SVM': 'svm',
        'K-Nearest Neighbors': 'knn',
        'Decision Tree': 'decision_tree'
    }
    
    results = {}
    trained_models = {}
    
    for model_name, model_type in models_to_try.items():
        print(f"\n{'-' * 60}")
        print(f"Training {model_name}...")
        
        # Create and train model
        model = create_classifier(model_type)
        model = train_model(model, X_train, y_train)
        trained_models[model_name] = model
        
        # Evaluate model
        metrics = evaluate_model(model, X_test, y_test, detailed=True)
        results[model_name] = metrics
        
        # Cross-validation
        cv_results = cross_validate_model(model, X_train, y_train)
    
    # Step 8: Select best model
    print("\n" + "=" * 60)
    print("MODEL COMPARISON SUMMARY")
    print("=" * 60)
    for model_name, metrics in results.items():
        print(f"\n{model_name}:")
        print(f"  Accuracy:  {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall:    {metrics['recall']:.4f}")
        print(f"  F1 Score:  {metrics['f1_score']:.4f}")
    
    # Find best model by accuracy
    best_model_name = max(results, key=lambda x: results[x]['accuracy'])
    best_model = trained_models[best_model_name]
    best_accuracy = results[best_model_name]['accuracy']
    
    print(f"\n{'=' * 60}")
    print(f"BEST MODEL: {best_model_name}")
    print(f"Accuracy: {best_accuracy:.4f}")
    print("=" * 60)
    
    # Step 9: Save best model
    print(f"\nSaving best model...")
    save_model(best_model, config.MODEL_PATH)
    
    # Step 10: Create final visualizations
    print("\nCreating final visualizations...")
    try:
        # Confusion matrix for best model
        y_pred = best_model.predict(X_test)
        plot_confusion_matrix(
            y_test, 
            y_pred,
            labels=sorted(y_test.unique()),
            save_path=os.path.join(config.RESULTS_DIR, 'confusion_matrix.png')
        )
        
        # Feature importance (if available)
        if hasattr(best_model, 'feature_importances_'):
            plot_feature_importance(
                best_model,
                config.FEATURE_COLUMNS,
                save_path=os.path.join(config.RESULTS_DIR, 'feature_importance.png')
            )
    except Exception as e:
        print(f"Warning: Could not create final visualizations: {e}")
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print(f"\nModel saved to: {config.MODEL_PATH}")
    print(f"Results saved to: {config.RESULTS_DIR}")


if __name__ == "__main__":
    main()
