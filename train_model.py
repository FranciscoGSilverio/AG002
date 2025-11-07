import sys
import os

# Add project root to path - works regardless of where script is run from
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import config
from src.data_utils import (
    load_data, explore_data, preprocess_data, split_data
)
from src.model_utils import (
    create_classifier, save_model_and_scaler, train_model, evaluate_model, 
    cross_validate_model, predict_iris
) 

def main():
    """Main training pipeline."""
    print("=" * 60)
    print("FLOWER SPECIES CLASSIFICATION - MODEL TRAINING")
    print("=" * 60)
    
    # Load data
    print("\nLoading data...")
    df = load_data(config.RAW_DATA_PATH)
    
    # Explore data
    explore_data(df)

    # Preprocess data
    print("\n" + "=" * 60)
    print("Preprocessing data...")
    X, y, scaler = preprocess_data(df, scale_features=True)

    # Split data
    X_train, X_test, y_train, y_test = split_data(X, y)

    # Train multiple models and compare
    print("\n" + "=" * 60)
    print("Training multiple models...")
    
    models_to_try = {
        'Random Forest': 'random_forest',
        'K-Nearest Neighbors': 'knn',
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
        metrics = evaluate_model(model, X_test, y_test, detailed=True, model_name=model_name)
        results[model_name] = metrics
        
        # Cross-validation
        cross_validate_model(model, X_train, y_train)

    # Select best model
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
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETED SUCCESSFULLY!")
    
    # Step 8: Save best model and scaler
    print("\nSaving best model and scaler...")
    save_model_and_scaler(best_model, scaler)
    
    print("\nPlease enter features to make a prediction:")
    sepal_length = float(input("Sepal Length: "))
    sepal_width  = float(input("Sepal Width: "))
    petal_length = float(input("Petal Length: "))
    petal_width  = float(input("Petal Width: "))
    
    species =  predict_iris(sepal_length, sepal_width, petal_length, petal_width)

    print("Predicted species:", species)

if __name__ == "__main__":
    main()
