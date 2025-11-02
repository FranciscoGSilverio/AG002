"""
Script for making predictions using trained flower species classification model.

This script loads a trained model and makes predictions on new data.
"""

import sys
import os
import argparse

# Add project root to path - works regardless of where script is run from
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import pandas as pd
import config
from src.data_utils import load_data, preprocess_data
from src.model_utils import load_model, predict, predict_proba


def predict_from_csv(csv_path: str, model_path: str = None) -> pd.DataFrame:
    """
    Make predictions on data from a CSV file.
    
    Args:
        csv_path: Path to CSV file with data
        model_path: Path to trained model (uses default if not provided)
        
    Returns:
        DataFrame with predictions
    """
    if model_path is None:
        model_path = config.MODEL_PATH
    
    # Load model
    print(f"Loading model from {model_path}...")
    model = load_model(model_path)
    
    # Load data
    print(f"Loading data from {csv_path}...")
    df = load_data(csv_path)
    
    # Check if data has all required features
    missing_features = set(config.FEATURE_COLUMNS) - set(df.columns)
    if missing_features:
        raise ValueError(f"Missing required features: {missing_features}")
    
    # Extract features
    X = df[config.FEATURE_COLUMNS]
    
    # Make predictions
    print("Making predictions...")
    predictions = predict(model, X)
    
    # Try to get probabilities
    try:
        probabilities = predict_proba(model, X)
        prob_df = pd.DataFrame(
            probabilities,
            columns=[f'prob_{cls}' for cls in model.classes_]
        )
        result_df = pd.concat([df, prob_df], axis=1)
    except AttributeError:
        result_df = df.copy()
    
    result_df['predicted_species'] = predictions
    
    return result_df


def predict_from_values(sepal_length: float, 
                       sepal_width: float,
                       petal_length: float,
                       petal_width: float,
                       model_path: str = None) -> dict:
    """
    Make a prediction from individual feature values.
    
    Args:
        sepal_length: Sepal length in cm
        sepal_width: Sepal width in cm
        petal_length: Petal length in cm
        petal_width: Petal width in cm
        model_path: Path to trained model (uses default if not provided)
        
    Returns:
        Dictionary with prediction and probabilities
    """
    if model_path is None:
        model_path = config.MODEL_PATH
    
    # Load model
    model = load_model(model_path)
    
    # Create DataFrame with single row
    X = pd.DataFrame({
        'sepal_length': [sepal_length],
        'sepal_width': [sepal_width],
        'petal_length': [petal_length],
        'petal_width': [petal_width]
    })
    
    # Make prediction
    prediction = predict(model, X)[0]
    
    # Try to get probabilities
    result = {'predicted_species': prediction}
    try:
        probabilities = predict_proba(model, X)[0]
        result['probabilities'] = {
            cls: prob for cls, prob in zip(model.classes_, probabilities)
        }
    except AttributeError:
        pass
    
    return result


def main():
    """Main prediction pipeline with CLI interface."""
    parser = argparse.ArgumentParser(
        description='Make predictions using trained flower classifier'
    )
    
    parser.add_argument(
        '--csv',
        type=str,
        help='Path to CSV file with data to predict'
    )
    parser.add_argument(
        '--sepal-length',
        type=float,
        help='Sepal length in cm'
    )
    parser.add_argument(
        '--sepal-width',
        type=float,
        help='Sepal width in cm'
    )
    parser.add_argument(
        '--petal-length',
        type=float,
        help='Petal length in cm'
    )
    parser.add_argument(
        '--petal-width',
        type=float,
        help='Petal width in cm'
    )
    parser.add_argument(
        '--model',
        type=str,
        default=config.MODEL_PATH,
        help=f'Path to trained model (default: {config.MODEL_PATH})'
    )
    parser.add_argument(
        '--output',
        type=str,
        help='Path to save predictions CSV (only for --csv mode)'
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("FLOWER SPECIES CLASSIFICATION - PREDICTION")
    print("=" * 60)
    
    # Check if model exists
    if not os.path.exists(args.model):
        print(f"\nError: Model not found at {args.model}")
        print("Please train a model first using train_model.py")
        sys.exit(1)
    
    # CSV mode
    if args.csv:
        if not os.path.exists(args.csv):
            print(f"Error: CSV file not found at {args.csv}")
            sys.exit(1)
        
        results = predict_from_csv(args.csv, args.model)
        
        print("\nPredictions:")
        print(results[['predicted_species'] + [col for col in results.columns if col.startswith('prob_')]])
        
        if args.output:
            results.to_csv(args.output, index=False)
            print(f"\nPredictions saved to {args.output}")
    
    # Individual values mode
    elif all([args.sepal_length, args.sepal_width, args.petal_length, args.petal_width]):
        result = predict_from_values(
            args.sepal_length,
            args.sepal_width,
            args.petal_length,
            args.petal_width,
            args.model
        )
        
        print("\nInput features:")
        print(f"  Sepal length: {args.sepal_length} cm")
        print(f"  Sepal width:  {args.sepal_width} cm")
        print(f"  Petal length: {args.petal_length} cm")
        print(f"  Petal width:  {args.petal_width} cm")
        
        print(f"\nPredicted species: {result['predicted_species']}")
        
        if 'probabilities' in result:
            print("\nClass probabilities:")
            for species, prob in result['probabilities'].items():
                print(f"  {species}: {prob:.4f}")
    
    else:
        parser.print_help()
        print("\nError: Please provide either --csv or all individual feature values")
        sys.exit(1)
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
