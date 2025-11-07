"""
Model training and evaluation utilities for flower species classification.
"""

import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import (
    accuracy_score, confusion_matrix, precision_score, recall_score, f1_score,
    classification_report
)
from typing import Dict, Any
import config
import joblib
from config import MODELS_DIR

def create_classifier(model_type: str = 'random_forest', **kwargs) -> Any:
    """
    Create a classifier model.
    
    Args:
        model_type: Type of classifier ('random_forest', 'knn')
        **kwargs: Additional parameters for the classifier
        
    Returns:
        Initialized classifier object
    """
    models = {
        'random_forest': RandomForestClassifier(random_state=config.RANDOM_STATE, **kwargs),
        'knn': KNeighborsClassifier(**kwargs),
    }
    
    if model_type not in models:
        raise ValueError(f"Unknown model type: {model_type}. Available: {list(models.keys())}")
    
    return models[model_type]


def train_model(model: Any, X_train: pd.DataFrame, y_train: pd.Series) -> Any:
    """
    Train a classifier model.
    
    Args:
        model: Classifier object
        X_train: Training features
        y_train: Training labels
        
    Returns:
        Trained model
    """
    print(f"\nTraining {type(model).__name__}...")
    model.fit(X_train, y_train)
    print("Training completed!")
    return model


def evaluate_model(
    model: Any, 
    X_test: pd.DataFrame, 
    y_test: pd.Series,
    detailed: bool = True,
    model_name: str = "model"
) -> Dict[str, float]:
    """
    Evaluate a trained model on test data and save confusion matrix plot.
    
    Args:
        model: Trained classifier
        X_test: Test features
        y_test: Test labels
        detailed: Whether to print detailed evaluation metrics
        
    Returns:
        Dictionary containing evaluation metrics
    """

    # Make predictions
    y_pred = model.predict(X_test)

    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, average='weighted'),
        'recall': recall_score(y_test, y_pred, average='weighted'),
        'f1_score': f1_score(y_test, y_pred, average='weighted')
    }

    # Print metrics
    if detailed:
        print("\n=== Model Evaluation ===")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        print(f"F1 Score: {metrics['f1_score']:.4f}")

        print("\n=== Classification Report ===")
        print(classification_report(y_test, y_pred))

        print("\n=== Confusion Matrix ===")
        print(confusion_matrix(y_test, y_pred))

    # === Save confusion matrix plot ===
    try:
        cm = confusion_matrix(y_test, y_pred)
        labels = sorted(y_test.unique())

        plt.figure(figsize=(6, 5))
        sns.heatmap(
            cm, annot=True, fmt="d",
            cmap="Blues",
            xticklabels=labels,
            yticklabels=labels
        )
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title("Confusion Matrix - " + model_name)

        save_path = os.path.join(config.METRICS_DIR, f"confusion_matrix_{model_name}.png")
        plt.savefig(save_path, bbox_inches="tight")
        plt.close()

        if detailed:
            print(f"\nConfusion matrix saved to: {save_path}")

    except Exception as e:
        print(f"Warning: Could not save confusion matrix plot: {e}")

    return metrics


def cross_validate_model(model: Any,
                        X: pd.DataFrame,
                        y: pd.Series,
                        cv: int = None) -> Dict[str, Any]:
    """
    Perform cross-validation on a model.
    
    Args:
        model: Classifier object
        X: Features
        y: Labels
        cv: Number of cross-validation folds
        
    Returns:
        Dictionary containing cross-validation results
    """
    if cv is None:
        cv = config.N_SPLITS
    
    print(f"\nPerforming {cv}-fold cross-validation...")
    scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
    
    results = {
        'scores': scores,
        'mean_score': scores.mean(),
        'std_score': scores.std()
    }
    
    print(f"Cross-validation scores: {scores}")
    print(f"Mean accuracy: {results['mean_score']:.4f} (+/- {results['std_score']:.4f})")
    
    return results


def predict(model: Any, X: pd.DataFrame) -> np.ndarray:
    """
    Make predictions using a trained model.
    
    Args:
        model: Trained classifier
        X: Features to predict
        
    Returns:
        Array of predictions
    """
    predictions = model.predict(X)
    return predictions

def save_model_and_scaler(model: Any, scaler: Any) -> None:
    """
    Save the trained model and scaler to disk.
    
    Args:
        model: Trained classifier
        scaler: Fitted scaler
        model_path: Path to save the model
        scaler_path: Path to save the scaler
    """

    joblib.dump(model, os.path.join(MODELS_DIR, "best_model.pkl"))
    joblib.dump(scaler, os.path.join(MODELS_DIR, "scaler.pkl"))
    print(f"Model saved to {os.path.join(MODELS_DIR, 'best_model.pkl')}")
    print(f"Scaler saved to {os.path.join(MODELS_DIR, 'scaler.pkl')}")
    
    
def predict_iris(sepal_length, sepal_width, petal_length, petal_width):
    # load artifacts
    model = joblib.load(os.path.join(config.MODELS_DIR, "best_model.pkl"))
    scaler = joblib.load(os.path.join(config.MODELS_DIR, "scaler.pkl"))

    # format input
    X = pd.DataFrame([{
        "sepal_length": sepal_length,
        "sepal_width": sepal_width,
        "petal_length": petal_length,
        "petal_width": petal_width
    }])

    # scale features
    X_scaled = scaler.transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=config.FEATURE_COLUMNS, index=X.index)

    # predict
    y_pred = predict(model, X_scaled)

    return y_pred[0]