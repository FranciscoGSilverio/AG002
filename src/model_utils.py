"""
Model training and evaluation utilities for flower species classification.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)
import pickle
from typing import Dict, Any, Tuple
import config


def create_classifier(model_type: str = 'random_forest', **kwargs) -> Any:
    """
    Create a classifier model.
    
    Args:
        model_type: Type of classifier ('random_forest', 'logistic_regression', 'svm', 'knn', 'decision_tree')
        **kwargs: Additional parameters for the classifier
        
    Returns:
        Initialized classifier object
    """
    models = {
        'random_forest': RandomForestClassifier(random_state=config.RANDOM_STATE, **kwargs),
        'logistic_regression': LogisticRegression(random_state=config.RANDOM_STATE, max_iter=1000, **kwargs),
        'svm': SVC(random_state=config.RANDOM_STATE, **kwargs),
        'knn': KNeighborsClassifier(**kwargs),
        'decision_tree': DecisionTreeClassifier(random_state=config.RANDOM_STATE, **kwargs)
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


def evaluate_model(model: Any, 
                  X_test: pd.DataFrame, 
                  y_test: pd.Series,
                  detailed: bool = True) -> Dict[str, float]:
    """
    Evaluate a trained model on test data.
    
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


def hyperparameter_tuning(model_type: str,
                         X_train: pd.DataFrame,
                         y_train: pd.Series,
                         param_grid: Dict[str, list]) -> Tuple[Any, Dict[str, Any]]:
    """
    Perform hyperparameter tuning using GridSearchCV.
    
    Args:
        model_type: Type of classifier
        X_train: Training features
        y_train: Training labels
        param_grid: Dictionary of parameters to search
        
    Returns:
        Tuple of (best model, best parameters)
    """
    print(f"\nPerforming hyperparameter tuning for {model_type}...")
    
    base_model = create_classifier(model_type)
    
    grid_search = GridSearchCV(
        base_model,
        param_grid,
        cv=config.N_SPLITS,
        scoring='accuracy',
        n_jobs=-1,
        verbose=1
    )
    
    grid_search.fit(X_train, y_train)
    
    print(f"\nBest parameters: {grid_search.best_params_}")
    print(f"Best cross-validation score: {grid_search.best_score_:.4f}")
    
    return grid_search.best_estimator_, grid_search.best_params_


def save_model(model: Any, filepath: str) -> None:
    """
    Save a trained model to disk.
    
    Args:
        model: Trained model to save
        filepath: Path where to save the model
    """
    with open(filepath, 'wb') as f:
        pickle.dump(model, f)
    print(f"Model saved to {filepath}")


def load_model(filepath: str) -> Any:
    """
    Load a trained model from disk.
    
    Args:
        filepath: Path to the saved model
        
    Returns:
        Loaded model object
    """
    with open(filepath, 'rb') as f:
        model = pickle.load(f)
    print(f"Model loaded from {filepath}")
    return model


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


def predict_proba(model: Any, X: pd.DataFrame) -> np.ndarray:
    """
    Get prediction probabilities from a trained model.
    
    Args:
        model: Trained classifier
        X: Features to predict
        
    Returns:
        Array of prediction probabilities
    """
    if hasattr(model, 'predict_proba'):
        probabilities = model.predict_proba(X)
        return probabilities
    else:
        raise AttributeError(f"{type(model).__name__} does not support probability predictions")
