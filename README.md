# Flower Species Classification Project

A machine learning project that uses pandas and scikit-learn to classify flower species from CSV data. This project demonstrates a complete ML pipeline including data preprocessing, model training, evaluation, and prediction.

## Overview

This project implements a flower species classifier using the famous Iris dataset. It includes:
- Data loading and preprocessing utilities
- Multiple classification algorithms (Random Forest, Logistic Regression, SVM, KNN, Decision Tree)
- Model evaluation and comparison
- Visualization tools for data exploration and results
- Command-line interfaces for training and prediction

## Features

- **Data Processing**: Load CSV data, handle missing values, scale features
- **Multiple Models**: Train and compare 5 different classification algorithms
- **Visualization**: Generate plots for feature distributions, correlations, and model performance
- **Model Persistence**: Save and load trained models
- **Easy Prediction**: Make predictions via command-line interface or programmatically
- **Cross-Validation**: Evaluate models using k-fold cross-validation

## Installation

1. Clone the repository:
```bash
git clone https://github.com/FranciscoGSilverio/AG002.git
cd AG002
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Project Structure

```
AG002/
│
├── config.py                  # Configuration and settings
├── train_model.py             # Main training script
├── predict.py                 # Prediction script
├── requirements.txt           # Python dependencies
│
├── src/                       # Source code modules
│   ├── data_utils.py          # Data loading and preprocessing
│   ├── model_utils.py         # Model training and evaluation
│   └── visualization.py       # Plotting and visualization
│
├── data/                      # Data directory
│   └── iris.csv               # Dataset (created automatically)
│
├── models/                    # Saved models directory
│   └── flower_classifier.pkl  # Trained model (created after training)
│
└── results/                   # Results and plots directory
    ├── correlation_matrix.png
    ├── feature_distributions.png
    ├── confusion_matrix.png
    └── feature_importance.png
```

## Usage

### Training a Model

Train the model using the iris dataset:

```bash
python train_model.py
```

This will:
1. Create sample iris dataset if it doesn't exist
2. Explore and visualize the data
3. Train multiple classification models
4. Compare their performance
5. Save the best model to `models/flower_classifier.pkl`
6. Generate visualization plots in the `results/` directory

### Making Predictions

#### From a CSV file:

```bash
python predict.py --csv path/to/data.csv --output predictions.csv
```

#### From individual values:

```bash
python predict.py --sepal-length 5.1 --sepal-width 3.5 --petal-length 1.4 --petal-width 0.2
```

#### Using a specific model:

```bash
python predict.py --csv data.csv --model path/to/model.pkl
```

### Programmatic Usage

```python
from src.data_utils import load_data, preprocess_data
from src.model_utils import load_model, predict
import config

# Load data
df = load_data('path/to/data.csv')
X, y, scaler = preprocess_data(df)

# Load trained model
model = load_model(config.MODEL_PATH)

# Make predictions
predictions = predict(model, X)
```

## Dataset

The project uses the Iris dataset, which contains 150 samples of iris flowers with:
- **Features**: 
  - Sepal length (cm)
  - Sepal width (cm)
  - Petal length (cm)
  - Petal width (cm)
- **Target**: Species (setosa, versicolor, virginica)

The dataset is automatically created when you run `train_model.py` for the first time.

## Models

The project trains and compares the following classifiers:
1. **Random Forest** - Ensemble of decision trees
2. **Logistic Regression** - Linear probabilistic classifier
3. **Support Vector Machine (SVM)** - Margin-based classifier
4. **K-Nearest Neighbors (KNN)** - Instance-based learning
5. **Decision Tree** - Tree-based classifier

## Configuration

Edit `config.py` to customize:
- Data paths
- Model parameters
- Train/test split ratio
- Cross-validation folds
- Feature and target column names

## Dependencies

- pandas >= 2.0.0
- scikit-learn >= 1.3.0
- numpy >= 1.24.0
- matplotlib >= 3.7.0
- seaborn >= 0.12.0
- jupyter >= 1.0.0 (optional, for notebooks)

## Results

After training, you'll find:
- **Model file**: `models/flower_classifier.pkl`
- **Visualizations**: `results/` directory
  - Feature correlation matrix
  - Feature distributions by species
  - Confusion matrix
  - Feature importance (for tree-based models)

## Example Output

```
BEST MODEL: Random Forest
Accuracy: 0.9667

Classification Report:
              precision    recall  f1-score   support
      setosa       1.00      1.00      1.00        10
  versicolor       1.00      0.90      0.95        10
   virginica       0.91      1.00      0.95        10

Model saved to: models/flower_classifier.pkl
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is open source and available under the MIT License.

## Author

Francisco G. Silverio