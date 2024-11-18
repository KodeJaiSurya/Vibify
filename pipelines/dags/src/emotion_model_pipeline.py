import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense
from typing import Dict, Tuple
from tensorflow.keras.callbacks import History

"""
def load_data(X_path: str = 'X.npy', y_path: str = 'y.npy', test_size: float = 0.2, random_state: int = 42) -> Dict[str, np.ndarray]:

    Loads preprocessed data from .npy files and splits it into training and testing sets.
    
    Args:
        X_path (str): Path to the preprocessed features file (X.npy).
        y_path (str): Path to the preprocessed labels file (y.npy).
        test_size (float): Proportion of the dataset to include in the test split.
        random_state (int): Seed used by the random number generator.
    
    Returns:
        dict: A dictionary containing X_train, X_test, y_train, y_test.
    
    from sklearn.model_selection import train_test_split

    X = np.load(X_path)
    y = np.load(y_path)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    return {'X_train': X_train, 'X_test': X_test, 'y_train': y_train, 'y_test': y_test}
"""
def load_data(X_path: str = 'X.npy', y_path: str = 'y.npy', test_size: float = 0.2, random_state: int = 42) -> Dict[str, np.ndarray]:
    """
    Loads preprocessed data from .npy files and splits it into training and testing sets.
   
    Args:
        X_path (str): Path to the preprocessed features file (X.npy).
        y_path (str): Path to the preprocessed labels file (y.npy).
        test_size (float): Proportion of the dataset to include in the test split.
        random_state (int): Seed used by the random number generator.
   
    Returns:
        dict: A dictionary containing X_train, X_test, y_train, y_test.
    """
 
    X = np.load(X_path)
    y = np.load(y_path)
    indices = pd.Series(range(len(X)))
   
    X_train, X_test, y_train, y_test, _, indices_test = train_test_split(
        X, y, indices, test_size=test_size, random_state=random_state
    )
   
    return {'X_train': X_train, 'X_test': X_test, 'y_train': y_train, 'y_test': y_test, 'indices_test': indices_test}


def create_model(input_shape: Tuple[int, ...]) -> Sequential:
    """
    Defines and returns the model architecture.
    
    Args:
        input_shape (tuple): Shape of the input data.
    
    Returns:
        Sequential: A Keras Sequential model instance.
    """
    model = Sequential([
        Dense(128, activation='relu', input_shape=input_shape),
        Dense(64, activation='relu'),
        Dense(4, activation='softmax')  # Assuming 4 output classes
    ])
    
    return model


def compile_model(model: Sequential) -> Sequential:
    """
    Compiles the model with specified optimizer, loss, and metrics.
    
    Args:
        model (Sequential): A Keras model instance.
    
    Returns:
        Sequential: Compiled Keras model.
    """
    model.compile(
        optimizer='adam', 
        loss='sparse_categorical_crossentropy', 
        metrics=['accuracy']
    )
    
    return model


def train_model(model: Sequential, X_train: np.ndarray, y_train: np.ndarray, 
                epochs: int = 30, batch_size: int = 64, validation_split: float = 0.2) -> Dict[str, object]:
    """
    Trains the model on the provided data.
    
    Args:
        model (Sequential): Compiled Keras model.
        X_train (np.ndarray): Training features.
        y_train (np.ndarray): Training labels.
        epochs (int): Number of training epochs.
        batch_size (int): Size of training batches.
        validation_split (float): Fraction of training data to use for validation.
    
    Returns:
        dict: A dictionary containing the trained model and training history.
    """
    history: History = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=validation_split,
        verbose=1
    )
    
    return {'model': model, 'history': history.history}


def save_model(model: Model, filename: str = 'emotion_detection_model_4_classes.h5') -> str:
    """
    Saves the trained model to a file.
    
    Args:
        model (Model): Trained Keras model.
        filename (str): Path to save the model file.
    
    Returns:
        str: The path to the saved model file.
    """
    model.save(filename)
    print(f"Model saved as {filename}")
    return filename
