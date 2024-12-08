import numpy as np
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout, BatchNormalization
from typing import Dict, Tuple
from tensorflow.keras.callbacks import History
import logging

logging.basicConfig(level=logging.INFO)

def load_data(X_path: str = 'X.npy', y_path: str = 'y.npy', test_size: float = 0.2, random_state: int = 42) -> Dict[str, np.ndarray]:
    """
    Loads preprocessed data from .npy files and splits it into training and testing sets.
    """
    from sklearn.model_selection import train_test_split
    
    # Load data
    X = np.load(X_path)
    y = np.load(y_path)
    logging.info(f"Original X shape: {X.shape}")
    logging.info(f"Original y shape: {y.shape}")
    
    # Normalize pixel values
    X = X / 255.0
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    return {
        'X_train': X_train, 
        'X_test': X_test, 
        'y_train': y_train, 
        'y_test': y_test
    }

def create_model() -> Sequential:
    """
    Defines and returns the CNN model architecture with fixed input shape.
    """
    model = Sequential([
        # First Convolutional Block
        Conv2D(32, (3, 3), padding='same', input_shape=(48, 48, 1)),
        BatchNormalization(),
        Conv2D(32, (3, 3), padding='same', activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),

        # Second Convolutional Block
        Conv2D(64, (3, 3), padding='same'),
        BatchNormalization(),
        Conv2D(64, (3, 3), padding='same', activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),

        # Third Convolutional Block
        Conv2D(128, (3, 3), padding='same'),
        BatchNormalization(),
        Conv2D(128, (3, 3), padding='same', activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),

        # Flatten and Dense Layers
        Flatten(),
        Dense(512, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(4, activation='softmax')
    ])
    
    return model

def compile_model(model: Sequential) -> Sequential:
    """
    Compiles the model with specified optimizer, loss, and metrics.
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
    """
    model.save(filename)
    logging.info(f"Model saved as {filename}")
    return filename

if __name__ == "__main__":
    try:
        # Load and preprocess data
        data_dict = load_data('/tmp/X.npy', '/tmp/y.npy')
        
        # Create and compile model with fixed input shape
        model = create_model()
        model = compile_model(model)
        
        # Train model
        training_results = train_model(
            model=model,
            X_train=data_dict['X_train'],
            y_train=data_dict['y_train']
        )
        
        # Save model
        save_model(training_results['model'])
        
    except Exception as e:
        logging.error(f"Training failed: {str(e)}")
        raise