import numpy as np
from sklearn.metrics import accuracy_score
import logging

def evaluate_model(model, X_test):
    """
    Evaluates the model on the test data and returns predictions.

    Args:
        model (tf.keras.Model): Trained model.
        X_test (np.ndarray): Test features.

    Returns:
        np.ndarray: Predicted labels for the test set.
    """
    y_pred_probs = model.predict(X_test)
    y_pred = np.argmax(y_pred_probs, axis=1)
    return y_pred

def validate_accuracy(model, X_test, y_test, threshold=0.6):
    """
    Validates model accuracy against a threshold.

    Args:
        model (tf.keras.Model): Trained model.
        X_test (np.ndarray): Test features.
        y_test (np.ndarray): True labels for the test set.
        threshold (float): Accuracy threshold for validation.

    Returns:
        int: Returns 1 if accuracy > threshold, otherwise 0.
    """
    # Get predictions using evaluate_model function
    y_pred = evaluate_model(model, X_test)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    logging.info("Validation Accuracy: ", accuracy)
    
    # Check if accuracy meets the threshold
    return 1 if accuracy > threshold else 0