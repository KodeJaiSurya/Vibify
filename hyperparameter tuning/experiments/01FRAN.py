# emotion_model_executor.py
import numpy as np
from emotion_model_pipeline import load_data, create_model, compile_model, train_model

def execute_model_pipeline():
    """
    Executes the load_data function to load and split the dataset, then creates, compiles, and trains the model.
    Each step is executed in separate try blocks.
    """
    try:
        # Specify the file paths for X and y (these can be adjusted as needed)
        X_path = 'dags/data/preprocessed/X.npy'  # dags/data/preprocessed/X.npy
        y_path = 'dags/data/preprocessed/y.npy'  # dags/data/preprocessed/y.npy

        # Call load_data function
        data = load_data(X_path=X_path, y_path=y_path)

        # Print the shapes of the loaded data
        print(f"Loaded data shapes: X_train: {data['X_train'].shape}, X_test: {data['X_test'].shape}")
        print(f"y_train: {data['y_train'].shape}, y_test: {data['y_test'].shape}")
        
    except Exception as e:
        print(f"Error executing load_data: {e}")

    try:
        # Create the model using the training data shape
        input_shape = data['X_train'].shape[1:]  # Get the input shape from the training data
        model = create_model(input_shape)

        # Print a summary of the model
        model.summary()

    except Exception as e:
        print(f"Error executing create_model: {e}")

    try:
        # Compile the model
        model = compile_model(model)

        # Print model compile summary
        print("Model compiled successfully.")
        
    except Exception as e:
        print(f"Error executing compile_model: {e}")

    try:
        # Train the model
        result = train_model(model, data['X_train'], data['y_train'])

        # Print training history
        print(f"Training completed. History: {result['history']}")

    except Exception as e:
        print(f"Error executing train_model: {e}")

# Run the function to execute the data loading, model creation, compilation, and training
if __name__ == "__main__":
    execute_model_pipeline()
