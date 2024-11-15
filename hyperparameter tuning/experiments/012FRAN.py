import mlflow
import mlflow.keras
from emotion_model_pipeline import load_data, create_model, compile_model, train_model, save_model

def execute_model_pipeline():
    """
    Executes the load_data function to load and split the dataset, then creates, compiles, trains the model,
    and saves it to a file. Each step is executed in separate try blocks, and MLFlow tracks the experiments.
    """
    try:
        # Start the MLFlow experiment
        mlflow.start_run()

        # Specify the file paths for X and y (these can be adjusted as needed)
        X_path = 'dags/data/preprocessed/X.npy'  # dags/data/preprocessed/X.npy
        y_path = 'dags/data/preprocessed/y.npy'  # dags/data/preprocessed/y.npy

        # Call load_data function
        data = load_data(X_path=X_path, y_path=y_path)

        # Print the shapes of the loaded data
        print(f"Loaded data shapes: X_train: {data['X_train'].shape}, X_test: {data['X_test'].shape}")
        print(f"y_train: {data['y_train'].shape}, y_test: {data['y_test'].shape}")
        
        # Log the dataset information in MLFlow
        mlflow.log_param("X_train_shape", data['X_train'].shape)
        mlflow.log_param("y_train_shape", data['y_train'].shape)

    except Exception as e:
        print(f"Error executing load_data: {e}")

    try:
        # Create the model using the training data shape
        input_shape = data['X_train'].shape[1:]  # Get the input shape from the training data
        model = create_model(input_shape)

        # Print a summary of the model
        model.summary()

        # Log the model architecture in MLFlow
        mlflow.log_param("model_architecture", "Sequential with Dense layers")

    except Exception as e:
        print(f"Error executing create_model: {e}")

    try:
        # Compile the model
        model = compile_model(model)

        # Print model compile summary
        print("Model compiled successfully.")
        
        # Log model compile info
        mlflow.log_param("optimizer", "Adam")  # Example, adjust as needed
        mlflow.log_param("loss_function", "categorical_crossentropy")

    except Exception as e:
        print(f"Error executing compile_model: {e}")

    try:
        # Train the model
        result = train_model(model, data['X_train'], data['y_train'])

        # Print training history
        print(f"Training completed. History: {result['history']}")

        # Log training metrics in MLFlow
        mlflow.log_metric("train_loss", result['history']['loss'][-1])
        mlflow.log_metric("train_accuracy", result['history']['accuracy'][-1])

    except Exception as e:
        print(f"Error executing train_model: {e}")

    try:
        # Save the trained model after training
        model_filename = 'emotion_detection_model_4_classes.h5'
        save_model(model, filename=model_filename)  # Save the trained model
        print(f"Model saved successfully as {model_filename}")

        # Log the model in MLFlow
        mlflow.keras.log_model(model, "model")

    except Exception as e:
        print(f"Error executing save_model: {e}")

    finally:
        # End the MLFlow experiment
        mlflow.end_run()

# Run the function to execute the data loading, model creation, compilation, training, and saving
if __name__ == "__main__":
    execute_model_pipeline()
