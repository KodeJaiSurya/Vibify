import keras
from keras import layers
import keras_tuner as kt
import mlflow
import mlflow.keras
from emotion_model_pipeline import load_data
import tensorflow as tf

# Define your experiment name
experiment_name = "Vibify_emotions_2"

# Set the experiment
mlflow.set_experiment(experiment_name)

# Verify the experiment was created (you can print the current experiment to check)
experiment = mlflow.get_experiment_by_name(experiment_name)
print(experiment)

# Load the data
data = load_data(
    X_path='C:/Users/Surface/NU/Vibify/dags/data/preprocessed/X.npy',  # Path to the features file
    y_path='C:/Users/Surface/NU/Vibify/dags/data/preprocessed/y.npy',  # Path to the labels file
)

# Print the data shapes
print(f"Training data shape: {data['X_train'].shape}")
print(f"Testing data shape: {data['X_test'].shape}")

# Define the model-building function with hyperparameter tuning
def build_model(hp):
    model = keras.Sequential()
    
    # Choose number of convolutional layers
    model.add(layers.InputLayer(input_shape=(48, 48, 1)))
    
    for i in range(hp.Int('conv_layers', 1, 3)):  # Randomly choose number of conv layers (1 to 3)
        model.add(layers.Conv2D(
            filters=hp.Int(f'filters_{i}', min_value=32, max_value=128, step=32),  # Random filter size
            kernel_size=(3, 3),
            activation='relu'
        ))
        model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    
    model.add(layers.Flatten())
    
    # Add a fully connected layer with number of units
    model.add(layers.Dense(
        hp.Int('units', min_value=64, max_value=512, step=64),  # Random number of units
        activation='relu'
    ))

    # Output layer with 4 classes (emotion classification)
    model.add(layers.Dense(4, activation='softmax'))
    
    # Compile the model
    model.compile(
        optimizer=keras.optimizers.Adam(
            hp.Float('learning_rate', min_value=1e-5, max_value=1e-2, sampling='LOG')
        ),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

# Function to log results to MLFlow
def log_trial_to_mlflow(trial):
    # Ensure no active MLflow run exists
    if mlflow.active_run():
        mlflow.end_run()

    # Start a new MLflow run
    with mlflow.start_run():
        # Log hyperparameters
        for hp_name, value in trial.hyperparameters.values.items():
            mlflow.log_param(hp_name, value)

        # Log the validation accuracy
        val_accuracy = trial.metrics.get_best_value('val_accuracy')  # Directly get the best value for val_accuracy
        if val_accuracy is not None:
            mlflow.log_metric('val_accuracy', val_accuracy)


# Initialize the RandomSearch tuner with a unique directory
tuner = kt.RandomSearch(
    build_model,
    objective='val_accuracy',  # Objective to optimize
    max_trials=2,  # Increase number of trials for better tuning
    executions_per_trial=2,  # Increase executions per trial for averaging
    directory=f'tuner_dir_{experiment_name}',  # Unique directory
    project_name='emotion_model_random_search'  # Project name for the tuning
)

# Run the search for the hyperparameters
tuner.search(
    data['X_train'], data['y_train'], 
    epochs=5,  # Number of epochs per trial
    validation_data=(data['X_test'], data['y_test'])
)

# Log the best trials to MLFlow
for trial in tuner.oracle.get_best_trials():
    log_trial_to_mlflow(trial)

# Optionally, save the best model to MLFlow
best_model = tuner.get_best_models()[0]
mlflow.keras.log_model(best_model, 'best_model')
