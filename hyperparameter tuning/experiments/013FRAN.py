# Import the load_data function
from emotion_model_pipeline import load_data  

# Call the load_data function
data = load_data(
    X_path= 'dags/data/preprocessed/X.npy',  # Path to the features file
    y_path='dags/data/preprocessed/y.npy',  # Path to the labels file
    #test_size=0.2,   # Proportion for the test set
    #random_state=42  # Random seed for reproducibility
)

# Print the keys of the returned dictionary to confirm the output
print("Keys in the returned dictionary:", data.keys())
print(f"Training data shape: {data['X_train'].shape}")
print(f"Testing data shape: {data['X_test'].shape}")
