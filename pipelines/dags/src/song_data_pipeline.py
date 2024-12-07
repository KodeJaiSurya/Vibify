import pandas as pd
from sklearn.preprocessing import StandardScaler
import logging
from google.cloud import storage
import io
from pathlib import Path
from .dvc_wrapper import DVCWrapper  

# Set up logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
# Set up console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
# Define logging format
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
# Add handler to logger
if not logger.handlers:
    logger.addHandler(console_handler)

def load_song_data(bucket_name: str, blob_name: str) -> pd.DataFrame:
    """
    Loads dataset from Google Cloud Storage bucket
    
    Args:
        bucket_name (str): Name of the GCS bucket
        blob_name (str): Path to the file within the bucket
        
    Returns:
        pd.DataFrame: Loaded dataset or None if error occurs
    """
    logger.info(f"Starting download of song dataset from GCS bucket: {bucket_name}")
    try:
        # Initialize GCS client
        storage_client = storage.Client()
        bucket = storage_client.get_bucket(bucket_name)
        blob = bucket.blob(blob_name)

        # Download as string
        content = blob.download_as_string()
        
        # Load into pandas
        df = pd.read_csv(io.StringIO(content.decode('utf-8')))
        logger.info("Data loaded successfully from GCS.")

        if df is not None:
            raw_path = f"data/raw/{Path(blob_name).name}"
            df.to_csv(raw_path, index=False)
            # dvc = DVCWrapper(bucket_name)
            # dvc.track_file(raw_path)

        return df
    except Exception as e:
        logger.error(f"Error loading data from GCS: {str(e)}")
        return None

def data_cleaning(df: pd.DataFrame) -> pd.DataFrame:
    """Cleans the data by dropping duplicates and filtering out required columns"""
    logger.info("Starting data cleaning")
    df = df.dropna(subset=['song_name', 'uri'])
    df = df.drop_duplicates(subset=['song_name', 'uri'])
    
    cols = [
        'danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness',
        'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo',
        'uri', 'genre', 'song_name'
    ]
    
    filtered_df = df[cols]
    del df
    logger.info("Data cleaning completed")
    return filtered_df

def scale_features(df: pd.DataFrame) -> pd.DataFrame:
    """Scales specified features in the dataframe."""
    logger.info("Starting feature scaling")
    features = [i for i in df.columns if df[i].dtype != 'object']
    
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df[features])
    
    scaled_df = pd.DataFrame(scaled_data, columns=features)
    logger.info("Feature scaling completed")
    return scaled_df

def scale_features(df: pd.DataFrame) -> pd.DataFrame:
    """Scales numerical features in the dataframe and retains non-numerical features."""
    logger.info("Starting feature scaling")
    # Separate numerical and non-numerical features
    numerical_features = df.select_dtypes(include=['float64', 'int64']).columns
    non_numerical_features = df.select_dtypes(exclude=['float64', 'int64']).columns
    # Scale numerical features
    scaler = StandardScaler()
    scaled_numerical_data = scaler.fit_transform(df[numerical_features])
    scaled_numerical_df = pd.DataFrame(scaled_numerical_data, columns=numerical_features, index=df.index)
    # Concatenate scaled numerical features with non-numerical features
    final_df = pd.concat([scaled_numerical_df, df[non_numerical_features]], axis=1)
    logger.info("Feature scaling completed")
    return final_df

def save_features(df: pd.DataFrame, bucket_name: str, blob_name: str) -> None:
    """
    Saves the preprocessed dataframe to GCS
    
    Args:
        df (pd.DataFrame): DataFrame to save
        bucket_name (str): Name of the GCS bucket
        blob_name (str): Path where the file should be saved in the bucket
    """
    logger.info(f"Saving preprocessed song data to GCS bucket: {bucket_name}")
    try:
        # Convert DataFrame to CSV string
        csv_buffer = df.to_csv(index=False)
        
        # Initialize GCS client
        storage_client = storage.Client()
        bucket = storage_client.get_bucket(bucket_name)
        blob = bucket.blob(blob_name)
        
        # Upload to GCS
        blob.upload_from_string(csv_buffer, content_type='text/csv')

        # Add DVC tracking
        # processed_path = f"data/preprocessed/{Path(blob_name).name}"
        # df.to_csv(processed_path, index=False)
        # dvc = DVCWrapper(bucket_name)
        # dvc.track_file(processed_path)
        
        del df
        logger.info(f"Preprocessed song data saved to gs://{bucket_name}/{blob_name}")
    except Exception as e:
        logger.error(f"Error saving data to GCS: {str(e)}")