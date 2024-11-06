import pandas as pd
import gdown
from sklearn.preprocessing import StandardScaler
import os
import logging

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

def load_song_data(file_id: str) -> pd.DataFrame:
    """Downloads dataset from Google Drive"""
    url = f"https://drive.google.com/uc?id={file_id}"
    file_path = "dags/data/raw/song_dataset.csv"
    
    logger.info("Starting download of song dataset")
    try:
        gdown.download(url, file_path, quiet=False)
        df = pd.read_csv(file_path)
        logger.info("Data loaded successfully.")
        return df
    except FileNotFoundError:
        logger.error("File not found.")
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

def save_features(df: pd.DataFrame, output_dir: str = "dags/data/preprocessed") -> None:
    """Saves the preprocessed dataframe"""
    logger.info("Saving preprocessed song data to local directory")
    
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "preprocessed_features.csv")
    df.to_csv(output_path, index=False)
    
    del df
    logger.info(f"Preprocessed song data saved to {output_path}")

if __name__ == "__main__":
    file_id  = "1zckGHmd_tJfyMqePfol0L-lIScstOCh9"
    load_song_data(file_id)