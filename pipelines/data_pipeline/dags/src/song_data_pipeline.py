import pandas as pd
import gdown
from sklearn.preprocessing import StandardScaler
import os

def load_song_data(file_id: str) -> pd.DataFrame:
    """Downloads dataset from Google Drive"""
    url = f"https://drive.google.com/uc?id={file_id}"
    file_path = "dags/data/raw/song_dataset.csv"
    gdown.download(url, file_path, quiet=False)
    df = pd.read_csv(file_path)
    print("Data loaded successfully.")
    return df

def data_cleaning(df: pd.DataFrame) -> pd.DataFrame:
  """Cleans the data by dropping duplicates and filtering out required coloumns"""
  df = df.dropna(subset=['song_name', 'uri'])
  df = df.drop_duplicates(subset=['song_name', 'uri'])
  cols = ['danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness',
       'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo','uri','genre','song_name']
  filtered_df = df[cols]
  del df
  return filtered_df

def scale_features(df: pd.DataFrame) -> pd.DataFrame:
    """Scales specified features in the dataframe."""
    features = [i for i in df.columns if df[i].dtype != 'object']
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df[features])
    return pd.DataFrame(scaled_data, columns=features)

def save_features(df: pd.DataFrame, output_dir: str = "dags/data/preprocessed") -> None:
   """Saves the preprocessed dataframe"""
   os.makedirs(output_dir, exist_ok=True)
   output_path = os.path.join(output_dir, "preprocessed_features.csv")
   df.to_csv(output_path, index=False)
   del df
   print("Preprocessed song data saved to local")