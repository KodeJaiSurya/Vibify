import os
import logging
import numpy as np
import pandas as pd
from google.cloud import storage
import tensorflow as tf
from emotion_model_pipeline import (
    load_data,
    create_model,
    compile_model,
    train_model,
    save_model
)
from song_model_pipeline import (
    apply_pca,
    apply_kmeans,
    assign_mood
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ModelTrainer:
    def __init__(self):
        self.bucket_name = 'vibebucketoncloudv1'
        self.emotion_data_path = "data/preprocessed/facial_expression"
        self.song_data_path = "data/preprocessed/spotify"
        self.local_output_dir = "/tmp/models"
        self.gcs_output_dir = "models"
        
        # Ensure local output directory exists
        os.makedirs(self.local_output_dir, exist_ok=True)
        os.makedirs(f"{self.local_output_dir}/emotion_model", exist_ok=True)
        os.makedirs(f"{self.local_output_dir}/song_model", exist_ok=True)

    def download_from_gcs(self, source_blob_name: str, destination_file_name: str) -> None:
        """Downloads a file from Google Cloud Storage."""
        storage_client = storage.Client()
        bucket = storage_client.bucket(self.bucket_name)
        blob = bucket.blob(source_blob_name)
        blob.download_to_filename(destination_file_name)
        logging.info(f"Downloaded {source_blob_name} to {destination_file_name}")

    def upload_to_gcs(self, source_file_name: str, destination_blob_name: str) -> None:
        """Uploads a file to Google Cloud Storage."""
        storage_client = storage.Client()
        bucket = storage_client.bucket(self.bucket_name)
        blob = bucket.blob(destination_blob_name)
        blob.upload_from_filename(source_file_name)
        logging.info(f"Uploaded {source_file_name} to {destination_blob_name}")

    def upload_directory_to_gcs(self, local_directory: str, gcs_directory: str) -> None:
        """Uploads an entire directory to Google Cloud Storage."""
        for root, dirs, files in os.walk(local_directory):
            for file in files:
                local_path = os.path.join(root, file)
                # Create relative path for GCS
                relative_path = os.path.relpath(local_path, local_directory)
                gcs_path = os.path.join(gcs_directory, relative_path)
                self.upload_to_gcs(local_path, gcs_path)

    def train_emotion_model(self) -> None:
        """Trains the emotion detection model."""
        logging.info("Starting emotion model training")
        
        # Download emotion data
        self.download_from_gcs(f"{self.emotion_data_path}/X.npy", "/tmp/X.npy")
        self.download_from_gcs(f"{self.emotion_data_path}/y.npy", "/tmp/y.npy")
        
        # Load and split data
        data_dict = load_data(X_path="/tmp/X.npy", y_path="/tmp/y.npy")
        
        # Create and compile model
        model = create_model()
        model = compile_model(model)
        
        # Train model
        training_results = train_model(
            model=model,
            X_train=data_dict['X_train'],
            y_train=data_dict['y_train'],
            epochs=30,
            batch_size=64
        )
        
        # Save model locally
        local_model_path = f"{self.local_output_dir}/emotion_model"
        training_results['model'].save(local_model_path)
        logging.info(f"Emotion model saved locally to {local_model_path}")
        
        # Upload to GCS
        gcs_model_path = f"{self.gcs_output_dir}/emotion_model"
        self.upload_directory_to_gcs(local_model_path, gcs_model_path)
        logging.info(f"Emotion model uploaded to GCS: {gcs_model_path}")

    def train_song_model(self) -> None:
        """Process songs and save cluster assignments."""
        logging.info("Starting song clustering process")
        
        # Download song data
        self.download_from_gcs(f"{self.song_data_path}/genres_v2.csv", "/tmp/songs.csv")
        
        # Load data
        df = pd.read_csv("/tmp/songs.csv")
        
        # Apply PCA
        principal_components = apply_pca(df, n_components=10)
        
        # Apply KMeans clustering
        df = apply_kmeans(df, principal_components, n_clusters=4)
        
        # Assign moods
        df = assign_mood(df)
        
        # Save clustered data locally
        local_file_path = os.path.join(self.local_output_dir, "clusters.csv")
        df.to_csv(local_file_path, index=False)
        logging.info(f"Clustered data saved locally to {local_file_path}")
        
        # Upload to GCS
        gcs_file_path = f"{self.gcs_output_dir}/clusters.csv"
        self.upload_to_gcs(local_file_path, gcs_file_path)
        logging.info(f"Clustered data uploaded to GCS: {gcs_file_path}")
        
        # Cleanup
        del df
        if os.path.exists(local_file_path):
            os.remove(local_file_path)
            logging.info(f"Cleaned up local file: {local_file_path}")

    def train(self) -> None:
        """Coordinates the training of both models."""
        try:
            self.train_emotion_model()
            self.train_song_model()
            logging.info("Training completed successfully")
        except Exception as e:
            logging.error(f"Training failed: {str(e)}")
            raise

def main():
    trainer = ModelTrainer()
    trainer.train()

if __name__ == "__main__":
    main()