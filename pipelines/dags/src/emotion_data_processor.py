import pandas as pd
import numpy as np
import logging
import os
from typing import List, Tuple, Optional, Generator, Dict
from pathlib import Path
import tempfile
from .emotion_gcs_handler import GCSHandler

class DataProcessor:
    """Handles data processing operations with GCS integration"""
    
    def __init__(self, 
                 gcs_handler: GCSHandler,
                 selected_emotions: List[int] = [0, 3, 4, 6],
                 emotion_map: Dict[int, int] = {0: 0, 3: 1, 4: 2, 6: 3},
                 image_width: int = 48,
                 image_height: int = 48,
                 chunk_size: int = 1000):
        self.gcs_handler = gcs_handler
        self.selected_emotions = selected_emotions
        self.emotion_map = emotion_map
        self.image_width = image_width
        self.image_height = image_height
        self.chunk_size = chunk_size
        
        self.temp_dir = Path(tempfile.mkdtemp())
        self._setup_logger()
    
    def _setup_logger(self):
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)
        
        if not self.logger.handlers:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.DEBUG)
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)
    
    def load_data_chunks(self, file_path: Path) -> Generator[pd.DataFrame, None, None]:
        """Loads the emotion dataset in chunks"""
        self.logger.info("Loading data in chunks")
        for chunk in pd.read_csv(file_path, chunksize=self.chunk_size):
            yield chunk
    
    def process_chunk(self, chunk: pd.DataFrame) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Process a single chunk of data"""
        self.logger.info("Processing a chunk of data")
        try:
            chunk = self.filter_emotions(chunk)
            if len(chunk) == 0:
                self.logger.warning("Chunk filtered out due to lack of selected emotions")
                return None, None
            
            chunk = self.map_emotions(chunk)
            X = self.preprocess_pixels(chunk)
            y = self.preprocess_labels(chunk)
            
            self.logger.info("Chunk processed successfully")
            return X, y
            
        except Exception as e:
            self.logger.error(f"Error processing chunk: {e}")
            return None, None
    
    def filter_emotions(self, data: pd.DataFrame) -> pd.DataFrame:
        """Filters data to include only selected emotions"""
        return data[data['emotion'].isin(self.selected_emotions)]
    
    def map_emotions(self, data: pd.DataFrame) -> pd.DataFrame:
        """Maps emotions to new labels"""
        data = data.copy()
        data['emotion'] = data['emotion'].map(self.emotion_map)
        return data
    
    def preprocess_pixels(self, data: pd.DataFrame) -> np.ndarray:
        """Processes pixel data into normalized arrays"""
        pixels = data['pixels'].tolist()
        X = []
        for pixel_sequence in pixels:
            pixel_array = np.array(pixel_sequence.split(' '), dtype=float)
            pixel_array = pixel_array.reshape(self.image_width, self.image_height, 1)
            X.append(pixel_array)
        X = np.array(X)
        return X / 255.0
    
    def preprocess_labels(self, data: pd.DataFrame) -> np.ndarray:
        """Extracts emotion labels"""
        return np.array(data['emotion'].tolist())
    
    def save_chunk(self, X: np.ndarray, y: np.ndarray, chunk_idx: int) -> Tuple[str, str]:
        """Saves a processed chunk to GCS"""
        self.logger.info(f"Saving chunk {chunk_idx} to GCS")
        
        X_local = self.temp_dir / f"X_chunk_{chunk_idx}.npy"
        y_local = self.temp_dir / f"y_chunk_{chunk_idx}.npy"
        
        np.save(X_local, X)
        np.save(y_local, y)
        
        X_gcs = f"data/processed/chunks/X_chunk_{chunk_idx}.npy"
        y_gcs = f"data/processed/chunks/y_chunk_{chunk_idx}.npy"
        
        self.gcs_handler.upload_file(X_local, X_gcs)
        self.gcs_handler.upload_file(y_local, y_gcs)
        
        os.remove(X_local)
        os.remove(y_local)
        
        return X_gcs, y_gcs
    
    def process_all_chunks(self, gcs_path: str) -> List[Tuple[str, str]]:
        """Process all chunks from GCS data and return paths to saved chunks"""
        self.logger.info("Starting processing of all chunks")
        chunk_paths = []
        
        temp_file = self.gcs_handler.download_to_temp(gcs_path)
        
        try:
            for chunk_idx, chunk in enumerate(self.load_data_chunks(temp_file)):
                self.logger.info(f"Processing chunk {chunk_idx + 1}")
                X, y = self.process_chunk(chunk)
                
                if X is not None and y is not None:
                    chunk_paths.append(self.save_chunk(X, y, chunk_idx))
            
            self.logger.info("All chunks processed and saved")
            return chunk_paths
        finally:
            os.remove(temp_file)