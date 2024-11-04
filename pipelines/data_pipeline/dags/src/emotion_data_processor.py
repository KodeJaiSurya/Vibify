from pathlib import Path
from typing import Optional, Dict, List, Tuple, Generator
import pandas as pd
import numpy as np

class DataProcessor:
    """Handles data processing operations"""
    
    def __init__(self, 
                 selected_emotions: List[int] = [0,3,4,6],
                 emotion_map: Dict[int, int] = {0:0,3:1,4:2,6:3},
                 image_width: int = 48,
                 image_height: int = 48,
                 chunk_size: int = 1000):
        self.selected_emotions = selected_emotions
        self.emotion_map = emotion_map
        self.image_width = image_width
        self.image_height = image_height
        self.chunk_size = chunk_size
        self.temp_dir = Path("temp_chunks")
        self.temp_dir.mkdir(exist_ok=True)
    
    def load_data_chunks(self, file_path: str) -> Generator[pd.DataFrame, None, None]:
        """Loads the emotion dataset in chunks"""
        for chunk in pd.read_csv(file_path, chunksize=self.chunk_size):
            yield chunk
    
    def process_chunk(self, chunk: pd.DataFrame) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Process a single chunk of data"""
        try:
            # Filter and map emotions
            chunk = self.filter_emotions(chunk)
            if len(chunk) == 0:
                return None, None
                
            chunk = self.map_emotions(chunk)
            
            # Process features and labels
            X = self.preprocess_pixels(chunk)
            y = self.preprocess_labels(chunk)
            
            return X, y
            
        except Exception as e:
            print(f"Error processing chunk: {e}")
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
        return X/255.0
    
    def preprocess_labels(self, data: pd.DataFrame) -> np.ndarray:
        """Extracts emotion labels"""
        return np.array(data['emotion'].tolist())
    
    def save_chunk(self, X: np.ndarray, y: np.ndarray, chunk_idx: int) -> Tuple[Path, Path]:
        """Saves a processed chunk to disk"""
        X_path = self.temp_dir / f"X_chunk_{chunk_idx}.npy"
        y_path = self.temp_dir / f"y_chunk_{chunk_idx}.npy"
        
        np.save(X_path, X)
        np.save(y_path, y)
        
        return X_path, y_path
    
    def process_all_chunks(self, file_path: str) -> List[Tuple[Path, Path]]:
        """Process all chunks and return paths to saved chunks"""
        chunk_paths = []
        
        for chunk_idx, chunk in enumerate(self.load_data_chunks(file_path)):
            print(f"Processing chunk {chunk_idx + 1}")
            X, y = self.process_chunk(chunk)
            
            if X is not None and y is not None:
                chunk_paths.append(self.save_chunk(X, y, chunk_idx))
        
        return chunk_paths