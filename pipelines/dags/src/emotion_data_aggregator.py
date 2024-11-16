from pathlib import Path
import numpy as np
import os
from typing import List, Tuple
import logging

class DataAggregator:
    """Handles combining and saving final processed data"""
    
    def __init__(self, temp_dir: str = "temp_chunks"):
        self.temp_dir = Path(temp_dir)
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)
        
        # Set up console handler for logging
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.DEBUG)
        
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(formatter)
        
        # Add handler to logger
        if not self.logger.handlers:
            self.logger.addHandler(console_handler)
    
    def combine_chunks(self, chunk_paths: List[Tuple[Path, Path]]) -> Tuple[np.ndarray, np.ndarray]:
        """Combines all chunks into final arrays"""
        self.logger.info("Starting to combine chunks")
        try:
            X_list = [np.load(x_path) for x_path, _ in chunk_paths]
            y_list = [np.load(y_path) for _, y_path in chunk_paths]
            self.logger.info("Chunks combined successfully")
            return np.concatenate(X_list), np.concatenate(y_list)
        except Exception as e:
            self.logger.error(f"Error during chunk combination: {e}")
            raise
    
    def cleanup_chunks(self, chunk_paths: List[Tuple[Path, Path]]):
        """Removes temporary chunk files"""
        self.logger.info("Starting cleanup of chunk files")
        for x_path, y_path in chunk_paths:
            try:
                os.remove(x_path)
                os.remove(y_path)
                self.logger.debug(f"Cleaned previous chunk files: {x_path}, {y_path}")
            except Exception as e:
                self.logger.error(f"Error cleaning chunk files {x_path} and {y_path}: {e}")
    
    def save_final(self, X: np.ndarray, y: np.ndarray) -> bool:
        """Saves final processed arrays"""
        os.makedirs("dags/data/preprocessed", exist_ok=True)
        self.logger.info("Saving final data arrays")
        try:
            np.save("dags/data/preprocessed/X.npy", X)
            np.save("dags/data/preprocessed/y.npy", y)
            self.logger.info("X and Y saved successfully!")
            return True
        except Exception as e:
            self.logger.error(f"Error saving final data: {e}")
            return False