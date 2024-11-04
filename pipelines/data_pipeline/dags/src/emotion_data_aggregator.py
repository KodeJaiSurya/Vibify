from pathlib import Path
import numpy as np
import os
from typing import List, Tuple

class DataAggregator:
    """Handles combining and saving final processed data"""
    
    def __init__(self, temp_dir: str = "temp_chunks"):
        self.temp_dir = Path(temp_dir)
    
    def combine_chunks(self, chunk_paths: List[Tuple[Path, Path]]) -> Tuple[np.ndarray, np.ndarray]:
        """Combines all chunks into final arrays"""
        X_list = [np.load(x_path) for x_path, _ in chunk_paths]
        y_list = [np.load(y_path) for _, y_path in chunk_paths]
        
        return np.concatenate(X_list), np.concatenate(y_list)
    
    def cleanup_chunks(self, chunk_paths: List[Tuple[Path, Path]]):
        """Removes temporary chunk files"""
        for x_path, y_path in chunk_paths:
            os.remove(x_path)
            os.remove(y_path)
    
    def save_final(self, X: np.ndarray, y: np.ndarray) -> bool:
        """Saves final processed arrays"""
        os.makedirs("dags/data/preprocessed",exist_ok=True)
        try:
            np.save("dags/data/preprocessed/X.npy", X)
            np.save("dags/data/preprocessed/y.npy", y)
            print("X and Y saved!")
            return True
        except Exception as e:
            print(f"Error saving final data: {e}")
            return False