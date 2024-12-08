import numpy as np
import logging
import os
from typing import List, Tuple
from pathlib import Path
import tempfile
from .emotion_gcs_handler import GCSHandler
# from .dvc_wrapper import DVCWrapper

class DataAggregator:
    """Handles combining and saving final processed data with GCS integration"""
    
    def __init__(self, gcs_handler: GCSHandler):
        self.gcs_handler = gcs_handler
        self.temp_dir = Path(tempfile.mkdtemp())
        self._setup_logger()
        # self.dvc = DVCWrapper(gcs_handler.bucket_name)
    
    def _setup_logger(self):
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)
        
        if not self.logger.handlers:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.DEBUG)
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)
    
    def combine_chunks(self, chunk_paths: List[Tuple[str, str]]) -> Tuple[np.ndarray, np.ndarray]:
        """Combines all chunks from GCS into final arrays"""
        self.logger.info("Starting to combine chunks from GCS")
        try:
            X_list = []
            y_list = []
            
            for x_gcs, y_gcs in chunk_paths:
                x_temp = self.gcs_handler.download_to_temp(x_gcs)
                y_temp = self.gcs_handler.download_to_temp(y_gcs)
                
                X_list.append(np.load(x_temp))
                y_list.append(np.load(y_temp))
                
                os.remove(x_temp)
                os.remove(y_temp)
            
            self.logger.info("Chunks combined successfully")
            return np.concatenate(X_list), np.concatenate(y_list)
        except Exception as e:
            self.logger.error(f"Error during chunk combination: {e}")
            raise
    
    def cleanup_chunks(self, chunk_paths: List[Tuple[str, str]]):
        """Removes chunk files from GCS"""
        self.logger.info("Starting cleanup of chunk files from GCS")
        for x_gcs, y_gcs in chunk_paths:
            try:
                self.gcs_handler.bucket.blob(x_gcs).delete()
                self.gcs_handler.bucket.blob(y_gcs).delete()
                self.logger.debug(f"Cleaned GCS chunk files: {x_gcs}, {y_gcs}")
            except Exception as e:
                self.logger.error(f"Error cleaning GCS chunk files {x_gcs} and {y_gcs}: {e}")
    
    def save_final(self, X: np.ndarray, y: np.ndarray) -> bool:
        """Saves final processed arrays to GCS"""
        self.logger.info("Saving final data arrays to GCS")
        try:
            X_local = self.temp_dir / "X.npy"
            y_local = self.temp_dir / "y.npy"
            
            np.save(X_local, X)
            np.save(y_local, y)
            
            X_gcs = "data/preprocessed/facial_expression/X.npy"
            y_gcs = "data/preprocessed/facial_expression/y.npy"
            
            success = (self.gcs_handler.upload_file(X_local, X_gcs) and 
                      self.gcs_handler.upload_file(y_local, y_gcs))
            
            # if success:
            #    self.dvc.track_file(X_gcs)
            #    self.dvc.track_file(y_gcs)
            
            os.remove(X_local)
            os.remove(y_local)
            
            if success:
                self.logger.info("Final data saved to GCS successfully!")
            return success
        except Exception as e:
            self.logger.error(f"Error saving final data to GCS: {e}")
            return False