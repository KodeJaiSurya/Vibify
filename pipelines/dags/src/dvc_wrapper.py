from dvc.repo import Repo
from pathlib import Path
import logging

class DVCWrapper:
    """DVC wrapper to track data files"""
    
    def __init__(self, gcs_bucket: str):
        self.logger = logging.getLogger(__name__)
        self.gcs_bucket = gcs_bucket
        self.repo = self._init_dvc()
    
    def _init_dvc(self):
        """Initialize DVC repository"""
        try:
            repo = Repo.init()
            repo.add_remote(
                name='storage',
                url=self.gcs_bucket,
                default=True
            )
            self.logger.info("DVC initialized successfully")
            return repo
        except Exception as e:
            self.logger.error(f"DVC initialization error: {e}")
            raise
    
    def track_file(self, file_path: str):
        """Track a single file with DVC"""
        try:
            self.repo.add(file_path)
            self.repo.commit()
            self.repo.push()
            self.logger.info(f"Successfully tracked {file_path} in DVC")
        except Exception as e:
            self.logger.error(f"Error tracking file in DVC: {e}")