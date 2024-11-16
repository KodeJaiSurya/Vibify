from pathlib import Path
import opendatasets as od
import logging

class DataDownloader:
    """Handles downloading of datasets from Google Drive"""
    
    def __init__(self, base_path: str = "dags/data/raw"):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        
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
    
    def download_from_kaggle(self) -> str:
        """Downloads dataset from Kaggle"""
        self.logger.info("Starting download from Kaggle")
        
        try:
            url = f"https://www.kaggle.com/datasets/nicolejyt/facialexpressionrecognition"
            file_path = str(self.base_path)
            od.download_kaggle_dataset(url, file_path)
            self.logger.info("FER2013 dataset downloaded successfully.")
            return file_path
        except Exception as e:
            self.logger.error(f"Error downloading file: {e}")
            raise

if __name__ == "__main__":
    downloader = DataDownloader()
    downloader.download_from_kaggle()