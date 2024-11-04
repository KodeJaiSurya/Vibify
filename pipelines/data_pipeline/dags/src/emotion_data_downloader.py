from pathlib import Path
import gdown

class DataDownloader:
    """Handles downloading of datasets from Google Drive"""
    
    def __init__(self, base_path: str = "dags/data/raw"):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
    
    def download_from_gdrive(self, file_id: str) -> str:
        """Downloads dataset from Google Drive"""
        try:
            url = f"https://drive.google.com/uc?id={file_id}"
            file_path = str(self.base_path / "dataset_fer2013.csv")
            gdown.download(url, file_path, quiet=False)
            print("FER2013 dataset downloaded.")
            return file_path
        except Exception as e:
            print(f"Error downloading file: {e}")
            raise