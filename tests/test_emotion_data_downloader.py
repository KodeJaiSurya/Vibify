import os
import sys
import unittest
from unittest.mock import patch
from pathlib import Path

# Adding the project root to `sys.path`
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

from pipelines.data_pipeline.dags.src.emotion_data_downloader import DataDownloader

class TestDataDownloader(unittest.TestCase):
    
    @patch('pipelines.data_pipeline.dags.src.emotion_data_downloader.gdown.download')
    def test_successful_download(self, mock_download):
        """Testing successful download of a dataset"""
        downloader = DataDownloader()
        file_id = "dummy_file_id"
        expected_path = str(downloader.base_path / "dataset_fer2013.csv")
        mock_download.return_value = expected_path
        result = downloader.download_from_gdrive(file_id)
        mock_download.assert_called_once_with(f"https://drive.google.com/uc?id={file_id}", expected_path, quiet=False)
        # Checking that the returned path is correct and the directory was created
        self.assertEqual(result, expected_path)
        self.assertTrue(Path(downloader.base_path).exists()) 
    
    @patch('pipelines.data_pipeline.dags.src.emotion_data_downloader.gdown.download')
    def test_download_error_handling(self, mock_download):
        """Test error handling during download"""
        downloader = DataDownloader()
        file_id = "dummy_file_id"
        mock_download.side_effect = Exception("Download failed")
        with self.assertRaises(Exception) as context:
            downloader.download_from_gdrive(file_id)
        self.assertEqual(str(context.exception), "Download failed")
    
    def test_directory_creation(self):
        """Test if the base directory is created by init"""
        base_path = "test_data/raw" 
        downloader = DataDownloader(base_path=base_path)
        self.assertTrue(Path(base_path).exists())
        # Cleanup
        for child in Path(base_path).iterdir():
            if child.is_file():
                child.unlink()
        Path(base_path).rmdir() 

if __name__ == "__main__":
    unittest.main()
