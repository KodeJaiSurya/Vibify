import unittest
from unittest.mock import patch, MagicMock
from pathlib import Path
import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

# Importing the classes and functions to test
from pipelines.dags.src.emotion_data_downloader import DataDownloader
from pipelines.dags.src.emotion_data_processor import DataProcessor
from pipelines.dags.src.emotion_data_aggregator import DataAggregator
from pipelines.dags.src.emotion_data_pipeline import (
    download_emotion_data,
    process_emotion_data,
    aggregate_filtered_data,
)

class TestEmotionDataPipeline(unittest.TestCase):

    @patch('pipelines.dags.src.emotion_data_pipeline.DataDownloader')
    def test_download_emotion_data(self, MockDataDownloader):
        """Test downloading emotion data"""
        mock_downloader = MockDataDownloader.return_value
        mock_downloader.download_from_gdrive.return_value = "mock_file_path"
        file_id = "test_file_id"
        result = download_emotion_data(file_id)
        self.assertEqual(result, "mock_file_path")
        mock_downloader.download_from_gdrive.assert_called_once_with(file_id)

    @patch('pipelines.dags.src.emotion_data_pipeline.DataProcessor')
    def test_process_emotion_data(self, MockDataProcessor):
        """Test processing emotion data"""
        mock_processor = MockDataProcessor.return_value
        mock_processor.process_all_chunks.return_value = [(Path("chunk_X.npy"), Path("chunk_y.npy"))]
        file_path = "mock_file_path"
        result = process_emotion_data(file_path)
        self.assertEqual(result, [(Path("chunk_X.npy"), Path("chunk_y.npy"))])
        mock_processor.process_all_chunks.assert_called_once_with(file_path)

    @patch('pipelines.dags.src.emotion_data_pipeline.DataAggregator')
    def test_aggregate_filtered_data(self, MockDataAggregator):
        """Test aggregating filtered data"""
        mock_aggregator = MockDataAggregator.return_value
        mock_aggregator.combine_chunks.return_value = (MagicMock(), MagicMock())
        mock_aggregator.save_final.return_value = True
        chunk_paths = [(Path("chunk_X.npy"), Path("chunk_y.npy"))]
        result = aggregate_filtered_data(chunk_paths)
        self.assertTrue(result)
        mock_aggregator.combine_chunks.assert_called_once_with(chunk_paths)
        mock_aggregator.save_final.assert_called_once()
        mock_aggregator.cleanup_chunks.assert_called_once_with(chunk_paths)

    @patch('pipelines.dags.src.emotion_data_pipeline.DataAggregator')
    def test_aggregate_filtered_data_save_failure(self, MockDataAggregator):
        """Test aggregating filtered data with a save failure"""
        mock_aggregator = MockDataAggregator.return_value
        mock_aggregator.combine_chunks.return_value = (MagicMock(), MagicMock())
        mock_aggregator.save_final.return_value = False
        chunk_paths = [(Path("chunk_X.npy"), Path("chunk_y.npy"))]
        result = aggregate_filtered_data(chunk_paths)
        self.assertFalse(result)
        mock_aggregator.combine_chunks.assert_called_once_with(chunk_paths)
        mock_aggregator.save_final.assert_called_once()
        mock_aggregator.cleanup_chunks.assert_called_once_with(chunk_paths)

if __name__ == "__main__":
    unittest.main()
