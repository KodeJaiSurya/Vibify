import unittest
from unittest.mock import patch, MagicMock
from pathlib import Path
import pandas as pd
import numpy as np
import os
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

from pipelines.dags.src.emotion_data_processor import DataProcessor

class TestDataProcessor(unittest.TestCase):

    def setUp(self):
        """Setting up a DataProcessor instance and sample data"""
        self.processor = DataProcessor()
        # Providing 2304 values for the pixel data
        pixel_data = " ".join([str(i % 256) for i in range(2304)])
        self.sample_data = pd.DataFrame({
            'emotion': [0, 3, 4, 6, 0, 3],  
            'pixels': [pixel_data] * 6
        })

    def test_initialization(self):
        """Test if the DataProcessor initializes correctly"""
        self.assertEqual(self.processor.selected_emotions, [0, 3, 4, 6])
        self.assertEqual(self.processor.image_width, 48)
        self.assertEqual(self.processor.image_height, 48)
        self.assertTrue(self.processor.temp_dir.exists())

    def test_filter_emotions_valid(self):
        """Test filtering with valid emotions"""
        filtered_data = self.processor.filter_emotions(self.sample_data)
        self.assertTrue(filtered_data['emotion'].isin([0, 3, 4, 6]).all())
        self.assertEqual(filtered_data.shape[0], 6)

    def test_filter_emotions_no_match(self):
        """Test filtering with no matching emotions"""
        data = pd.DataFrame({'emotion': [7, 8, 9], 'pixels': ["1 2 3 4"] * 3})
        filtered_data = self.processor.filter_emotions(data)
        self.assertTrue(filtered_data.empty)

    def test_filter_emotions_missing_column(self):
        """Test filtering with missing 'emotion' column"""
        data = pd.DataFrame({'pixels': ["1 2 3 4"]})
        with self.assertRaises(KeyError):
            self.processor.filter_emotions(data)

    def test_map_emotions_valid(self):
        """Test mapping emotions with a valid map"""
        mapped_data = self.processor.map_emotions(self.sample_data)
        self.assertTrue(mapped_data['emotion'].isin([0, 1, 2, 3]).all())

    def test_preprocess_pixels_valid(self):
        """Test pixel preprocessing with valid data"""
        processed_pixels = self.processor.preprocess_pixels(self.sample_data)
        self.assertEqual(processed_pixels.shape, (6, 48, 48, 1))
        self.assertTrue(np.all((processed_pixels >= 0) & (processed_pixels <= 1)))

    def test_preprocess_pixels_malformed_data(self):
        """Test pixel preprocessing with malformed data"""
        data = pd.DataFrame({'emotion': [0], 'pixels': ["malformed data"]})
        with self.assertRaises(ValueError):
            self.processor.preprocess_pixels(data)

    def test_preprocess_labels_valid(self):
        """Test label preprocessing with valid data"""
        labels = self.processor.preprocess_labels(self.sample_data)
        self.assertTrue(np.array_equal(labels, [0, 3, 4, 6, 0, 3]))

    def test_process_chunk_success(self):
        """Test successful chunk processing"""
        X, y = self.processor.process_chunk(self.sample_data)
        self.assertIsNotNone(X)
        self.assertIsNotNone(y)
        self.assertEqual(X.shape, (6, 48, 48, 1))
        self.assertEqual(y.shape, (6,))

    def test_process_chunk_empty(self):
        """Test processing an empty chunk"""
        empty_data = pd.DataFrame(columns=['emotion', 'pixels'])
        X, y = self.processor.process_chunk(empty_data)
        self.assertIsNone(X)
        self.assertIsNone(y)

    @patch('pipelines.dags.src.emotion_data_processor.np.save')
    def test_save_chunk(self, mock_save):
        """Test saving a processed chunk"""
        X = np.random.rand(6, 48, 48, 1)
        y = np.array([0, 1, 2, 3, 4, 5])
        X_path, y_path = self.processor.save_chunk(X, y, 0)
        self.assertTrue(mock_save.called)
        self.assertEqual(X_path, self.processor.temp_dir / "X_chunk_0.npy")
        self.assertEqual(y_path, self.processor.temp_dir / "y_chunk_0.npy")

    @patch('pipelines.dags.src.emotion_data_processor.pd.read_csv')
    def test_process_all_chunks(self, mock_read_csv):
        """Test processing all chunks from a file"""
        mock_read_csv.return_value = [self.sample_data] 
        chunk_paths = self.processor.process_all_chunks("dummy_path.csv")
        self.assertEqual(len(chunk_paths), 1)

if __name__ == "__main__":
    unittest.main()
