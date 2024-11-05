import unittest
from unittest.mock import patch, MagicMock
import numpy as np
import os
from pathlib import Path
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

from pipelines.data_pipeline.dags.src.emotion_data_aggregator import DataAggregator


class TestDataAggregator(unittest.TestCase):

    def setUp(self):
        """Setting up a DataAggregator instance and necessary paths"""
        self.aggregator = DataAggregator(temp_dir="test_temp_chunks")
        self.chunk_paths = [
            (Path("test_temp_chunks/X_chunk_0.npy"), Path("test_temp_chunks/y_chunk_0.npy")),
            (Path("test_temp_chunks/X_chunk_1.npy"), Path("test_temp_chunks/y_chunk_1.npy"))
        ]
        # temporary test data
        os.makedirs("test_temp_chunks", exist_ok=True)
        np.save(self.chunk_paths[0][0], np.random.rand(5, 48, 48, 1))  # X_chunk_0
        np.save(self.chunk_paths[0][1], np.array([0, 1, 2, 3, 4]))  # y_chunk_0
        np.save(self.chunk_paths[1][0], np.random.rand(5, 48, 48, 1))  # X_chunk_1
        np.save(self.chunk_paths[1][1], np.array([5, 6, 7, 8, 9]))  # y_chunk_1

    def tearDown(self):
        """Cleanup the temporary test files"""
        for x_path, y_path in self.chunk_paths:
            if x_path.exists():
                os.remove(x_path)
            if y_path.exists():
                os.remove(y_path)
        if Path("test_temp_chunks").exists():
            os.rmdir("test_temp_chunks")

    def test_combine_chunks(self):
        """Test combining chunk files into final arrays"""
        X, y = self.aggregator.combine_chunks(self.chunk_paths)
        self.assertEqual(X.shape, (10, 48, 48, 1))  # 10 samples from two chunks
        self.assertEqual(y.shape, (10,))  # 10 labels from two chunks
        self.assertTrue(np.array_equal(y, np.arange(10)))  # Labels should be [0, 1, 2, ..., 9]

    @patch('os.remove') 
    def test_cleanup_chunks(self, mock_remove):
        """Test cleanup of temporary chunk files"""
        self.aggregator.cleanup_chunks(self.chunk_paths)
        self.assertTrue(mock_remove.called)
        self.assertEqual(mock_remove.call_count, len(self.chunk_paths) * 2) 

    @patch('numpy.save') 
    @patch('os.makedirs')  
    def test_save_final(self, mock_makedirs, mock_save):
        """Test saving final processed arrays"""
        X = np.random.rand(10, 48, 48, 1)
        y = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        result = self.aggregator.save_final(X, y)
        self.assertTrue(result)
        self.assertTrue(mock_save.called)
        self.assertEqual(mock_save.call_count, 2)  # Two save calls for X and y

    @patch('numpy.save', side_effect=Exception("Save error"))
    def test_save_final_error(self, mock_save):
        """Test saving final processed arrays with error handling"""
        X = np.random.rand(10, 48, 48, 1)
        y = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        result = self.aggregator.save_final(X, y)
        self.assertFalse(result)

if __name__ == "__main__":
    unittest.main()
