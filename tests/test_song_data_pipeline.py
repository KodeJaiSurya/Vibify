import unittest
from io import StringIO
from unittest import mock
from unittest.mock import patch, mock_open
from pandas.testing import assert_frame_equal
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
from unittest.mock import MagicMock
import os
import sys

# Adding the project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

# importing the functions
from pipelines.dags.src.song_data_pipeline import save_features, load_song_data, data_cleaning, scale_features

class TestLoadSongData(unittest.TestCase):

    @patch("gdown.download")
    @patch("pandas.read_csv")
    def test_success(self, mock_read_csv, mock_gdown):
        """Testing for valid file"""
        mock_gdown.return_value = "dags/data/raw/song_dataset.csv"
        sample_data = pd.DataFrame({"col1": [1, 2], "col2": [3, 4]})
        mock_read_csv.return_value = sample_data
        file_id = "valid_file_id"
        result = load_song_data(file_id)
        mock_gdown.assert_called_once_with(f"https://drive.google.com/uc?id={file_id}", "dags/data/raw/song_dataset.csv", quiet=False)
        self.assertTrue(result.equals(sample_data))

    @patch("gdown.download", side_effect=FileNotFoundError)
    def test_file_not_found(self, mock_gdown):
        """Test for invalid file"""
        file_id = "invalid_file_id"
        result = load_song_data(file_id)
        self.assertIsNone(result)

    def test_missing_argument(self):
      """Test for missing required argument, fileID"""
      with self.assertRaises(TypeError):
        load_song_data()

class TestDataCleaning(unittest.TestCase):
    def test_missing_values(self):
        """testing removal of missing values"""
        data = {
            'song_name': ['Song A', None, 'Song B'],
            'uri': ['uri1', 'uri2', 'uri3'],
            'danceability': [0.5, 0.6, 0.7],
            'energy': [0.8, 0.9, 0.7],
            'key': [1, 2, 3],
            'loudness': [-5, -6, -4],
            'mode': [1, 0, 1],
            'speechiness': [0.05, 0.03, 0.04],
            'acousticness': [0.2, 0.3, 0.5],
            'instrumentalness': [0.0, 0.0, 0.0],
            'liveness': [0.1, 0.2, 0.3],
            'valence': [0.6, 0.7, 0.8],
            'tempo': [120, 130, 140],
            'genre': ['Pop', 'Rock', 'Jazz']
        }
        df = pd.DataFrame(data)
        cleaned_df = data_cleaning(df)
        expected_data = {
            'danceability': [0.5, 0.7],
            'energy': [0.8, 0.7],
            'key': [1, 3],
            'loudness': [-5, -4],
            'mode': [1, 1],
            'speechiness': [0.05, 0.04],
            'acousticness': [0.2, 0.5],
            'instrumentalness': [0.0, 0.0],
            'liveness': [0.1, 0.3],
            'valence': [0.6, 0.8],
            'tempo': [120, 140],
            'uri': ['uri1', 'uri3'],
            'genre': ['Pop', 'Jazz'],
            'song_name': ['Song A', 'Song B']
        }
        expected_df = pd.DataFrame(expected_data)
        pd.testing.assert_frame_equal(cleaned_df.reset_index(drop=True), expected_df.reset_index(drop=True))

    def test_duplicates(self):
        """test removal of duplicate values"""
        data = {
            'song_name': ['Song A', 'Song A', 'Song B'],
            'uri': ['uri1', 'uri1', 'uri2'],
            'danceability': [0.5, 0.5, 0.7],
            'energy': [0.8, 0.8, 0.7],
            'key': [1, 1, 3],
            'loudness': [-5, -5, -4],
            'mode': [1, 1, 1],
            'speechiness': [0.05, 0.05, 0.04],
            'acousticness': [0.2, 0.2, 0.5],
            'instrumentalness': [0.0, 0.0, 0.0],
            'liveness': [0.1, 0.1, 0.3],
            'valence': [0.6, 0.6, 0.8],
            'tempo': [120, 120, 140],
            'genre': ['Pop', 'Pop', 'Jazz']
        }
        df = pd.DataFrame(data)
        cleaned_df = data_cleaning(df)
        expected_data = {
            'danceability': [0.5, 0.7],
            'energy': [0.8, 0.7],
            'key': [1, 3],
            'loudness': [-5, -4],
            'mode': [1, 1],
            'speechiness': [0.05, 0.04],
            'acousticness': [0.2, 0.5],
            'instrumentalness': [0.0, 0.0],
            'liveness': [0.1, 0.3],
            'valence': [0.6, 0.8],
            'tempo': [120, 140],
            'uri': ['uri1', 'uri2'],
            'genre': ['Pop', 'Jazz'],
            'song_name': ['Song A', 'Song B']
        }
        expected_df = pd.DataFrame(expected_data)
        pd.testing.assert_frame_equal(cleaned_df.reset_index(drop=True), expected_df.reset_index(drop=True))

    def test_no_valid_data(self):
        """Test invalid data"""
        data = {
            'song_name': [None, None],
            'uri': [None, None],
            'danceability': [None, None],
            'energy': [None, None],
            'key': [None, None],
            'loudness': [None, None],
            'mode': [None, None],
            'speechiness': [None, None],
            'acousticness': [None, None],
            'instrumentalness': [None, None],
            'liveness': [None, None],
            'valence': [None, None],
            'tempo': [None, None],
            'genre': [None, None]
        }
        df = pd.DataFrame(data)
        cleaned_df = data_cleaning(df)
        self.assertTrue(cleaned_df.empty)

    def test_empty_dataframe(self):
        """Test empty dataframe"""
        df = pd.DataFrame(columns=['song_name', 'uri', 'danceability', 'energy',
                                   'key', 'loudness', 'mode', 'speechiness',
                                   'acousticness', 'instrumentalness',
                                   'liveness', 'valence', 'tempo', 'genre'])
        cleaned_df = data_cleaning(df)
        self.assertTrue(cleaned_df.empty)

    def test_with_correct_columns(self):
        """Testing data cleaning with proper data dataframe"""
        data = {
            'song_name': ['Song A', 'Song B'],
            'uri': ['uri1', 'uri2'],
            'danceability': [0.5, 0.6],
            'energy': [0.8, 0.9],
            'key': [1, 2],
            'loudness': [-5, -6],
            'mode': [1, 0],
            'speechiness': [0.05, 0.03],
            'acousticness': [0.2, 0.3],
            'instrumentalness': [0.0, 0.0],
            'liveness': [0.1, 0.2],
            'valence': [0.6, 0.7],
            'tempo': [120, 130],
            'genre': ['Pop', 'Rock']
        }
        df = pd.DataFrame(data)
        cleaned_df = data_cleaning(df)
        expected_columns = ['danceability', 'energy', 'key', 'loudness',
                            'mode', 'speechiness', 'acousticness',
                            'instrumentalness', 'liveness', 'valence',
                            'tempo', 'uri', 'genre', 'song_name']
        self.assertListEqual(list(cleaned_df.columns), expected_columns)

    def test_missing_argument(self):
      """Test for missing required argument, dataframe"""
      with self.assertRaises(TypeError):
        data_cleaning()

class TestScaleFeatures(unittest.TestCase):

    def test_with_numerical_data(self):
        """Testing with only numerical data"""
        df = pd.DataFrame({'feature1': [1, 2, 3], 'feature2': [4, 5, 6]})
        scaled_df = scale_features(df)
        # expected output after scaling
        scaler = StandardScaler()
        expected_data = scaler.fit_transform(df[['feature1', 'feature2']])
        expected_df = pd.DataFrame(expected_data, columns=['feature1', 'feature2'])
        assert_frame_equal(scaled_df, expected_df)

    def test_mixed_data(self):
        """Testing with mixed data"""
        df = pd.DataFrame({
            'feature1': [1, 2, 3],
            'feature2': [4, 5, 6],
            'non_numeric': ['a', 'b', 'c']
        })
        scaled_df = scale_features(df)
        scaler = StandardScaler()
        expected_data = scaler.fit_transform(df[['feature1', 'feature2']])
        expected_df = pd.DataFrame(expected_data, columns=['feature1', 'feature2'])
        assert_frame_equal(scaled_df, expected_df)

    def test_empty_dataframe(self):
        """Testing with empty dataframe"""
        df = pd.DataFrame(columns=['feature1', 'feature2'])
        with self.assertRaises(ValueError):
            scale_features(df)

    def test_one_row(self):
        """Testing with one row of data"""
        df = pd.DataFrame({'feature1': [1], 'feature2': [4]})
        scaled_df = scale_features(df)
        expected_df = pd.DataFrame({'feature1': [0.0], 'feature2': [0.0]})
        assert_frame_equal(scaled_df, expected_df)

class TestSaveFeatures(unittest.TestCase):
    
    @patch("os.makedirs")
    @patch("pandas.DataFrame.to_csv")
    def test_successful_save(self, mock_to_csv, mock_makedirs):
        """Test saving features successfully"""
        # Mock inputs
        df = pd.DataFrame({"col1": [1, 2, 3], "col2": [4, 5, 6]})
        output_dir = "test_data/preprocessed"
        save_features(df, output_dir)
        mock_makedirs.assert_called_once_with(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "preprocessed_features.csv")
        mock_to_csv.assert_called_once_with(output_path, index=False)
    
    @patch("os.makedirs", side_effect=PermissionError)
    def test_permission_error(self, mock_makedirs):
        """Test handling of PermissionError when creating directories"""
        df = pd.DataFrame({"col1": [1, 2, 3], "col2": [4, 5, 6]})
        with self.assertRaises(PermissionError):
            save_features(df, "restricted_dir")

    @patch("os.makedirs")
    @patch("pandas.DataFrame.to_csv", side_effect=Exception("Save failed"))
    def test_csv_exception(self, mock_to_csv, mock_makedirs):
        """Test handling of exception when saving the dataframe"""
        df = pd.DataFrame({"col1": [1, 2, 3], "col2": [4, 5, 6]})
        with self.assertRaises(Exception) as context:
            save_features(df, "test_data/preprocessed")
        self.assertEqual(str(context.exception), "Save failed")

if __name__ == "__main__":
    unittest.main()
