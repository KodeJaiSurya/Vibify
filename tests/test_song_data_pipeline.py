import unittest
from io import StringIO
from unittest import mock
from unittest.mock import patch, mock_open
from pandas.testing import assert_frame_equal
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
import logging
from unittest.mock import MagicMock
import os
import sys

# Adding the project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

# importing the functions
from pipelines.dags.src.song_data_pipeline import save_features, load_song_data, data_cleaning, scale_features

class TestLoadSongData(unittest.TestCase):

    @patch("pipelines.dags.src.song_data_pipeline.DVCWrapper")  
    @patch("pandas.DataFrame.to_csv") 
    @patch("google.cloud.storage.Client")  # Mock GCS Client
    def test_load_song_data_success(self, mock_storage_client, mock_to_csv, mock_dvc_wrapper):
        """Test successful data loading from GCS"""
        mock_bucket = MagicMock()
        mock_blob = MagicMock()
        mock_blob.download_as_string.return_value = b"col1,col2\n1,3\n2,4"  # Mock CSV data
        mock_bucket.blob.return_value = mock_blob
        mock_storage_client.return_value.get_bucket.return_value = mock_bucket
        # test inputs
        bucket_name = "test_bucket"
        blob_name = "test_blob.csv"
        result = load_song_data(bucket_name, blob_name)
        expected_df = pd.DataFrame({"col1": [1, 2], "col2": [3, 4]})
        pd.testing.assert_frame_equal(result, expected_df)
        mock_storage_client.return_value.get_bucket.assert_called_once_with(bucket_name)
        mock_bucket.blob.assert_called_once_with(blob_name)
        mock_to_csv.assert_called_once() 
        mock_dvc_wrapper.assert_called_once_with(bucket_name)  # Verify DVCWrapper initialization


    @patch("google.cloud.storage.Client")
    def test_file_not_found(self, mock_storage_client):
        """Test file not found on GCS"""
        mock_bucket = MagicMock()
        mock_blob = MagicMock()
        mock_blob.download_as_string.side_effect = FileNotFoundError("File not found")
        mock_bucket.blob.return_value = mock_blob
        mock_storage_client.return_value.get_bucket.return_value = mock_bucket
        bucket_name = "test_bucket"
        blob_name = "nonexistent_blob.csv"
        result = load_song_data(bucket_name, blob_name)
        # Asserting results
        self.assertIsNone(result)
        mock_storage_client.return_value.get_bucket.assert_called_once_with(bucket_name)
        mock_bucket.blob.assert_called_once_with(blob_name)

    @patch("google.cloud.storage.Client")
    def test_missing_argument(self, mock_storage_client):
        """Test for missing required arguments"""
        with self.assertRaises(TypeError):
            load_song_data()  # No arguments provided to test this


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
        expected_df['non_numeric'] = df['non_numeric'] 
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
    
    @patch("google.cloud.storage.Client")
    def test_successful_save(self, mock_storage_client):
        """Test saving features successfully to GCS"""
        # Mocking GCS client and blob behavior
        mock_bucket = MagicMock()
        mock_blob = MagicMock()
        mock_storage_client.return_value.get_bucket.return_value = mock_bucket
        mock_bucket.blob.return_value = mock_blob
        # Mock DataFrame to CSV conversion
        df = pd.DataFrame({"col1": [1, 2, 3], "col2": [4, 5, 6]})
        bucket_name = "test_bucket"
        blob_name = "test_blob.csv"
        save_features(df, bucket_name, blob_name)
        mock_storage_client.return_value.get_bucket.assert_called_once_with(bucket_name)
        mock_bucket.blob.assert_called_once_with(blob_name)
        mock_blob.upload_from_string.assert_called_once()

    @patch('google.cloud.storage.Client')
    @patch('pipelines.dags.src.song_data_pipeline.logger')
    def test_save_features_exception(self, mock_logger, mock_storage_client):
        df = pd.DataFrame({'column1': [1, 2, 3], 'column2': ['a', 'b', 'c']})
        # Simulating an exception when getting the bucket
        mock_gcs_client = MagicMock()
        mock_storage_client.return_value = mock_gcs_client
        mock_gcs_client.get_bucket.side_effect = Exception("Bucket not found")
        # Running the function and assert it logs the error
        save_features(df, 'test-bucket', 'test-path')
        # Checking that the logger called error to indicate failure
        mock_logger.error.assert_called_once_with('Error saving data to GCS: Bucket not found')
    
    @patch('google.cloud.storage.Client')
    @patch('pipelines.dags.src.song_data_pipeline.logger')
    def test_save_features_empty_dataframe(self, mock_logger, mock_storage_client):
        df = pd.DataFrame()
        # Mock GCS storage client and bucket
        mock_gcs_client = MagicMock()
        mock_storage_client.return_value = mock_gcs_client
        mock_bucket = MagicMock()
        mock_gcs_client.get_bucket.return_value = mock_bucket
        mock_blob = MagicMock()
        mock_bucket.blob.return_value = mock_blob
        save_features(df, 'test-bucket', 'test-path')
        # Asserting that the GCS client methods were called even for empty DataFrame
        mock_storage_client.assert_called_once()
        mock_gcs_client.get_bucket.assert_called_once_with('test-bucket')
        mock_bucket.blob.assert_called_once_with('test-path')
        mock_blob.upload_from_string.assert_called_once()

if __name__ == "__main__":
    unittest.main()
