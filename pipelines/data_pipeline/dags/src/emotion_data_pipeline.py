from pathlib import Path
from typing import List, Tuple

from .emotion_data_downloader import DataDownloader
from .emotion_data_processor import DataProcessor
from .emotion_data_aggregator import DataAggregator

def download_emotion_data(file_id: str) -> str:
    """Task to download emotion data"""
    downloader = DataDownloader()
    return downloader.download_from_gdrive(file_id)

def process_emotion_data(file_path: str, chunk_size: int = 1000) -> List[Tuple[Path, Path]]:
    """Task to process emotion data in chunks"""
    processor = DataProcessor(chunk_size=chunk_size)
    return processor.process_all_chunks(file_path)

def aggregate_filtered_data(chunk_paths: List[Tuple[Path, Path]]) -> bool:
    """Task to combine chunks and save final data"""
    aggregator = DataAggregator()
    X, y = aggregator.combine_chunks(chunk_paths)
    success = aggregator.save_final(X, y)
    aggregator.cleanup_chunks(chunk_paths)
    return success